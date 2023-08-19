"""
Quick and dirty image composition script.
Requires the `pillow` package (PIL fork) for image i/o.
Requires the blend_mode and numpy packages for multiply blend modes.

Usage:
    python _merge.py -h

Steps:
    1. Ensure you are using Python 3.
    2. Move this file to the same dir as the image files you want composited.
    3. Define layer files to be composited in the input file (_info.txt by
       default). The layers will be stacked in the order they appear in the
       input file (preceding files overlay subsequent files).
    4. Add a blank line to start a new stack.
    5. Take note of comments and multiply-mode notation (see docstrings below).
    6. Review optional args for running this script:
         python _merge.py -h
    7. Composited files are placed in the output directory (default: 'buffer').
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import os
import threading

from blend_modes import multiply   # pip install blend_modes
import numpy                       # pip install numpy (blend_modes depends on matrices)
from PIL import Image              # pip install Pillow
from rich.progress import Progress # pip install rich


INFO = """

CG_01_10
CG_01_00

CG_01_11
CG_01_01

"""




DEFAULT_CACHE_FILE = '._merge'

progress = Progress()
console = progress.console


class BlendMode(Enum):
    """Enumerates available blend modes"""
    ADD = 1
    MULTIPLY = 2


@dataclass
class LayerSpec:
    file: str
    mode: BlendMode
    offset: tuple[int, int]
    comment: str

    def hash(self) -> str:
        return f'{self.file}{self.mode}{self.offset}{self.comment}'


@dataclass
class Stack:
    layers: list[LayerSpec]

    @staticmethod
    def make_filename(args, i) -> str:
        template: str = '{prefix}{i:0>%d}{suffix}.png' % (args.digits)
        return template.format(prefix=args.prefix,
                               i=i,
                               suffix=args.suffix)

    def hash(self, args, i) -> str:
        filename = self.make_filename(args, i)
        layers_hash = ' || '.join(
            layer.hash()
            for layer in self.layers
        )
        return f'{filename} {layers_hash}'

class Parser:
    # Specify the file extension of the input images expected by this script so you
    # don't have to repeat in in the input file.
    EXT = '.png'

    # Works just like a Python comment. Inline within a line of data is okay.
    COMMENT_PREFIX = '#'

    # Place this immediately after a filename to indicate this layer should be
    # composited using a multiply blend mode instead of a regular overlap. It's
    # called a suffix but you can put this symbol anywhere in the filename.
    MULTIPLY_SUFFIX = '*'

    PASTE_POSITION = '@'
    PASTE_POSITION_DELIM = ','

    @classmethod
    def parse_line(cls, line: str) -> LayerSpec:
        # Remove inline comments, if any
        line, *comment = line.split(cls.COMMENT_PREFIX, 1)
        comment = comment[0] if comment else ''
        line = line.strip()

        # Blend mode
        mode = BlendMode.ADD
        if cls.MULTIPLY_SUFFIX in line:
            mode = BlendMode.MULTIPLY
            line = line.replace(cls.MULTIPLY_SUFFIX, '')

        # Find offset in form @x,y
        file, *offset = line.split(cls.PASTE_POSITION, 1)
        file = file.strip()

        # Normalize offset to tuple[int, int]
        if offset:
            offset = offset[0].split(cls.PASTE_POSITION_DELIM, 1)
            offset = tuple(map(int, offset))
        else:
            offset = ()

        return LayerSpec(
            file=file + cls.EXT if file else '',
            mode=mode,
            offset=offset,
            comment=comment,
        )
    
    @classmethod
    def parse(cls, args, lines: str) -> list[Stack]:
        """
        Reads input file and parses into stacks.
        
        Each stack is a list of layers, so the output is list[list[Layer]].
        """
        lines: list[str] = INFO.splitlines()

        # Parse lines into layers
        layers: list[LayerSpec] = []
        # Flush groups of layers into a list of lists
        layers_list: list[list[LayerSpec]] = []

        for line in lines:
            parsed = Parser.parse_line(line)

            # Non-empty line: Add to buffer
            if parsed.file:
                layers.append(parsed)
                continue
            # Lines with comments only: avoid flushing the buffer
            elif parsed.comment:
                continue

            # Empty line: flush buffered layers to output
            if layers:
                layers_list.append(list.copy(layers))
                layers = []

            # Nothing in stack due to multiple empty lines: do nothing
            else:
                continue

        # Flush one last time
        if layers:
            layers_list.append(list.copy(layers))

        # Convert [][]layer -> []Stack
        return [Stack(layers=layers) for layers in layers_list]


class Composer:
    @classmethod
    def compose(cls, args, i, stack: Stack):
        """
        Merges the input stack of filenames

        fn_list is a list of filenames representing a group of layers.
        i is the ordinal of this current layer group in the current script run.
        """
        # Determine filename
        out_fn = stack.make_filename(args, i)

        # In input, lines are ordered in the order they're stacked, but when
        # iterating the file, we encounter the layers top-first.
        # Reverse so we start merging the stack from the bottom layer.
        layers_bot_first = stack.layers[::-1]

        # Prep first layer for compositing
        spec = layers_bot_first.pop(0)
        base = Image.open(spec.file).convert('RGBA')

        # Merge remaining layers into first layer
        for spec in layers_bot_first:
            do_multiply = spec.mode == BlendMode.MULTIPLY

            layer = Image.open(spec.file).convert('RGBA')

            if do_multiply and spec.offset:
                console.log(f'{spec.file}: multiply and offset not supported, skipping layer')
                continue

            if spec.offset:
                console.log(f'pasting {spec.file} @ {spec.offset}')
                # New paste - build empty layer and composite onto base
                newlayer = Image.new(mode='RGBA', size=base.size)
                newlayer.paste(layer, spec.offset)
                base.alpha_composite(newlayer)

            elif do_multiply:
                console.log(f'multiplying {spec.file}')
                base = cls.blend_multiply(base, layer)

            else:
                base.alpha_composite(layer)

        # Write
        out_path = os.path.join(args.outdir, out_fn)
        console.log(f'Writing output file: {out_path}')
        base.save(out_path)
        return out_path

    @staticmethod
    def blend_multiply(base: Image, new: Image, opacity: float = 1) -> Image:
        base_f = numpy.array(base).astype(float)
        new_f = numpy.array(new).astype(float)
        blended_f = multiply(base_f, new_f, opacity)
        return Image.fromarray(numpy.uint8(blended_f))


class OverwritingCache:
    def __init__(self, cache_file):
        self._file: str = cache_file
        self._buffer: set[str] = set()
        self._initial: set[str] = self._load_initial()

    @classmethod
    @contextmanager
    def load(cls, filename: str = DEFAULT_CACHE_FILE):
        cache = cls(filename)
        console.log(f'Loaded cache: {filename}')

        err = None
        try:
            yield cache
        
        except BaseException as exc:
            err = exc

        finally:
            # Don't overwrite the existing cache if there was an error
            if err:
                console.log('Encountered exception, skipping cache flush')
                raise err
            else:
                cache._flush()

    def _load_initial(self) -> set[str]:
        if os.path.exists(self._file):
            with open(self._file, 'r', encoding='utf8') as f:
                return {line.strip() for line in f}
        return set()

    def _flush(self):
        with open(self._file, 'w', encoding='utf8') as f:
            f.write('\n'.join(self._buffer))

    def peek(self, hashed: str) -> bool:
        return hashed in self._initial

    def buffer(self, hashed: str):
        self._buffer.add(hashed)
    
    def diff(self):
        return self._buffer.difference(self._initial)


class ResultThread(threading.Thread):
    def __init__(self, *a, **kwargs):
        self._result = None

        target = kwargs.get('target')
        if target:
            kwargs['target'] = self.wrap(target)

        super().__init__(*a, **kwargs)

    @property
    def result(self):
        return self._result

    def start(self):
        super().start()

    def wrap(self, func):
        def wrapper(*a, **k):
            res = func(*a, **k)
            self._result = res
        return wrapper


def main():
    parser = ArgumentParser()

    parser.add_argument('-C', '--ignore-cache',
                        action='store_true')

    parser.add_argument('-d', '--digits',
                        action='count',
                        default=3,
                        help='number of digits to zero-pad to; default=3 (-ddd)')

    parser.add_argument('-o', '--outdir',
                        type=str,
                        default='buffer',
                        help="folder to store output files; default='buffer'")

    parser.add_argument('-p', '--prefix',
                        type=str,
                        default='',
                        help='optional prefix for output filenames')

    parser.add_argument('-u', '--suffix',
                        type=str,
                        default='',
                        help='optional suffix for output filenames')

    args = parser.parse_args()

    # Prep dir for output
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    stacks = Parser.parse(args, INFO)

    with OverwritingCache.load() as cache:
        threads: list[ResultThread] = []

        for i, stack in enumerate(stacks):
            h = stack.hash(args, i)
            cache.buffer(h)

            if args.ignore_cache or (not cache.peek(h)):
                thread = ResultThread(target=Composer.compose, args=(args, i, stack))
                thread.start()
                threads.append(thread)

        task = progress.add_task('Composing images...', total=len(threads))
        with progress:
            for thread in threads:
                thread.join()
                progress.advance(task)

        if args.ignore_cache:
            console.log(f'New files: {len(stacks)}')
        else:
            console.log(f'New files: {len(cache.diff())}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.log('Aborted')
        pass
