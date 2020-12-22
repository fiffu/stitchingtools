"""
Combine raw frames into animations. Can output in gif, apng, webm.

Usage: edit the SCORE string below then call:
    $ python3 mkwebm.py

* Outputs go directly into the working dir.
* A `temp` folder is created to store intermediate knitted frames and can be
  deleted after the script completes.

____Usage (extended)____
You define a score, each containing CueSheets separated by a blank line.
CueSheets `start` an animation from idx 0 and `halt` at the specified frame.
(specifying "30, halt" means the last frame should be idx 29)

Each CueSheet specifies `stack` of frame components that can be "knitted" 
(composited) into a single complete frame before it is "composed" into the 
output image/video.

Each component is defined by a filename, frame_range, and LoopType.
* filename doesn't include the extension (defaulting to .png).
* frame_range uses ~ as delimiter: a01~3 => [a01, a02, a03]
* LoopType comes last. a01~11^ => [a01 ... a10, a11, a10 ... a01, a02 ...]

Various options are available for CueSheets, prefixed with /
See OPTION_TYPES below for a summary of what they do.
See also other options like FPS, OUT_FORMAT, VERBOSITY below.
"""

from glob import glob
import io
from itertools import zip_longest
import os
from os.path import exists
import re
from subprocess import run, STDOUT, DEVNULL
from typing import Tuple

from apng import APNG, PNG  # pip install apng
from PIL import Image  # pip install pillow

# typedefs
RgbTuple = Tuple[int, int, int]


FPS = 7
OUT_FORMAT = (
    # '.webm'
    # '.apng'
    '.gif'
)
VERBOSITY = 1

# Frames only have to be written to disk in case of .webm output.
# Set True to write frames anyway (warning: slow!)
FORCE_WRITE_FRAMES = False

OPTION_TYPES = {
    'rescale': float,
    'charset': str,  # 'int' or 'ascii' or 'fullwidth'
    'align': str, # [w|s|c]  (compass cardinals) (warning: slow!)
    'bgcol': RgbTuple,  # specify 255,0,0 for #ff0000 (no spaces!)
}

SYMBOLS_TO_LOOPTYPE = {
    '^': 'ABA',
    '@': 'AB',
    '>': 'ABB',
}

SCORE = """
/bgcol 50,50,50
/align w

0, scene_effects_under1~3@, scene_2-1~8, scene_effects_above1~3@
8, halt
"""



class ParseError(ValueError):
    pass

class Loopr:
    class LoopType:
        A = 'A'        # 0000
        AB = 'AB'      # 0123
        ABB = 'ABB'    # 0123333
        ABAB = 'ABAB'  # 0123 0123
        ABA = 'ABA'    # 0 123 210 123 210
        ABBA = 'ABBA'  # 0123 3210 0123 3210

    __slots__ = 'by val inverse start steps'.split()

    def __init__(self, val, by: str = 'A', start=0, steps=-1):
        self.by = by
        self.val = val
        self.inverse = False
        self.start = start
        self.steps = steps

        if hasattr(self.LoopType, by):
            pass
        else:
            ay = ''.join('AB'[x == 'A'] for x in by)
            if hasattr(self.LoopType, ay):
                self.inverse = True
                self.val = val[::-1]
                self.by = ay
            else:
                raise NotImplementedError(f'unsupported LoopType: {by}')

    def __iter__(self):
        return self.loop()

    def _loop(self):
        lty = self.LoopType  # shorthands

        if self.by == lty.A:
            while True:
                yield self.val

        if self.by == lty.AB:
            yield from self.val
            return

        if self.by == lty.ABB:
            yield from self.val
            while True:
                yield self.val[-1]

        if self.by == lty.ABAB:
            # if val=='0123' and start_idx==2, yield 23 0123 0123...
            # if wanted 343434... should just pass val=='34'
            while True:
                yield from self.val

        if self.by in [lty.ABA, lty.ABBA]:
            val, nmax = self.val, (len(self.val) - 1)

            yield self.val[0]

            n, step = 1, +1

            while True:
                yield val[n]

                # "Bounce" at start and end of iterable
                if n == 0 or n == nmax:
                    step = -step
                    # For ABBA, yield the extra "A" or "B" before stepping n
                    if self.by == lty.ABBA:
                        yield val[0] if n == 0 else val[-1]
                    else:
                        first = False
                n += step

    def loop(self, start=None, steps=None):
        start = start or self.start
        steps = steps or self.steps
        ctr = 0
        for i, val in enumerate(self._loop()):
            if i < start:
                continue
            if ctr == steps:
                break
            yield val
            ctr += 1

    def __repr__(self):
        args = ', '.join(f'{k}={getattr(self, k)}' for k in self.__slots__)
        return f'Loopr({repr(self.val)}, {args})'


class NumLoopr(Loopr):
    CHARSETS = {
        'ascii': '0123456789',
        'fullwidth': '０１２３４５６７８９'
    }

    def __init__(self, upperbound, start=0, by='AB', charset='ascii', places=0, **kwargs):
        val = list(range(start, upperbound + 1))
        if charset:
            cs = self.get_charset(charset)
            val = self.numbers(start, upperbound, places, cs)
        return super().__init__(val=val, by=by, **kwargs)

    @classmethod
    def numbers(cls, start, upper_bound_inclusive, places, charset) -> list:
        """Only works for base10"""
        val = []
        # bind globals
        int_, str_ = int, str
        charset_len = len(charset)
        for n in range(start, upper_bound_inclusive + 1):
            num = ('{:0%d}' % places).format(n)
            mapped = ''.join(charset[int_(d)] for d in num)
            val.append(mapped)
        return val

    @classmethod
    def get_charset(cls, charset):
        cs = cls.CHARSETS.get(charset)
        if cs:
            return cs
        raise NotImplementedError(f'unsupported charset: {charset}')

class LazyAccessDict:
    """Maps dotted access to dict key, returns fallback value on absent keys"""
    def __init__(self, dictionary, fallback=None):
        self._LazyAccessDict_data = dictionary
        self._LazyAccessDict_fallback = fallback
    def __getattr__(self, key):
        val = self._LazyAccessDict_data.get(key, self._LazyAccessDict_fallback)
        setattr(self, key, val)
        return val
    def __str__(self):
        return str(self._LazyAccessDict_data)



def format_columns(rendered):
    ncols = max(len(row) for row in rendered)
    longest_per_col = [0] * ncols
    for row in rendered:
        for n, token in enumerate(row):
            length = len(token) if token else 0
            if length > longest_per_col[n]:
                longest_per_col[n] = length

    formatted = []
    for row in rendered:
        line = ' - '.join(
            ('{:%d}' % width).format(token or '')
            for token, width in zip_longest(row, longest_per_col, fillvalue='')
        )
        formatted.append(line)
    return formatted


def get_frame_range(string, charset='ascii'):
    digits = r'\d'
    if charset:
        cs = ''.join(NumLoopr.get_charset(charset))
        digits = f'[{cs}]'
    found = re.search(r"""(%s+)~(%s+)([@\^\>])?$""" % (digits, digits), string)
    if not found:
        raise ParseError(f'bad syntax: expected format "name123~456" with '
                         f'optional trailing ^ or @ or >, got "{string}"')

    starts, ends, rep = found.groups()

    pad = len(starts)
    stub = string[:found.start()]

    repeat = SYMBOLS_TO_LOOPTYPE.get(rep, None)

    return stub, pad, int(starts), int(ends), repeat


def infinite_generator(value=None):
    yield from Loopr(value, by='A')



class CueSheet:
    def __init__(self, cues, max_layers, haltidx, options):
        self.cues = cues
        self.max_layers = max_layers
        self.haltidx = haltidx
        self.opt = self.load_options(options)


    @classmethod
    def load_options(self, options):
        opt = {}
        for key, typecast in OPTION_TYPES.items():
            if key in options:
                value = options[key]
                if typecast is RgbTuple:
                    opt[key] = tuple(int(n) for n in value.split(','))
                else:
                    opt[key] = typecast(value)
        return LazyAccessDict(opt)


    @classmethod
    def generate_frames(cls, frame_stub, pad, start, end, charset=None, repeat=None):
        idx = start
        template = '{}{}'
        charset = charset or 'ascii'

        loop = NumLoopr(end, by=repeat or 'AB', start=start, charset=charset, places=pad)
        for idx in loop:
            yield template.format(frame_stub, idx)

        yield from infinite_generator(None)


    @classmethod
    def layer_to_generators(cls, layertext, charset=None):
        try:
            stub, pad, start, end, rep = get_frame_range(layertext, charset)
            return cls.generate_frames(stub, pad, start, end, charset, rep)
        except ParseError:
            return infinite_generator(layertext)


    def render(self):
        score = [infinite_generator(None)] * self.max_layers
        i = 0
        while True:
            if i == self.haltidx:
                break

            if i in self.cues:
                layers = self.cues[i]
                # Update generators for each layer...
                for j, layertext in enumerate(layers):
                    if not layertext:
                        continue
                    gen = self.layer_to_generators(layertext, self.opt.charset)
                    score[j] = gen

            # Yield for this iteration, then increment
            yield [next(g) for g in score]
            i += 1


    def open_img(self, imgfile, ext='.png'):
        if not imgfile:
            return None

        if not imgfile.endswith(ext):
            imgfile += ext

        if not exists(imgfile):
            return None

        img = Image.open(imgfile).convert('RGBA')

        scale = self.opt.rescale
        if scale:
            w, h = [int(dim * scale) for dim in img.size]
            img = img.resize((w, h))

        return img


    def pad_img_to(self, img, width, height):
        bgcol = (0, 0, 0, 255 if OUT_FORMAT == '.webm' else 0)
        x, y = 0, 0
        align = self.opt.align or 'c'
        new = Image.new('RGBA', (width, height), color=bgcol)
        if align == 's':
            x = width // 2 - (img.width // 2)
            y = height - img.height
        elif align == 'c':
            x = width // 2 - (img.width // 2)
            y = height // 2 - (img.height // 2)
        elif align == 'w':
            x = 0
            y = height // 2 - (img.height // 2)
        new.paste(img, box=(x, y), mask=img)
        return new


    def knit(self, *layers, pad_dimensions=None):
        images = list(filter(None, [self.open_img(lyr) for lyr in layers]))
        if pad_dimensions:
            images = [self.pad_img_to(img, *pad_dimensions) for img in images]
        if self.opt.bgcol:
            dim = images[0].size
            images.insert(0, Image.new('RGBA', dim, color=self.opt.bgcol))
        base = images[0]

        for img in images[1:]:
            base.alpha_composite(img)

        return base


    def compose(self, outname, fps=FPS, force_write_frames=FORCE_WRITE_FRAMES):
        fmt = 'temp/%05d.png'

        if exists('temp'):
            for file in glob('temp/*'):
                os.remove(file)
        else:
            os.mkdir('temp')

        frames = []
        filenames = []
        rendered = list(self.render())
        formatted = format_columns(rendered)

        pad = []
        if self.opt.align:
            pad = [0, 0]
            for layers in rendered:
                images = list(filter(None, [self.open_img(lyr) for lyr in layers]))
                w = max(img.width for img in images) if images else 0
                h = max(img.height for img in images) if images else 0
                if w > pad[0]:
                    pad[0] = w
                if h > pad[1]:
                    pad[1] = h

        for i, layers in enumerate(rendered):
            if VERBOSITY > 0:
                print(f'{i:>3} | {formatted[i]} | {i:>3}')

            frame = self.knit(*layers, pad_dimensions=pad)
            frames.append(frame)

        for i, frame in enumerate(frames):
            if force_write_frames or outname.endswith('.webm'):
                filename = fmt % i
                frame.save(filename)

        if outname.endswith('.webm'):
            self.make_webm(fmt, fps, outname)
        elif outname.endswith('.apng'):
            self.make_apng(fmt, fps, outname, frames)
        elif outname.endswith('.gif'):
            self.make_gif(fmt, fps, outname, frames)


        if VERBOSITY > 0:
            print('---')


    @staticmethod
    def make_webm(fmt, fps, outname):
        args = [
            'ffmpeg'
            ,'-f'
            ,'image2'
            ,'-r'
            ,str(fps)  # input framerate
            ,'-i'
            ,fmt
            ,'-c:v'
            ,'libvpx-vp9'
            ,'-g'
            ,'1'
            ,'-lossless'
            ,'1'
            ,'-rc_lookahead'
            ,'1'
            ,'-r'
            ,str(fps)
            ,outname
            ,'-y'  # overwrite output
        ]
        run(args, stdout=DEVNULL, stderr=DEVNULL)


    @staticmethod
    def make_apng(fmt, fps, outname, frames):
        def pil_to_apng(pil_frame):
            with io.BytesIO() as buf:
                pil_frame.save(buf, 'PNG', optimize=True)
                return PNG.from_bytes(buf.getvalue())

        interval_ms = 1000 // fps

        img = APNG()
        for frame in map(pil_to_apng, frames):
            img.append(frame, delay=interval_ms)

        img.save(outname)


    @staticmethod
    def make_gif(fmt, fps, outname, frames):
        interval_ms = 1000 // fps
        frames[0].save(outname,
                       save_all=True,
                       append_images=frames[1:],
                       optimize=False,
                       duration=interval_ms,
                       loop=0)



class Parser:
    commentchar = '#'
    optionchar = '/'
    delim = ','
    inf_gen = infinite_generator()


    def __init__(self, text, delim=None, commentchar=None):
        self.commentchar = commentchar or self.commentchar
        self.delim = delim or self.delim

        self.stacks = self.read(text)


    @classmethod
    def read_file(cls, file, encoding='utf8', **kwargs):
        with open(file, mode, encoding) as f:
            text = f.read()
            return cls(text, **kwargs)

    @classmethod
    def read(cls, text):
        stacks = []
        buffer = []

        for line in text.splitlines():
            # Flush buffer to stacks[] on empty line
            if not line.strip():
                if buffer:
                    stacks.append(buffer.copy())
                    buffer = []

            info, *comment = line.split(cls.commentchar, 1)
            info = info.strip()
            if info:
                buffer.append(info)

        # At the end of all the lines, flush again as needed
        if buffer:
            stacks.append(buffer.copy())

        return stacks


    @classmethod
    def parse_option(cls, line):
        line = line[len(cls.optionchar):]
        opt, *optval = line.split()
        if len(optval) == 1:
            return {opt: optval[0]}
        return {}


    def parse_stack(self, stacklines):
        cues = {
            'options': {}
        }
        for line in stacklines:
            if line.startswith(self.optionchar):
                option = self.parse_option(line)
                if option:
                    cues['options'].update(option)
                    continue
                else:
                    raise ParseError(f'invalid option syntax: "line"')

            idx, *layers = line.split(self.delim)
            idx = eval(idx)
            layers = [lyr.strip() for lyr in layers]
            cues[idx] = layers

        max_layers = max(map(len, cues.values()))
        return cues, max_layers


    def parse(self):
        cue_sheets = []
        options = {}
        for stacklines in self.stacks:
            cues, max_layers = self.parse_stack(stacklines)
            if cues['options']:
                options.update(cues['options'])
                continue

            if not 0 in cues:
                raise ParseError(f'missing frame 0 for "{stacklines[0]}"')

            halt = [k for k, v in cues.items() if 'halt' in v]
            if not halt:
                raise ParseError(f'missing halt point for "{stacklines[0]}"')

            haltidx = max(halt)

            cs = CueSheet(cues, max_layers, haltidx, options)
            cue_sheets.append(cs)
        return cue_sheets



if __name__ == '__main__':
    p = Parser(SCORE)
    for i, cs in enumerate(p.parse()):
        cs.compose(f'{i + 1:>03}{OUT_FORMAT}')
