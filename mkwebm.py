from enum import Enum
import io
import os
from os.path import exists
import re
import shutil
from subprocess import run, STDOUT, DEVNULL
from types import SimpleNamespace

from apng import APNG, PNG
from PIL import Image  # pip install pillow


FPS = 7
OUT_FORMAT = (
    # '.webm'
    '.apng'
    # '.gif'
)

VERBOSITY = 1
FORCE_WRITE_FRAMES = True

OPTION_TYPES = {
    'rescale': int
}


text = """
/rescale 3


0, 白部屋 下, 1-1-1.1~4
4, halt
"""

class ParseError(ValueError):
    pass

class RepeatBy(Enum):
    LOOP = 1
    PENDULUM = 2

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



def format_list(li):
    joined = ', '.join(str(x) for x in li)
    return f'[ {joined} ]'


def get_frame_range(string):
    found = re.search(r"""(\d+)~(\d+)([@\^])?$""", string)
    if not found:
        raise ParseError(f'bad syntax: expected format "name123~456" with '
                         f'optional trailing ^ or @, got "{string}"')

    starts, ends, rep = found.groups()

    pad = len(starts)
    stub = string[:found.start()]

    repeat = {
        '^': RepeatBy.PENDULUM,
        '@': RepeatBy.LOOP,
    }.get(rep, None)

    return stub, pad, int(starts), int(ends), repeat


def infinite_generator(value=None):
    while 1:
        yield value


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
                opt[key] = typecast(options[key])
        return LazyAccessDict(opt)


    @classmethod
    def generate_frames(cls, frame_stub, pad, start, end, repeat=None):
        idx = start
        template = '{}{:0%d}' % pad
        frame = template.format(frame_stub, idx)

        step = 1
        while True:
            yield frame
            idx += step
            frame = template.format(frame_stub, idx)

            if repeat == RepeatBy.LOOP:
                if idx > end:
                    idx = start
                    frame = template.format(frame_stub, idx)

            elif repeat == RepeatBy.PENDULUM:
                if idx == start or idx == end:
                    step = -step

            elif idx > end:
                break

        yield from infinite_generator(None)


    @classmethod
    def layer_to_generators(cls, layertext):
        try:
            stub, pad, start, end, rep = get_frame_range(layertext)
            return cls.generate_frames(stub, pad, start, end, rep)
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
                    gen = self.layer_to_generators(layertext)
                    score[j] = gen

            # Yield for this iteration, then increment
            yield [next(g) for g in score]
            i += 1


    def open_img(self, imgfile, ext='.png'):
        if not imgfile.endswith(ext):
            imgfile += ext

        if not exists(imgfile):
            return None

        img = Image.open(imgfile).convert('RGBA')

        scale = self.opt.rescale
        if scale:
            w, h = img.size
            img = img.resize((w * scale, h * scale))

        return img


    def knit(self, filename, *layers):
        base = self.open_img(layers[0])

        for lyr in layers[1:]:
            if not lyr:
                continue

            img = self.open_img(lyr)
            if not img:
                continue

            base.alpha_composite(img)

        if filename:
            base.save(filename)

        return base


    def compose(self, outname, fps=FPS, force_write_frames=FORCE_WRITE_FRAMES):
        fmt = 'temp/%05d.png'

        if exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        frames = []

        for i, layers in enumerate(self.render()):
            if VERBOSITY > 0:
                print(f'{i:>4} ', format_list(layers))

            filename = None  # If truthy, write frames to disk
            if force_write_frames or outname.endswith('.webm'):
                filename = fmt % i

            frame = self.knit(filename, *layers)
            frames.append(frame)

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
    p = Parser(text)
    for i, cs in enumerate(p.parse()):
        cs.compose(f'{i + 1:>03}{OUT_FORMAT}')

