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
from enum import Enum
import os
import time

from blend_modes import multiply  # pip install blend_modes
import numpy  # blend_modes depends on matrices; pip install numpy
from PIL import Image  # pip install Pillow


# Specify the file extension of the input images expected by this script so you
# don't have to repeat in in the input file.
EXT = '.png'

# Works just like a Python comment. Inline within a line of data is okay.
COMMENT_PREFIX = '#'

# Place this immediately after a filename to indicate this layer should be
# composited using a multiply blend mode instead of a regular overlap. It's
# called a suffix but you can put this symbol anywhere in the filename.
MULTIPLY_SUFFIX = '*'


class BlendMode(Enum):
    """Enumerates available blend modes"""
    ADD = 1
    MULTIPLY = 2



def blend_multiply(base: Image, new: Image, opacity: float = 1) -> Image:
    base_f = numpy.array(base).astype(float)
    new_f = numpy.array(new).astype(float)
    blended_f = multiply(base_f, new_f, opacity)
    return Image.fromarray(numpy.uint8(blended_f))


def parse_line(line):
    # Remove inline comments, if any
    line, *comment = line.split(COMMENT_PREFIX, 1)
    comment = comment[0] if comment else ''
    line = line.strip()

    # Blend mode
    mode = BlendMode.ADD
    if MULTIPLY_SUFFIX in line:
        mode = BlendMode.MULTIPLY
        line = line.replace(MULTIPLY_SUFFIX, '')

    # Find offset in form @x,y
    file, *offset = line.split('@', 1)
    file = file.strip()

    # Normalize offset to Tuple(Int)
    if offset:
        offset = offset[0].split(',', 1)
        offset = tuple(int(o) for o in offset)
    else:
        offset = ()

    return file + EXT if file else '', mode, offset, comment



def parse(args):
    """Reads input file and parses into a list of layer stacks

    Output is a list of lists, each list containing layers to be merged.
    Input file must end with two newlines.
    """
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stacks = []
    stack = []
    for fn in lines:
        parsed = parse_line(fn)
        file, *_, comment = parsed  # ignore mode and offset returned by parser

        # Non-empty line: Add to buffer
        if file:
            stack.append(parsed)
            continue

        # Lines with comments only: avoid flushing the buffer
        elif comment:
            continue

        # Empty line: flush buffered layers to output
        if not stack:
            continue
        stacks.append(list.copy(stack))
        stack = []

    # Flush one last time
    if stack:
        stacks.append(list.copy(stack))

    return stacks


def compose(args, i, fn_list):
    """Merges the input stack of filenames

    fn_list is a list of filenames representing a group of layers.
    i is the ordinal of this current layer group in the current script run.
    """
    # Determine filename
    template = '{prefix}{i:0>%d}{suffix}.png' % (args.digits)
    out_fn = template.format(prefix=args.prefix,
                             i=i + args.countfrom,
                             suffix=args.suffix)

    # Check for existing output target
    out_path = os.path.join(args.outdir, out_fn)
    if os.path.exists(out_path):
        if args.skipconflict:
            print(f'{out_fn} exists, skipping')
            return None

    # In input, lines are ordered in the order they're stacked, but when
    # iterating the file, we encounter the layers top-first.
    # Reverse so we start merging the stack from the bottom layer.
    layers_bot_first = fn_list[::-1]

    # Prep first layer for compositing
    file, mode, offset, comment = layers_bot_first.pop(0)
    base = Image.open(file).convert('RGBA')

    # Merge remaining layers into first layer
    for file, mode, offset, comment in layers_bot_first:
        do_multiply = mode == BlendMode.MULTIPLY

        layer = Image.open(file).convert('RGBA')

        if do_multiply and offset:
            print(f'{file}: multiply and offset not supported, skipping layer')
            continue

        if offset:
            # 3rd arg for paste() uses the layer as alpha mask
            print(f'pasting {file} @ {offset}')
            base.paste(layer, offset, layer)

        elif do_multiply:
            print(f'multiplying {file}')
            base = blend_multiply(base, layer)

        else:
            base.alpha_composite(layer)

    # Write
    print(f'Writing {out_fn}...\n')
    base.save(out_path)
    return out_path


def main():
    parser = ArgumentParser()

    parser.add_argument('-s', '--skipconflict',
                        action='store_true',
                        help='skip on conflict instead of overwrite if output '
                             'file exists')

    parser.add_argument('-c', '--countfrom',
                        type=int,
                        default=0,
                        help='number for output files to start counting from; '
                             'default=0')

    parser.add_argument('-d', '--digits',
                        action='count',
                        default=0,
                        help='number of digits to zero-pad to; default=3 (-ddd)')

    parser.add_argument('-D', '--diff',
                        action='store_true',
                        help='open generated files when done')

    parser.add_argument('-i', '--input',
                        type=str,
                        default='_info.txt',
                        help="folder to store output files; default='_info.txt'")

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

    new_files = []
    for i, grp in enumerate(parse(args)):
        file = compose(args, i, grp)
        if file:
            new_files.append(file)


    while args.diff and new_files:
        print()
        print('Showing output files -- Ctrl-C to quit')
        file = new_files.pop(0)
        img = Image.open(file)
        img.show()

        if new_files:
            # Give time for user to press the interrupt
            time.sleep(0.5)


if __name__ == '__main__':
    main()
