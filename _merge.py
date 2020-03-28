"""
Quick and dirty image composition script.
Requires the `pillow` package (`pip install Pillow` -- not vanilla PIL!)

Usage:
    python _merge.py

Steps:
    1. Ensure you are using Python 3.
    2. Move this file to the same dir as the image files you want composited
    3. Define layer files to be composited in the input file (_info.txt)
         NOTE: Ensure your input file ends with at least 2 empty lines.
    4. Take note of comments and multiply-mode notation (see docstrings)
    5. Take note of optional settings like simple affixes etc.
    6. Run this script:
        python _merge.py
    7. Composited files are deposited in the output directory

Hacking this script:
    - You can use the pre/postprocess functions to inject any changes you want
      directly on every layer before/after compositing (like cropping).
"""

from argparse import ArgumentParser
from enum import Enum
import os
import time

from blend_modes import multiply  # pip install blend_modes
import numpy  # blend_modes depends on matrices
from PIL import Image  # pip install Pillow


# File to read layer composition from. Images are layered and stacked in the
# order they appear in this file. You specify the file extension of the input
# images in this script so you don't have to repeat in in the input file.
INPUT_FILE = '_info.txt'
EXT = '.png'

# Works just like a Python comment. Inline within a line of data is okay.
COMMENT_PREFIX = '#'

# Place this immediately after a filename to indicate this layer should be
# composited using a multiply blend mode instead of a regular overlap. It's
# called a suffix but you can put this symbol anywhere in the filename.
MULTIPLY_SUFFIX = '*'

# Action to take in case the output file exists. Accepted values so far are
# 'overwrite' and 'skip'
ON_CONFLICT = 'overwrite'
# ON_CONFLICT = 'skip'

# Name of directory to put the composited images into
OUT_DIR = "buffer"

# A prefix to put in front of the generated filenames. Okay to leave blank.
PREPEND = ''

# A cardinal number to start counting up from in the generated filenames.
COUNT_FROM = 0



# Enums
class BlendMode(Enum):
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

    out = []
    buffer = []
    for fn in lines:
        parsed = parse_line(fn)
        file, *_, comment = parsed

        # Non-empty line: Add layer to image stack
        if file:
            buffer.append(parsed)
            continue

        # Lines with comments only: avoid flushing the buffer
        elif comment:
            continue

        # Empty line: flush buffered layers to output
        if not buffer:
            continue
        out.append(list.copy(buffer))
        buffer = []

    # Flush one last time
    if buffer:
        out.append(list.copy(buffer))

    return out


def compose(args, i, fn_list):
    """Merges the input stack of filenames

    fn_list is a list of filenames representing a group of layers.
    i is the ordinal of this current layer group in the current script run.
    """
    # Determine filename
    i += args.countfrom
    out_fn = f'{args.prefix}{i:0>3}{args.suffix}.png'

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
        multiply = mode == BlendMode.MULTIPLY

        L = Image.open(file).convert('RGBA')

        if multiply and offset:
            print(f'{file}: multiply and offset not supported, skipping layer')
            continue

        if offset:
            # 3rd arg for paste() uses L as alpha mask
            print(f'pasting {file} @ {offset}')
            base.paste(L, offset, L)

        elif multiply:
            print(f'multiplying {file}')
            base = blend_multiply(base, L)

        else:
            base.alpha_composite(L)

    # Write
    print(f'Writing {out_fn}...\n')
    base.save(out_path)
    return out_path



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-s', '--skipconflict',
                        action='store_true',
                        help='skip on conflict instead of overwrite if output '
                             'file exists')

    parser.add_argument('-c', '--countfrom',
                        type=int,
                        default=1,
                        help='number for output files to start counting from; '
                             'default=1')

    parser.add_argument('-d', '--diff',
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
        file = new_files.pop(0)
        img = Image.open(file)
        img.show()

        if new_files:
            time.sleep(0.5)

