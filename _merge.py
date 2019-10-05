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

import os

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



def blend(base: Image, new: Image, opacity: float = 1) -> Image:
    base_f = numpy.array(base).astype(float)
    new_f = numpy.array(new).astype(float)
    blended_f = multiply(base_f, new_f, opacity)
    return Image.fromarray(numpy.uint8(blended_f))


def parse():
    """Reads input file and parses into a list of layer stacks

    Output is a list of lists, each list containing layers to be merged.
    Input file must end with two newlines.
    """
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fnames = [line.strip() + EXT
              for line in lines
              if not line.startswith(COMMENT_PREFIX)]

    out = []
    buffer = []
    for fn in fnames:
        # Non-empty line: Add layer to image stack
        if fn != EXT:
            # Remove inline comments, if any
            n, *comment = fn.split(COMMENT_PREFIX, 1)
            n = n.strip()
            buffer.append(n)
            continue

        # Empty line: flush buffered layers to output
        if not buffer:
            continue
        out.append(list.copy(buffer))
        buffer = []
    return out


def compose(i, fn_list):
    """Merges the input stack of filenames

    fn_list is a list of filenames representing a group of layers.
    i is the ordinal of this current layer group in the current script run.
    """
    # Determine filename
    i += COUNT_FROM
    out_fn = f'{PREPEND}{i:0>3}.png'
    out_path = os.path.join(OUT_DIR, out_fn)
    if os.path.exists(out_path):
        if ON_CONFLICT is 'skip':
            print(f'{out_fn} exists, skipping')
            return

    # In input, lines are ordered in the order they're stacked, but when
    # iterating the file, we encounter the layers top-first.
    # Reverse so we start merging the stack from the bottom layer.
    layers_bot_first = fn_list[::-1]

    # Prep first layer for compositing
    base = Image.open(layers_bot_first[0]).convert('RGBA')

    # Merge remaining layers into first layer
    for file in layers_bot_first[1:]:
        useblend = MULTIPLY_SUFFIX in file
        L = Image.open(file.replace(MULTIPLY_SUFFIX, '')).convert('RGBA')
        L = preprocess(L)

        method = Image.alpha_composite
        if useblend:
            print('blend', L)
            method = blend

        base = method(base, L)

    # Postprocessing on composed base layer
    base = postprocess(base)

    # Write
    print(f'Writing {out_fn}...')
    base.save(out_path)


def preprocess(layer):
    """Postprocessing step for each layer before compositing into base."""
    return layer


def postprocess(composited_base):
    """Postprocessing step for each image after compositing layers."""
    # return composited_base.crop((322, 0, 640, 480))
    return composited_base


if __name__ == '__main__':
    # Prep dir for output
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for i, grp in enumerate(parse()):
        compose(i, grp)
