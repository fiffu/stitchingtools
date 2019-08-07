"""
Quick and dirty image composition script.
Requires the `pillow` package (not vanilla PIL!).
Parses an input file for filenames to merge.
"""

import os
from PIL import Image

INPUT_FILE = '_info.txt'
COMMENT_MARKER = '#'

ON_CONFLICT = 'overwrite'
# ON_CONFLICT = 'skip'

OUT_DIR = "buffer"

PREPEND = '3_'
COUNT_FROM = 0
EXT = '.png'

def parse():
    """Reads input file and parses into a list of layer stacks

    Output is a list of lists, each list containing layers to be merged.
    Input file must end with two newlines.
    """

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fnames = [line.strip() + EXT 
              for line in lines
              if not line.startswith(COMMENT_MARKER)]
    
    out = []
    buffer = []
    for fn in fnames:
        # Non-empty line: Add layer to image stack
        if fn != EXT:
            # Remove inline comments, if any
            n, *comment = fn.split(COMMENT_MARKER, 1)
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
    def rgba(layer):
        return layer.convert('RGBA')

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
    layers_bot_first = [rgba(Image.open(x)) for x in fn_list[::-1]]

    # Prep first layer for compositing
    base = layers_bot_first[0]

    # Merge remaining layers into first layer
    for L in layers_bot_first[1:]:
        L = preprocess(L)
        base = Image.alpha_composite(base, L)

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

    # compose() overwrites existing filenames in the output dir.
    for i, grp in enumerate(parse()):
        compose(i, grp)
