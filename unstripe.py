"""
Quick and dirty script to extract frames tiled inside spritesheets.
"""

from collections import namedtuple
import os

from PIL import Image

Box = namedtuple('Box', 'left upper right lower')
Coords = namedtuple('Coords', 'x, y')


def blocks(img, blockx, blocky):
    wid, hgt = img.size

    if wid % blockx:
        raise ValueError(f'block width {blockx} does not tile properly with '
                         f'image size {wid}x{hgt} ({wid} % {blockx} != 0)')
    if hgt % blocky:
        raise ValueError(f'block height {blocky} does not tile properly with '
                         f'image size {wid}x{hgt} ({hgt} % {blocky} != 0)')

    for iy in range(hgt // blocky):
        for ix in range(wid // blockx):
            left = ix * blockx
            upper = iy * blocky

            right = left + blockx
            lower = upper + blocky

            block = img.crop(Box(left, upper, right, lower))
            yield block


def new_canvas(width, height, mode='RGBA'):
    return Image.new(mode, size=(width, height))


def join(canvas, blocks_iterable):
    canvas = canvas.copy()

    cursor = Coords(0, 0)

    wid_canvas, hgt_canvas = canvas.size

    for block in blocks_iterable:
        wid, hgt = block.size

        target_x = (cursor.x + wid) % wid_canvas
        target_y = (cursor.y + hgt) % hgt_canvas

        cursor = Coords(target_x, target_y)

        canvas.alpha_composite(block, dest=(cursor.x, cursor.y))

    return canvas


def unstripe(filename, block_wid, block_hgt):
    img = Image.open(filename)

    canvas = new_canvas(*img.size)

    canvas = join(canvas, blocks(img, block_wid, block_hgt))

    return canvas


def unreel(filename, block_wid, block_hgt, outdir=''):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    img = Image.open(filename)

    name, ext = os.path.splitext(filename)

    for i, block in enumerate(blocks(img, block_wid, block_hgt)):
        outfile = f'{name}_{i:>03}{ext}'
        block.save(os.path.join(outdir, outfile))
