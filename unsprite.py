from argparse import ArgumentParser
from collections import namedtuple
from glob import glob
import os

from PIL import Image

Box = namedtuple('Box', 'left upper right lower')
XY = namedtuple('XY', 'x, y')


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
            box = Box(
                left=ix * blockx,
                upper=iy * blocky,
                right=left + blockx,
                lower=upper + blocky
            )
            block = img.crop(Box(left, upper, right, lower))
            yield block


def new_canvas(width, height, mode='RGBA'):
    return Image.new(mode, size=(width, height))


def tile_out(canvas, blocks_iterable):
    canvas = canvas.copy()

    cursor = XY(0, 0)

    wid_canvas, hgt_canvas = canvas.size

    for block in blocks_iterable:
        wid, hgt = block.size

        target_x = (cursor.x + wid) % wid_canvas
        target_y = (cursor.y + hgt) % hgt_canvas

        cursor = XY(target_x, target_y)

        canvas.alpha_composite(block, dest=(cursor.x, cursor.y))

    return canvas


def unstripe(filename, block_wid, block_hgt):
    img = Image.open(filename)

    canvas = new_canvas(*img.size)

    canvas = tile_out(canvas, blocks(img, block_wid, block_hgt))

    return canvas


def unreel(filename, args):
    img = Image.open(filename)

    if args.box:
        block_wid, block_hgt = args.box
    elif args.matrix:
        block_wid, block_hgt = [a // b for a, b in zip(img.size, args.matrix)]
    else:
        return

    name, ext = os.path.splitext(filename)
    outdir = name if args.folderize else ''
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    outfiles = []
    for i, block in enumerate(blocks(img, block_wid, block_hgt)):
        outname = f'{name}_{i:>03}{ext}'
        outfile = os.path.join(outdir, outname)
        block.save(outfile)
        outfiles.append(outfile)

    return outfiles


def get_args():
    def int2ple(s):
        x, y = [int(n.strip()) for n in s.split(',')]
        return x, y

    parser = ArgumentParser()

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('-b', '--box',
                     type=int2ple,
                     metavar='WIDTH,HEIGHT')

    grp.add_argument('-m', '--matrix',
                     type=int2ple,
                     metavar='COLS,ROWS')

    parser.add_argument('--glob',
                        default='*.png',
                        metavar='PATTERN',
                        help='default: *.png')

    parser.add_argument('-f', '--folderize',
                        action='store_true',
                        help='store outputs in folders, one per spritesheet')

    return parser.parse_args()



def main():
    args = get_args()
    globstr = args.glob

    for file in glob(globstr):
        print(file)
        unreel(file, args)

if __name__ == '__main__':
    main()
