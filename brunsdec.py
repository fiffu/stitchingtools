import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import pathlib
from typing import Generator
from binascii import unhexlify

from rich.console import Console


console = Console()


class BrunsPNG:
    """Based on https://github.com/Bioruebe/brunsdec"""

    class Magic:
        # TXTTypeSignature = b'EENZ'
        PNGTypeSignature = b'EENC'
        XORValue = unhexlify('EFBEADDE')
        HeaderLength = len(PNGTypeSignature) + len(XORValue)

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._file: bytes = None

    @property
    def file(self):
        if not self._file:
            with open(self.path, 'rb') as file:
                self._file = file.read()
        return self._file

    @property
    def header(self) -> bytes:
        return self.file[:self.Magic.HeaderLength]

    @property
    def key(self):
        typesig_len = len(self.Magic.PNGTypeSignature)
        masked_key = self.header[typesig_len:]
        return bxor(masked_key, self.Magic.XORValue)

    @property
    def content(self):
        return self.file[self.Magic.HeaderLength:]

    @property
    def encrypted(self):
        return self.header.startswith(self.Magic.PNGTypeSignature)

    def decrypt(self):
        if not self.encrypted:
            return self.file
        return bxor(itertools.cycle(self.key), self.content)


def bxor(b1, b2):
    res = bytearray()
    for x, y in zip(b1, b2):
        res.append(x ^ y)
    return res


def glob(path: pathlib.Path, glob: str, recurse_depth=0) -> Generator[pathlib.Path, None, None]:
    for subpath in path.glob(glob):
        if subpath.is_file():
            yield subpath
        elif subpath.is_dir():
            if recurse_depth <= 0:
                return
            yield from glob(subpath, glob, recurse_depth - 1)


def decrypt(path: pathlib.Path, output_dir: pathlib.Path, skip_unencrypted: bool):
    file = BrunsPNG(path)
    if skip_unencrypted:
        if not file.encrypted:
            return

    console.log(path)
    out_path = output_dir / path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'wb') as f:
        f.write(file.decrypt())


def main(glob_pattern: str, input_dir: pathlib.Path, *args):
    files = list(glob(input_dir, glob_pattern, recurse_depth=5))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path in files:
            fut = pool.submit(decrypt, path, *args)
            if fut.exception():
                raise fut.exception()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir',
                        type=pathlib.Path, default='./images')
    parser.add_argument('-o', '--output-dir',
                        type=pathlib.Path, default='./extracted')
    parser.add_argument('-g', '--glob',
                        default='**/*.png')
    parser.add_argument('--skip-unencrypted',
                        help="if set, don't copy unecrypted assets to output dir",
                        action='store_true')

    args = parser.parse_args()
    main(args.glob, args.input_dir, args.output_dir, args.skip_unencrypted)
