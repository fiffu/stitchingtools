from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import sys
from typing import Generator

def progressbar(iterable, length=None):
    """Prints progressbar using click. Noop if click isn't installed."""
    try:
        import click
        return click.progressbar(iterable, length=length)
    except ImportError:
        from contextlib import nullcontext
        return nullcontext(iterable)

def format_error(e: BaseException):
    ecls = e.__class__.__name__
    emsg = str(e)
    return f'{ecls}: {emsg}'

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Yi{suffix}'

def exit(message):
    print(message)
    sys.exit(0)


@dataclass
class Args:
    root_path: Path
    output_folder: Path

    @classmethod
    def parse(cls):
        parser = ArgumentParser()

        parser.add_argument('-r', '--root',
                            type=Path,
                            default='.',
                            metavar='PATH',
                            help='Root directory; default: current directory')
        parser.add_argument('-o', '--out',
                            type=str,
                            default='./decrypted_pictures',
                            metavar='PATH',
                            help='Path relative to root directory where decrypted pictures '
                                 'are written to; default: ./decrypted_pictures')

        args = parser.parse_args()

        return cls(
            root_path=args.root,
            output_folder=args.root / Path(args.out)
        )


class Decryptor:
    HEADER_LENGTH = 16
    MAGIC_SIGNATURE = "5250474d56000000"
    MAGIC_VERSION = "000301"
    MAGIC_PADDING = "0000000000"

    MAGIC_HEADER = MAGIC_SIGNATURE + MAGIC_VERSION + MAGIC_PADDING

    PNG_EXTENSION = '.png'

    def __init__(self, root: Path):
        self.root = root
        self.key = None

    @property
    def pictures_path(self):
        return self.root.joinpath(*self.PICTURES_PATH)

    @property
    def system_json_path(self):
        return self.root.joinpath(*self.SYSTEM_JSON_PATH)

    @property
    def extension(self):
        return self.EXTENSION

    @property
    def output_extension(self):
        return self.PNG_EXTENSION

    @property
    def encryption_key(self):
        if self.key:
            return self.key

        self.key = self._load_key(self.system_json_path)
        return self.key

    def generate_output_path(self, path: Path):
        pictures_stem = str(self.pictures_path)
        p = str(path)
        p = p.removeprefix(pictures_stem)
        p = p.removesuffix(self.extension)
        p = '.' + p + self.output_extension
        return Path(p)

    def valid(self) -> bool:
        if self.encryption_key:
            for _ in self.walk_pictures():
                return True
        return False

    def decrypt(self, path: Path) -> bytearray:
        crypt = self._read_bytearray(path)

        # Trim "fake" header bytes
        crypt = crypt[self.HEADER_LENGTH:]

        # XOR key with real header
        real_header = crypt[:self.HEADER_LENGTH]

        for i, (h, k) in enumerate(zip(real_header, self.encryption_key)):
            crypt[i] = h ^ k

        return crypt

    def walk_pictures(self, path=None) -> Generator[Path, None, None]:
        for subpath in (path or self.pictures_path).iterdir():
            if subpath.is_file() and subpath.suffix == self.extension:
                yield subpath

            if subpath.is_dir():
                yield from self.walk_pictures(subpath)

    def _read_bytearray(self, path: Path) -> bytearray:
        with open(path, 'rb') as file:
            content = file.read()
            return bytearray(content)

    def _load_key(self, path: Path) -> bytearray:
        try:
            contents = path.read_text(encoding='utf8')
            contents_json = json.loads(contents)

            key: str = contents_json['encryptionKey']
            key_pairwise = [key[i:i+2] for i in range(0, len(key), 2)]

            parse_base16 = lambda n: int(n, 16)
            return bytearray(parse_base16(pair) for pair in key_pairwise)
        except FileNotFoundError:
            return None
        except KeyError:
            exit("Empty encryption key in game data. Most likely it's not encrypted!")
        except BaseException as e:
            raise RuntimeError(f'Found {path} but failed to read encryption key ({format_error(e)})')


class RMMV(Decryptor):
    PICTURES_PATH = 'www/img/pictures'.split('/')
    SYSTEM_JSON_PATH = 'www/data/System.json'.split('/')
    EXTENSION = '.rpgmvp'


class RMMV2(Decryptor):
    PICTURES_PATH = 'img/pictures'.split('/')
    SYSTEM_JSON_PATH = 'data/System.json'.split('/')
    EXTENSION = '.png_'


class Main:
    VERSIONS = [RMMV, RMMV2]

    def __init__(self, args: Args):
        self.args = args

    def call(self):
        decryptor = self.detect(self.args.root_path)
        if not decryptor:
            raise RuntimeError('cannot detect game version')

        if not self.args.output_folder.exists():
            self.args.output_folder.mkdir(parents=True)

        inputs = list(decryptor.walk_pictures())
        bytes_written = self.decrypt_all(decryptor, inputs)

        return len(inputs), bytes_written

    def decrypt_all(self, decryptor, inputs):
        written = 0

        with ThreadPoolExecutor() as pool:
            futures = pool.map(partial(self.decrypt_one, decryptor), inputs)

            with progressbar(futures, length=len(inputs)) as bar:
                for file_size in bar:
                    written += file_size

        return written

    def detect(self, root_path: Path) -> Decryptor:
        for version in self.VERSIONS:
            if version(root_path).valid():
                return version(root_path)

    def decrypt_one(self, decryptor: Decryptor, input_file: Path) -> int:
        output_relpath = decryptor.generate_output_path(input_file)

        output_path = self.args.output_folder / output_relpath
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        clear = decryptor.decrypt(input_file)
        return output_path.write_bytes(clear)


if __name__ == '__main__':
    args = Args.parse()

    started = datetime.now()
    files_written, bytes_written = Main(args).call()
    elapsed = datetime.now() - started

    bytes_written = sizeof_fmt(bytes_written)
    print(f'Decrypted {files_written} files totalling {bytes_written} in {elapsed.seconds} secs')

