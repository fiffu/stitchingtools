from argparse import ArgumentParser
import json
from glob import glob
import os

abspath = os.path.abspath


def parse_error(e):
    ecls = e.__class__.__name__
    emsg = str(e)
    return f'{ecls}: {emsg}'


def ngrams(it, chunk_size):
    """Yields n-ples from iterable where n is chunk_size"""
    for i in range(0, len(it), chunk_size):
        yield it[i:i+chunk_size]


def traverse(root_dir):
    join = os.path.join

    def walk(path):
        for (root, dirs, files) in os.walk(path):
            yield (root, dirs, files)

    for path in walk(root_dir):
        yield path


def select_files(relpath, select_extensions):
    join = os.path.join
    print(relpath)

    for curdir, subdirs, files in traverse(relpath):
        for file in files:
            for ext in select_extensions:
                if not file.endswith(ext):
                    continue
                path = join(curdir, file)
                yield (path, ext)
                break


def find_key(root_dir, target_file='System.json'):
    config = None
    _, ext = os.path.splitext(target_file)

    for relpath, _ in select_files(root_dir, [ext]):
        if relpath.endswith(target_file):
            config = os.path.join(root_dir, relpath)
            break

    else:
        raise FileNotFoundError(f'Could not find {target_file}')

    try:
        with open(config, 'r', encoding='utf-8') as f:
            return json.load(f).get('encryptionKey')

    except BaseException as e:
        raise RuntimeError(f'Found {target_file} but failed to read '
                           f'encryption key ({parse_error(e)})')


def mkdir(path):
    exists, split = os.path.exists, os.path.split

    breadcrumbs = []
    head, file = split(path)

    while head and (not exists(head)):
        breadcrumbs.append(head)
        head, _ = split(head)

    # First path in the breadcrumbs is deepest in folder tree
    # Iterate in reverse to build from shallowest folder first
    for head in breadcrumbs[::-1]:
        if not exists(head):
            os.mkdir(head)



def write(path, data, flags='wb', retry=True):
    try:
        with open(path, flags) as f:
            f.write(data)
    except FileNotFoundError:
        if not retry:
            raise

        mkdir(path)
        write(path, data, flags, retry=False)


class Decryptor:
    header_len = 16
    signature = "5250474d56000000"
    version = "000301"
    remain = "0000000000"

    def __init__(self, root_dir, key):
        self.root = root_dir
        self.key = key

    @classmethod
    def make_fake_header(cls, self):
        header = cls.signature + cls.version + cls.remain
        return [int(x, 16) for x in ngrams(header, 2)]


    @property
    def key_array(self):
        b16int = lambda n: int(n, 16)
        return bytearray(map(b16int, ngrams(self.key, 2)))


    def decrypt(self, crypt, ignore_fake_header=True):
        hlen = self.header_len

        if not ignore_fake_header:
            header = byteArray[:hlen]
            if not self.make_fake_header() == header:
                raise ValueError("Input file doesn't seem to have fake-header. "
                                 'Ensure that target file is actually encrypted '
                                 'of try again with ignore_fake_header=True')

        # Trim "fake" header
        crypt = crypt[hlen:]

        # XOR key with real header
        header = crypt[:hlen]
        for i, (h, k) in enumerate(zip(header, self.key_array)):
            crypt[i] = h ^ k

        return crypt


    def decrypt_file(self, relpath, outdir, from_ext, to_ext, trim_prefix=None):
        outpath = os.path.join(outdir, relpath.replace(from_ext, to_ext))
        if trim_prefix:
            outpath = outpath.replace(trim_prefix, '')
            if outpath[0] in '/\\':
                outpath = outpath[1:]
        print(outpath)

        with open(relpath, 'rb') as f:
            clear = f.read()
            if self.key:
                clear = self.decrypt(bytearray(clear))

        write(outpath, clear)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('-k',
                        type=str,
                        metavar='KEY',
                        help='Specify key to use')

    parser.add_argument('-u',
                        action='store_true',
                        help='Indicates unencrypted assets (aliases `-k ""`)')

    parser.add_argument('-x',
                        type=str,
                        default=['rpgmvp:png', 'png:png'],
                        action='append',
                        metavar='FROM:TO',
                        help='Extension mapping: pass this once for each '
                             'mapping; defaults to "-x rpgmvp:png -x png:png"')

    parser.add_argument('-r', '--root',
                        type=abspath,
                        default='.',
                        metavar='PATH',
                        help='Root directory; defaults to current directory')

    parser.add_argument('-i', '--in',
                        type=os.path.normpath,
                        default='www/img/pictures',
                        metavar='PATH',
                        help='Relative path to directory storing encrypted '
                             'files; defaults to "www/img/pictures"')

    parser.add_argument('-o', '--out',
                        type=os.path.normpath,
                        default=None,
                        metavar='PATH',
                        help='Relative path to directory to store decrypted '
                             'output; leave blank to infer from input_dir')

    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = os.path.split(args.input_dir)[-1]

    return args


def main():
    args = get_args()
    original_dir = os.getcwd()

    outdir = args.output_dir
    indir = args.input_dir
    root = args.root_dir

    exts = {}
    for pair in args.extension:
        xfrom, xto = pair.split(':')
        exts[xfrom] = xto

    try:
        os.chdir(root)

        key = '' if args.unencrypted else args.key
        if key is None:
            key = find_key(root)
            if not key:
                with open('__unencrypted', 'w'):
                    return

        dec = Decryptor(root, key)

        for file, ext in select_files(indir, list(exts.keys())):
            dec.decrypt_file(file, args.output_dir, ext, exts[ext], indir)

    finally:
        os.chdir(original_dir)


if __name__ == '__main__':
    main()
