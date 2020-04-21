"""
mvdecrypt.py

Usage:
    1. Ensure you have python 3
    2. Move this file into the same folder where the .rpgmvp files are kept.
    3. From terminal (command prompt), run this script.
        For Command Prompt on Windows, type:
            python mvdecrypt.py
    4. Files will be placed inside a folder called "decrypted".
"""

# Name of folder to put decrypted files in
OUTDIR = 'decrypted'


from argparse import ArgumentParser
import json
from glob import glob
import os
from threading import Thread

PROCESS_EXT = 'rpgmvp'
OUTPUT_EXT = 'png'

PROCESS_DIR = ['www', 'img', 'pictures']

def chunkify(it, chunk_size):
    """Yields n-ples from iterable where n is chunk_size"""
    for i in range(0, len(it), chunk_size):
        yield it[i:i+chunk_size]


def toBigram(s):
    """s -> List[[s[0], s[1]], [s[2], s[3]], ...]

    If length of sequence s is odd, the final list will have one element only.
    """
    return [x for x in chunkify(s, 2)]


def getRootDir(exe='Game.exe'):
    # Backtrack through file structure to root dir
    def relGlob(path, filepatt):
        return glob(pjoin(path, filepatt))

    print(exe)
    psplit = os.path.split
    pjoin = os.path.join
    pexists = os.path.exists

    path = os.getcwd()

    if pexists(exe):
        return path

    while not (relGlob(path, exe) or relGlob(path, '*.exe')):
        parents = psplit(path)
        if not any(parents):
            raise RuntimeError('Could not find root dir containing game '
                               'executable')
        path = pjoin(*parents[:-1])

    print(f'Detected root dir at {path}')
    return path


def getKey(rootdir):
    route = ['www', 'data', 'System.json']
    path = os.path.join(rootdir, *route)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            system = json.load(f)

        return system['encryptionKey']

    except BaseException as e:
        cls, msg = e.__class__.__name__, str(e)
        raise RuntimeError(f'Failed to find encryption key ({cls}: {msg})')


def getFiles(rootdir):
    path = os.path.join(rootdir, *PROCESS_DIR)
    for (root, dirs, files) in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            yield filepath

    # files = filter(lambda s: s.endswith(PROCESS_EXT), os.listdir(path))
    # files = [os.path.join(path, fn) for fn in files]
    # return files



class Decryptor():
    defaultHeaderLen = 16
    defaultSignature = "5250474d56000000"
    defaultVersion = "000301"
    defaultRemain = "0000000000"

    def __init__(self, key, rootdir):
        self.key = key
        self.rootdir = rootdir

        # Fake-header fields
        self._headerLen = None
        self._signature = None
        self._version = None
        self._remain = None


    @property
    def keyArray(self):
        key_bigram = [x for x in chunkify(self.key, 2)]
        b16int = lambda s: int(s, 16)
        return bytearray(map(b16int, key_bigram))

    @property
    def headerLen(self):
        return self._headerLen or self.defaultHeaderLen

    @property
    def signature(self):
        return self._signature or self.defaultSignature

    @property
    def version(self):
        return self._version or self.defaultVersion

    @property
    def remain(self):
        return self._remain or self.defaultRemain


    def verifyFakeHeader(self, fileHeader):
        fakeHeader = self.buildFakeHeader()
        return cmp(fakeHeader, fileHeader) == 0


    def buildFakeHeader(self):
        headerStructure = self.signature + self.version + self.remain
        fakeHeader = [
            int(x, 16) for x in toBigram(headerStructure)
        ]
        return fakeHeader


    def decrypt(self, byteArray, ignoreFakeHeader):
        headerLen = self.headerLen

        if not ignoreFakeHeader:
            header = byteArray[:headerLen]
            if not self.verifyFakeHeader(header):
                raise ValueError(
                    'Input file does not have a fake-header. '
                    'Make sure that input file is actually encrypted, or '
                    'try again with ignoreFakeHeader=True'
                )

        # Trim "fake" header
        byteArray = byteArray[headerLen:]

        # XOR on real header
        hdr = byteArray[:headerLen]
        keyba = self.keyArray
        hdr = [x ^ y for x, y in zip(hdr, keyba)]

        # Sub xor'd real header into file bytes
        for i, b in enumerate(hdr):
            byteArray[i] = b

        return byteArray


    def decryptFile(self, filePath, outputExt=OUTPUT_EXT):
        stub = os.path.join(os.getcwd(), *PROCESS_DIR)

        newfn = filePath.replace(stub, '')[1:].replace(PROCESS_EXT, OUTPUT_EXT)
        newpath = os.path.join(self.rootdir, OUTDIR, newfn)
        # raise ValueError(newpath)

        with open(filePath, 'rb') as f:
            crypt = f.read()
            if self.key:
                clear = self.decrypt(bytearray(crypt), True)
            else:
                clear = crypt

        try:
            self.write(newpath, clear)
        except FileNotFoundError:
            ensurePath(newpath)
            # self.write(newpath, clear)

        return newpath


    def write(self, path, data):
        with open(path, 'wb') as f:
            f.write(data)


def ensurePath(path):
    exists, isdir, split = os.path.exists, os.path.isdir, os.path.split
    print('ensure', path)

    breadcrumbs = []
    head, tail = split(path)

    while not exists(head):
        breadcrumbs.append(head)
        head, tail = split(head)

    # First path in the breadcrumbs is deepest in folder tree
    # Iterate in reverse to build from shallowest folder first
    for head in breadcrumbs[::-1]:
        print(head)
        if not exists(head):
            os.mkdir(head)


def worker(decryptor, fn):
    outfile = decryptor.decryptFile(fn)
    print('  wrote', outfile)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-k', '--key',
                        type=str,
                        help='Specify key to use')

    parser.add_argument('-x', '--exename',
                        type=str,
                        default='Game.exe',
                        help='Specify the name of the game executable'
                             '(for detecting game root dir)')

    args = parser.parse_args()
    rootdir = getRootDir(exe=args.exename)

    try:
        key = args.key or getKey(rootdir)
    except RuntimeError:
        key = None
    print(f'Using key: {key}')

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    dec = Decryptor(key, rootdir)
    threads = []
    for fn in getFiles(rootdir):
        t = Thread(target=worker, name=fn, args=(dec, fn))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
