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



import json
from glob import glob
import os
from threading import Thread

PROCESS_EXT = 'rpgmvp'
OUTPUT_EXT = 'png'

CWD = None


def chunkify(it, chunk_size):
    """Yields n-ples from iterable where n is chunk_size"""
    for i in range(0, len(it), chunk_size):
        yield it[i:i+chunk_size]


def toBigram(s):
    """s -> List[[s[0], s[1]], [s[2], s[3]], ...]

    If length of sequence s is odd, the final list will have one element only.
    """
    return [x for x in chunkify(s, 2)]


def getRootDir():
    # Backtrack through file structure to root dir
    global CWD

    if CWD is not None:
        return CWD

    psplit = os.path.split
    pjoin = os.path.join
    pexists = os.path.exists

    path = os.getcwd()

    while not glob(pjoin(path, '*.exe')):
        path = pjoin(psplit(path)[:-1])
        if not any(path):
            raise RuntimeError('Could not find root dir containing game '
                               'executable')

    print(f'Detected root dir at {path}')
    CWD = path
    return CWD


def getKey():
    route = ['www', 'data', 'System.json']
    path = os.path.join(getRootDir(), *route)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            system = json.load(f)

        return system['encryptionKey']

    except BaseException as e:
        cls, msg = e.__class__.__name__, str(e)
        raise RuntimeError(f'Failed to find encryption key ({cls}: {msg})')


def getFiles():
    route = ['www', 'img', 'pictures']
    path = os.path.join(getRootDir(), *route)
    files = filter(lambda s: s.endswith(PROCESS_EXT), os.listdir(path))
    files = [os.path.join(path, fn) for fn in files]
    return files



class Decryptor():
    defaultHeaderLen = 16
    defaultSignature = "5250474d56000000"
    defaultVersion = "000301"
    defaultRemain = "0000000000"

    def __init__(self, key):
        self.key = key

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
        newfn = os.path.basename(filePath).replace(PROCESS_EXT, OUTPUT_EXT)
        newfn = os.path.join(getRootDir(), OUTDIR, newfn)

        with open(filePath, 'rb') as f:
            ba = bytearray(f.read())
            clear = self.decrypt(ba, True)

        with open(newfn, 'wb') as g:
            g.write(clear)

        return newfn


def worker(decryptor, fn):
    outfile = decryptor.decryptFile(fn)
    print('  wrote', outfile)


if __name__ == '__main__':
    key = getKey()
    print(f'Using key {key}')

    dec = Decryptor(key)

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    files = getFiles()
    threads = []
    for fn in files:
        t = Thread(target=worker, name=fn, args=(dec, fn))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
