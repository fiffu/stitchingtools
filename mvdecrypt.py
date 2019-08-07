"""
mvdecrypt.py

Usage:
    1. Ensure you have python 3
    2. Look inside <game_dir>/www/data/System.json  (tip: jsonlint.com)
    3. Inside the file, near the end, there is a "encryptionKey" value. Copy.
    4. Paste the encryptionKey value below where it says KEY. Keep the quotes.
    5. Move this file into the same folder where the .rpgmvp files are kept.
    6. From terminal (command prompt), run this script.
        For Command Prompt on Windows, type:
            python mvdecrypt.py
"""

# For RJ248754
KEY = "d14c2267d848abeb81fd590f371d39bd"

OUTDIR = 'decrypted'
PROCESS_EXT = 'rpgmvp'
OUTPUT_EXT = 'png'

import os


def chunkify(it, chunk_size):
    """Yields n-ples from iterable where n is chunk_size"""
    for i in range(0, len(it), chunk_size):
        yield it[i:i+chunk_size]


def toBigram(s):
    """s -> List[[s[0], s[1]], [s[2], s[3]], ...]

    If length of sequence s is odd, the final list will have one element only.
    """
    return [x for x in chunkify(s, 2)]


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


    def decryptFile(self, fileName, outputExt=OUTPUT_EXT):
        newfn = os.path.join(OUTDIR,
                             fileName.replace(PROCESS_EXT, OUTPUT_EXT))

        with open(fileName, 'rb') as f:
            ba = bytearray(f.read())
            clear = self.decrypt(ba, True)

        with open(newfn, 'wb') as g:
            g.write(clear)

        return newfn


if __name__ == '__main__':

    files = filter(lambda s: s.endswith(PROCESS_EXT), os.listdir('.'))

    dec = Decryptor(KEY)

    for fn in files:
        newfn = dec.decryptFile(fn)
        print('  wrote', newfn)
