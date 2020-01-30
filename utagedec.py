from argparse import ArgumentParser
from glob import glob
import os

DEFAULT_KEY = 'InputOriginalKey'

def getbytes(file):
    with open(file, 'rb') as f:
        return bytearray(f.read())


def decrypt_file(filename, key=None, outdir=None):
    clear = decrypt(filename, key=key)

    if outdir:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
    else:
        outdir = ''
    
    name, ext = os.path.splitext(filename)
    ext = ext.replace('.utage', '')
    outfile = os.path.join(outdir, f'{name}{ext}')
    
    with open(outfile, 'wb') as f:
        f.write(clear)


def decrypt(filename, key=None):
    cipher = getbytes(filename)
    key_bytes = bytearray(key or DEFAULT_KEY, 'utf8')
    return xor(cipher, key_bytes)


def xor(cipher_bytes, key_bytes):
    arr = cipher_bytes.copy()
    key_length = len(key_bytes)

    for i, d in enumerate(arr):
        if d == 0:
            continue
        
        k = key_bytes[i % key_length]
        arr[i] = (arr[i] ^ k) or k
    return arr


if __name__ == '__main__':
    parser = ArgumentParser(description='Decryptor for utage encrypted files')

    parser.add_argument('input',
                        nargs='?',
                        type=str,
                        help='globstring for files to decrypt (default: *.utage)',
                        default='*.utage')
    
    parser.add_argument('-k', '--key',
                        type=str,
                        help='decryption key to use')

    parser.add_argument('-o', '--outdir',
                        type=str,
                        help='directory to write output files to',
                        metavar='DIR')

    args = parser.parse_args()

    files = glob(args.input)
    numfiles = len(files)

    for i, file in enumerate(files):
        decrypt_file(file, args.key, args.outdir)
        count = i + 1
        pct = int(count / numfiles * 100)
        print(f'{file}\t\t{count}/{numfiles}\t({pct}%)')
