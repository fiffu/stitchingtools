"""
Very lazy script for finding bits in a bytestack

usage: python3 sigscan.py --help
"""

from argparse import ArgumentParser
import os



def read_window(fp, window_size: int, step_size: int):
    chunk = fp.read(window_size)
    offset = 0
    
    while chunk:
        yield chunk, offset
        
        step = fp.read(step_size)
        if not step:
            return
        
        chunk = chunk[step_size:] + step
        offset += step_size


def scan(needle, path, count=None, **openargs):
    sz = len(needle)
    window_size = 2000 * sz
    step_size = window_size // 2

    flags = 'r' if isinstance(needle, str) else 'rb'

    ctr = 0
    count = max(0, count or 0)  # let count be 0 or +ve int
                                # if count is 0 then yield all matches

    with open(path, flags, **openargs) as fp:
        for chunk, offset in read_window(fp, window_size, step_size):
            found = chunk.find(needle)
            
            if found > -1:
                yield offset + found
                ctr += 1

                if ctr == count:
                    break


def parse_args():
    sigs = {
        'png': b'89504E470D0A1A0A',
        'jpg': b'FFD8FF',
        'zip': b'504B',
    }


    parser = ArgumentParser()

    parser.add_argument('files',
                        type=str,
                        nargs='+',
                        help='files to scan')

    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-n', '--needle',
                       help='value to find, interpreted as bytes in hex, or '
                            'as string if -s is passed')

    for ext in sigs.keys():
        mutex.add_argument(f'--{ext}',
                           action='store_true',
                           help='set needle to {ext.upper()} file signature')

    parser.add_argument('-s', '--str',
                        action='store_true',
                        help='specify that needle is a string literal')

    parser.add_argument('-1', '--one',
                        action='store_true',
                        help='if set, halts searching once the first match in '
                             'the file is found')

    args = parser.parse_args()

    # Typing
    if args.needle:
        args.needle = (str if args.str else bytes.fromhex)(args.needle)
    else:
        for ext, sig in sigs.items():
            if hasattr(args, ext):
                args.needle = sig

    return args


def main():
    args = parse_args()

    count = 1 if args.one else 0
    needle_hex = args.needle.hex()
    print(f'needle: {needle_hex}')

    for file in args.files:
        for offset in scan(args.needle, file, count=count):
            print(f'{needle_hex} offset {offset} in {file}')

if __name__ == '__main__':
    main()
