import os
import re

START_INDEX = 1


def zeropad(fname, num=None):
    patt = r'(?<=char_)\d+'
    h = int(re.search(patt, fname).group())
    new = '{:>04}'.format(num or h)
    if num:
        return 'char_{new}.png'.format(**locals())
    return re.sub(patt, new, fname)


if __name__ == '__main__':
    files = os.listdir('.')

    for notimg in [x for x in files if '.png' not in x]:
        if notimg.startswith('last-'):
            START_INDEX = int(notimg[5:]) + 1
        files.remove(notimg)

    if not files:
        raise Exception('No files to rename!')

    sortedfiles = sorted(files, key=zeropad)

    for i, fn in enumerate(sortedfiles):
        newname = zeropad(fn, START_INDEX+i)
        print(f'{i+1}. from {fn:>20}  to  {newname}')
        os.rename(fn, newname)
