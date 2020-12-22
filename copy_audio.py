"""
usage: python3 copy_audio.py --help

Mux audio and video streams from two different input videos.
Requires the ffmpeg binary.
"""

from os.path import basename
from subprocess import run, STDOUT, DEVNULL

import click  # pip install click


def shellrun(args, quietly):
    print(args)
    if quietly:
        run(args, stdout=DEVNULL, stderr=DEVNULL)
    else:
        run(args)


@click.command()
@click.option('--video', '-v', metavar='FILE',
              prompt='video source')
@click.option('--audio', '-a', metavar='FILE',
              prompt='audio source')
@click.option('--output', '-o', default="use video's", metavar='FILE',
              prompt='output file')
@click.option('--extra', '-x', default='',
              prompt='extra args for ffmpeg')
@click.option('--quiet/--noisy', default=True)
def main(**kwargs):
    cmd = """
        ffmpeg -i {video} -i {audio} -c copy
            -map 0:v:0 -map 1:a:0
            -shortest
            {output} {extra}
    """
    cmd = """
        ffmpeg -i {video} -i {audio} -c copy
            -map 0:0 -map 1:1
            -shortest
            {output} {extra}
    """
    args = []
    for c in cmd.split():
        arg = c
        if c.startswith('{') and c.endswith('}'):
            arg = kwargs.get(c[1:-1])
            if c == '{output}' and arg == "use video's":
                arg = basename(kwargs.get('video'))
        if arg:
            args.append(arg)

    shellrun(args, kwargs.get('quiet'))


if __name__ == '__main__':
    main()
