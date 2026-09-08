"""Run the measured file pipeline; requires a checkpoint and video."""
import sys

from infer import main


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--no-display' not in argv:
        argv.append('--no-display')
    if '--report' not in argv:
        argv.extend(['--report', 'benchmark.json'])
    main(argv)
