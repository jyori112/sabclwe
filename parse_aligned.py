import click
import sys
import numpy as np

@click.command()
def main():
    for line in sys.stdin:
        src, trg, _, _, scorePerChar, _, _, _ = line.strip().split('\t')

        src = src.replace('|', '').replace(':', '')
        trg = trg.replace('|', '').replace(':', '')

        scorePerChar = float(scorePerChar.split(':')[1])

        print("{}\t{}\t{:f}".format(src, trg, scorePerChar))

if __name__ == '__main__':
    main()


