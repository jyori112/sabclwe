import click
import sys

@click.command()
def main():
    for line in sys.stdin:
        coverage = line[9:15]
        accuracy = line[27:33]
        print("{:.2f}\t{:.2f}".format(float(coverage), float(accuracy)))

if __name__ == '__main__':
    main()


