import click
import numpy as np

@click.command()
@click.option('--train-out', type=click.Path())
@click.option('--dev-out', type=click.Path())
@click.option('--dev-size', type=int)
@click.option('--seed', type=int)
def main(train_out, dev_out, dev_size, seed):
    if seed is not None:
        np.random.seed(seed)

    src2trg = defaultdict(set)

    for line in sys.stdin:
        src, trg = line.strip().split()
        src2trg[src].add(trg)

    srcs = list(src2trg.keys())

    index = np.random.permutation(len(srcs))

    with open(dev_out, 'w') as f:
        for i in index[:dev_size]:
            src = srcs[i]
            trgs = src2trg[src]
            for trg in trgs:
                print("{}\t{}".format(src, trg), file=f)
    
    with open(train_out, 'w') as f:
        for i in index[dev_size:]:
            src = srcs[i]
            trgs = src2trg[src]
            for trg in trgs:
                print("{}\t{}".format(src, trg), file=f)

if __name__ == '__main__':
    main()
