import click
import numpy as np
from scipy.stats import wilcoxon

@click.command()
@click.argument('observation1', type=click.Path(exists=True))
@click.argument('observation2', type=click.Path(exists=True))
def main(observation1, observation2):
    obs1 = np.load(observation1)
    obs2 = np.load(observation2)

    t, p_value = wilcoxon(obs1, obs2)

    print('{:f}'.format(p_value))

if __name__ == '__main__':
    main()

