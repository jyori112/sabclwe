import click
import matplotlib.pyplot as plt
from pathlib import Path

@click.group()
def cli():
    pass

def sensitivity(results_path, output_path, baseline_path, xticks, xlim):
    parameters = []
    accuracies = []

    with open(baseline_path) as f:
        _, baseline_acc = f.read().strip().split()
        baseline_acc = float(baseline_acc)

    with open(results_path) as f:
        for line in f:
            parameter, _, accuracy = line.strip().split()
            parameters.append(float(parameter))
            accuracies.append(float(accuracy))

    plt.plot([-100, 100], [baseline_acc, baseline_acc], ':')
    plt.plot(parameters, accuracies, '-o')

    plt.xticks(xticks)

    plt.ylim(baseline_acc - 10, baseline_acc + 10)
    plt.xlim(*xlim)

    plt.savefig(output_path)


@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--baseline', type=click.Path(exists=True))
def sensitivity_align(results_path, output_path, baseline):
    sensitivity(results_path, output_path, baseline, [-2.5, -3.0, -3.5, -4.0, -4.5], (-2.3, -4.7))

@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--baseline', type=click.Path(exists=True))
def sensitivity_csls(results_path, output_path, baseline):
    sensitivity(results_path, output_path, baseline, [0.9, 0.8, 0.7, 0.6, 0.5], (0.92, 0.48))

if __name__ == '__main__':
    cli()
