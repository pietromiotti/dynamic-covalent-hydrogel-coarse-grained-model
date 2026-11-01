from network import Network
import click

import sys


@click.command()
@click.option('-npeg', '--npeg', default=1, help='number of PEG', type=click.INT)
@click.option('-nmon', '--nmon',
              default=4,
              help='number of monomer per Arm',
              type=click.INT)
@click.option('-o', '--output',
              default='network.atom',
              help='output network',
              type=click.STRING)
@click.option('-v', '--visualize',
              default='viz.atom',
              help='visualize network',
              type=click.STRING)
@click.option('-bond_length', '--bond_length',
              default=0.6,
              help='length of bond',
              type=click.FLOAT)
@click.option('-box_length_x', '--box_length_x',
              default=10,
              help='boxLengthx',
              type=click.FLOAT)
@click.option('-box_length_y', '--box_length_y',
              default=10,
              help='boxLengtht',
              type=click.FLOAT)
@click.option('-box_length_z', '--box_length_z',
              default=10,
              help='boxLengthz',
              type=click.FLOAT)

def main(npeg, nmon, output, visualize, bond_length, box_length_x, box_length_y, box_length_z):
    Net = Network()
    Net.Network(npeg, nmon, box_length_x, box_length_y, box_length_z, bond_length, 0, 0)
    Net.writeNetwork(output)
    Net.visualizeNetwork(visualize)


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
    except ValueError:
        print("value error")
