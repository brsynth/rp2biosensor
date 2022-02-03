"""Build a graph representation of the retrosynthetic network."""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

from rp2biosensor.RP2Objects import RP2parser
from rp2biosensor.RP2Objects import RetroGraph
from rp2biosensor.Utils import write

# Path to templates 
TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'

TARGET_ID = 'TARGET_0000000001'  # Default target ID
COFACTORS = [                    # Cofactor structures to be filtered
    'InChI=1S/O2/c1-2',  # O2
    'InChI=1S/H2O/h1H2',  # Water
    'InChI=1S/p+1',  # H+
    'InChI=1S/CO2/c2-1-3',  # CO2
]


def build_args_parser(prog='rp2biosensor'):
    desc = "Generate HTML outputs to explore Sensing Enabling Metabolic Pathway from RetroPath2 results."
    parser = argparse.ArgumentParser(description=desc, prog=prog)
    parser.add_argument('rp2_results',
                        help='RetroPath2.0 results')
    # parser.add_argument('--reverse_direction',
    #                     help='Reverse direction of reactions described in RetroPath2.0 results.',
    #                     default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--opath',
                        help=f'Output path. Default: {Path().resolve() / "biosensor.html"}.',
                        default=Path().resolve() / 'biosensor.html')
    parser.add_argument('--otype',
                        help='Output type. This could be either (i) "dir" which means '
                             'ouput files will outputted into this directory, or (ii) '
                             '"file" which means that all files will be embedded into '
                             'a single HTML page. Default: file',
                        default='file', choices=['dir', 'file'])
    return parser


def run(args):
    # Extract and build graph
    rparser = RP2parser(args.rp2_results)
    rgraph = RetroGraph(rparser.compounds, rparser.transformations)
    nb_paths = rgraph.keep_source_to_sink(to_skip=COFACTORS, target_id=TARGET_ID)
    rgraph.refine()
    # Write output
    json_str = rgraph.to_cytoscape_export()
    write(args, TEMPLATE_DIR, json_str)


def main():
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        datefmt='%d/%m/%Y %H:%M:%S',
        format='%(asctime)s -- %(levelname)s -- %(message)s'
    )
    parser = build_args_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()