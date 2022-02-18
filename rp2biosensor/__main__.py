"""Build a graph representation of the retrosynthetic network."""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

from rp2biosensor.RP2Objects import RP2parser
from rp2biosensor.RP2Objects import RetroGraph
from rp2biosensor.Utils import write, write_json

# Path to templates 
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

TARGET_ID = "TARGET_0000000001"  # Default target ID
COFACTORS = [                    # Cofactor structures to be filtered
    "InChI=1S/O2/c1-2",  # O2
    "InChI=1S/H2O/h1H2",  # Water
    "InChI=1S/p+1",  # H+
    "InChI=1S/CO2/c2-1-3",  # CO2
    "InChI=1S/C10H16N5O13P3/c11-8-5-9(13-2-12-8)15(3-14-5)10-7(17)6(16)4(26-10)1-25-30(21,22)28-31(23,24)27-29(18,19)20/h2-4,6-7,10,16-17H,1H2,(H,21,22)(H,23,24)(H2,11,12,13)(H2,18,19,20)",  # ATP
    "InChI=1S/C21H28N7O17P3/c22-17-12-19(25-7-24-17)28(8-26-12)21-16(44-46(33,34)35)14(30)11(43-21)6-41-48(38,39)45-47(36,37)40-5-10-13(29)15(31)20(42-10)27-3-1-2-9(4-27)18(23)32/h1-4,7-8,10-11,13-16,20-21,29-31H,5-6H2,(H7-,22,23,24,25,32,33,34,35,36,37,38,39)",  # NADP(+)
    "InChI=1S/C21H30N7O17P3/c22-17-12-19(25-7-24-17)28(8-26-12)21-16(44-46(33,34)35)14(30)11(43-21)6-41-48(38,39)45-47(36,37)40-5-10-13(29)15(31)20(42-10)27-3-1-2-9(4-27)18(23)32/h1,3-4,7-8,10-11,13-16,20-21,29-31H,2,5-6H2,(H2,23,32)(H,36,37)(H,38,39)(H2,22,24,25)(H2,33,34,35)",  # NADPH
    "InChI=1S/C10H15N5O10P2/c11-8-5-9(13-2-12-8)15(3-14-5)10-7(17)6(16)4(24-10)1-23-27(21,22)25-26(18,19)20/h2-4,6-7,10,16-17H,1H2,(H,21,22)(H2,11,12,13)(H2,18,19,20)",  # ADP
    "InChI=1S/C21H27N7O14P2/c22-17-12-19(25-7-24-17)28(8-26-12)21-16(32)14(30)11(41-21)6-39-44(36,37)42-43(34,35)38-5-10-13(29)15(31)20(40-10)27-3-1-2-9(4-27)18(23)33/h1-4,7-8,10-11,13-16,20-21,29-32H,5-6H2,(H5-,22,23,24,25,33,34,35,36,37)",  # NAD+
    "InChI=1S/C21H29N7O14P2/c22-17-12-19(25-7-24-17)28(8-26-12)21-16(32)14(30)11(41-21)6-39-44(36,37)42-43(34,35)38-5-10-13(29)15(31)20(40-10)27-3-1-2-9(4-27)18(23)33/h1,3-4,7-8,10-11,13-16,20-21,29-32H,2,5-6H2,(H2,23,33)(H,34,35)(H,36,37)(H2,22,24,25)",  # NADH
    "InChI=1S/C21H36N7O16P3S/c1-21(2,16(31)19(32)24-4-3-12(29)23-5-6-48)8-41-47(38,39)44-46(36,37)40-7-11-15(43-45(33,34)35)14(30)20(42-11)28-10-27-13-17(22)25-9-26-18(13)28/h9-11,14-16,20,30-31,48H,3-8H2,1-2H3,(H,23,29)(H,24,32)(H,36,37)(H,38,39)(H2,22,25,26)(H2,33,34,35)",  # CoA
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
    parser.add_argument("--ojson",
                        help="Output the graph as json file if the path is not None. "
                             "Default: None")
    return parser


def run(args):
    # Extract and build graph
    rparser = RP2parser(args.rp2_results)
    rgraph = RetroGraph(rparser.compounds, rparser.transformations)
    nb_paths = rgraph.keep_source_to_sink(to_skip=COFACTORS, target_id=TARGET_ID)
    rgraph.refine()
    # Write output
    json_str = rgraph.to_cytoscape_export()
    if args.ojson is not None:
        write_json(Path(args.ojson), json_str)
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