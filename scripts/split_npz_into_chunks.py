#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Split BoW npz into chunks
"""

import os
import sys
import argparse
from scipy import sparse
import numpy as np


def main():
    """main method"""

    args = parse_arguments()

    npz = sparse.load_npz(args.npz_file)
    print("- Loaded npz:", npz.shape)

    if not args.out_base:
        args.out_base = os.path.basename(args.npz_file).rsplit(".", 1)[0]

    out_dir = os.path.dirname(args.npz_file)

    six = 0
    eix = six + args.chunk_size
    if eix > npz.shape[0]:
        eix = npz.shape[0]

    i = 1
    while six < npz.shape[0]:
        out_f = os.path.join(out_dir, f"{args.out_base}_chunk_{i}.npz")
        sparse.save_npz(out_f, npz[six:eix, :])

        print("{:7d} {:7d}".format(six, eix), out_f, "saved.")

        six = eix
        eix += args.chunk_size
        if eix > npz.shape[0]:
            eix = npz.shape[0]
        i += 1


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "npz_file",
        help="path to npz file",
        type=str,
    )
    parser.add_argument("chunk_size", type=int, help="number of rows in a chunk")
    parser.add_argument(
        "-out_base",
        type=str,
        default="",
        help="out base name (will use same basename as input npz_file), \
chunk number will be appended",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
