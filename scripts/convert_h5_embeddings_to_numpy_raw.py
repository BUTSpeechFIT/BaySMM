#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Convert embeddings in h5 format to numpy raw format. Keeps only the mean parameter
"""

import os
import sys
import argparse
import h5py
import numpy as np
from pylibs.misc.io import get_ivectors_from_h5


def main():
    """ main method """

    args = parse_arguments()

    embs = get_ivectors_from_h5(args.emb_h5f)

    print('embs:', embs.shape)

    if args.out_type == 'tsv':
        np.savetxt(args.out_base + ".tsv", embs, delimiter='\t')

    elif args.out_type == 'npy':
        np.savet(args.out_base + ".tsv", embs)

    elif args.out_type == 'txt':
        np.savetxt(args.out_base + ".txt", embs)

    elif args.out_type == "raw":
        embs.tofile(args.out_base + ".raw")



def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emb_h5f")
    parser.add_argument("out_base", help="out file basename without ext")
    parser.add_argument("out_type", choices=['npy', 'tsv', 'txt', 'raw'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
