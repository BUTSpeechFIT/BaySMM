#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 04 Mar 2022
# Last modified : 04 Mar 2022

"""
Prints the number of tokens in the BoW stats from given train json file
"""

import os
import sys
import argparse
import json
import scipy.sparse
import numpy as np


def main():
    """main method"""

    args = parse_arguments()

    lang2sents = {}
    lang2tokens = {}
    lang2pairs = {}

    num_sents = 0
    num_tokens = 0
    num_nonzero = 0
    data_json = {}
    par_sets = 0

    for json_fname in args.train_json:
        print(json_fname)
        with open(json_fname, "r", encoding="utf-8") as fpr:
            data_json = json.load(fpr)

        par_sets += len(data_json)
        for p_num in data_json:
            par_set = data_json[p_num]

            lids = list(par_set["npz"].keys())
            for i, _ in enumerate(lids):
                for j, _ in enumerate(lids):
                    if lids[i] not in lang2pairs:
                        lang2pairs[lids[i]] = set()
                    #if lids[i] != lids[j]:
                    lang2pairs[lids[i]].add(lids[j])

            for lid in par_set["npz"]:
                npz = scipy.sparse.load_npz(par_set["npz"][lid]).tocsr()
                num_sents += npz.shape[0]
                num_tokens += npz.sum()
                num_nonzero += npz.count_nonzero()

                if lid not in lang2sents:
                    lang2sents[lid] = 0
                    lang2tokens[lid] = 0

                lang2sents[lid] += npz.shape[0]
                lang2tokens[lid] += npz.sum()

    print("- Num par sets: {:6d}".format(par_sets))
    print("- Num sents   : {:12d} = {:.2f} M".format(num_sents, num_sents / 1e6))
    print("- Num tokens  : {:12d} = {:.2f} M".format(num_tokens, num_tokens / 1e6))
    print("- Num nonzero : {:12d} = {:.2f} M".format(num_nonzero, num_nonzero / 1e6))
    for lid in lang2sents:
        print(
            "- {:3s} {:.2f} {:6.2f}".format(
                lid.upper(), lang2sents[lid] / 1e6, lang2tokens[lid] / 1e6
            ), sorted(list(lang2pairs[lid])), len(lang2pairs[lid])
        )




def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("train_json", nargs="+", help="path to train json file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
