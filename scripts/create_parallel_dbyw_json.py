#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 21 Jul 2021
# Last modified : 21 Jul 2021

"""
Create a json file comprising of file paths to  k-way parallel or bitexts
between various languages from several data sources.
This json file will be used for extracting DxW BoW statistics, which
will be further used to train the Multilingual BaySMM model.
"""

import os
import sys
import argparse
import json
import glob
from pprint import pprint
from pylibs.misc.io import read_simple_flist


def main():
    """main method"""

    args = parse_arguments()

    parallel_data = {}

    if len(args.langs) != len(args.fpaths):
        print("- Error: Number of langs should be equal to the number of fpaths")
        sys.exit()

    if os.path.exists(args.out_file):
        with open(args.out_file, "r", encoding="utf-8") as fpr:
            parallel_data = json.load(fpr)
        print(f"- Loading existing json file from: {args.out_file}")
        pprint(parallel_data)
        print("-" * 80)

    if args.num not in parallel_data:
        parallel_data[args.num] = {}

    for i, lang in enumerate(args.langs):
        if lang not in parallel_data[args.num]:
            parallel_data[args.num][lang] = os.path.realpath(args.fpaths[i])
        else:
            print(
                "- Error:",
                lang,
                "already in",
                args.num,
                parallel_data[args.num][lang],
                file=sys.stderr,
            )
            sys.exit()

    with open(args.out_file, "w", encoding="utf-8") as fpw:
        json.dump(parallel_data, fpw, indent=2, sort_keys=True)

    print("- Saved:", args.out_file)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "out_file",
        help="File to save train_data.json. If the file already exists, content will be appended.",
    )
    parser.add_argument(
        "num",
        type=str,
        help="parallel dataset number, should be same with-in the multi-aligned set",
    )
    parser.add_argument("-langs", required=True, type=str, nargs='+', help="ISO 639-1 two letter language code")
    parser.add_argument("-fpaths", required=True,  type=str, nargs='+', help="path to text that corresponds to the language")



    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
