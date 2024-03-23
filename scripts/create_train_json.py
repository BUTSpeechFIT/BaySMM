#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Create xtr data json given a folder or flist
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

    train_data = {}

    if os.path.exists(args.out_file):
        with open(args.out_file, "r", encoding="utf-8") as fpr:
            train_data = json.load(fpr)
        if args.verbose:
            print("- Loading existing json file.. will be updated..")
            pprint(train_data)
            print("-" * 80)

    if args.num not in train_data:
        train_data[args.num] = {"npz": {}}

    npz_file = ""
    # vocab_file = ""
    if os.path.exists(args.npz_file):
        npz_file = os.path.realpath(args.npz_file)
    else:
        print("- ERROR:", args.npz_file, "NOT FOUND.", file=sys.stderr)
        sys.exit()

    # if os.path.exists(args.vocab_file):
    #     vocab_file = os.path.realpath(args.vocab_file)
    # else:
    #     print(args.vocab_file, "NOT FOUND.", file=sys.stderr)
    #     sys.exit()

    if args.lang not in train_data[args.num]["npz"]:
        train_data[args.num]["npz"][args.lang] = npz_file
        print("-", args.num, args.lang, npz_file)
    else:
        print(
            "- WARNING:",
            args.lang,
            "already in",
            args.num,
            train_data[args.num]["npz"][args.lang],
            file=sys.stderr,
        )
        sys.exit()

    # if args.lang not in train_data[args.num]["vocab"]:
    #     train_data[args.num]["vocab"][args.lang] = vocab_file
    # else:
    #     print(
    #         "-WARNING:",
    #         args.lang,
    #         "already in",
    #         args.num,
    #         train_data[args.num]["vocab"][args.lang],
    #         file=sys.stderr,
    #     )
    #     sys.exit()

    if args.trainable:
        if "train" not in train_data[args.num]:
            train_data[args.num]["train"] = []
        train_data[args.num]["train"].append(args.lang)

    with open(args.out_file, "w", encoding="utf-8") as fpw:
        json.dump(train_data, fpw, indent=2, sort_keys=True)

    if args.verbose:
        pprint(train_data)
        print("- Saved:", args.out_file)





def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "num",
        type=str,
        help="parallel dataset number, should be same with-in the multi-aligned set",
    )
    parser.add_argument("lang", help="ISO 639-1 two letter language code")

    parser.add_argument("npz_file", help="path to npz stats file")
    # parser.add_argument("vocab_file", help="path to vocab file")

    parser.add_argument(
        "out_file",
        help="File to save train_data.json. If the file already exists, content will be appended.",
    )
    parser.add_argument("--trainable", action="store_true", help="add current lang to the train list.  This is useful while extending any existing model to newer languages.")

    parser.add_argument("--verbose", action="store_true", help="verbose - print json contents")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
