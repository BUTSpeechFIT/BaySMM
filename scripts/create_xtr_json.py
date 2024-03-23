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
    """ main method """

    args = parse_arguments()

    xtr_data = {}

    if args.out_file:
        if os.path.exists(args.out_file):
            with open(args.out_file, 'r', encoding='utf-8') as fpr:
                xtr_data = json.load(fpr)

    else:
        if args.in_dir:
            args.out_file = os.path.join(args.in_dir, "xtr_data.json")
        elif args.in_file:
            args.out_file = os.path.join(os.path.dirname(args.in_file), "xtr_data.json")
        else:
            args.out_file = os.path.join(os.path.dirname(args.in_flist), "xtr_data.json")


    if args.lang not in xtr_data:
        xtr_data[args.lang] = []

    fnames = []
    if args.in_dir:
        fnames = glob.glob(os.path.realpath(args.in_dir) + "/*.npz")
    else:
        if args.in_file:
            fnames = [os.path.realpath(args.in_file)]
        else:
            if os.path.exists(args.in_flist):
                fnames = read_simple_flist(args.in_flist)
            else:
                print(args.in_flist, "not found.")
                sys.exit()

    print("-", args.lang, "Files found:", len(fnames), end=" ")

    xtr_data[args.lang].extend(fnames)

    # remove duplicates, just in case
    fnames =  list(set(xtr_data[args.lang]))
    xtr_data[args.lang] = fnames


    with open(args.out_file, "w", encoding="utf-8") as fpw:
        json.dump(xtr_data, fpw, indent=4, sort_keys=True)

    print(args.out_file, "saved.")


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("lang", help="ISO 639-1 two letter language code")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("-in_file", help="path to single npz file")
    group.add_argument("-in_dir", help="path to input dir or flist")
    group.add_argument("-in_flist", help="path to input flist containing paths to npz files")

    parser.add_argument("-out_file", default="", help="default will be in xtr_data.json in same dir. If the file exists, content will be appended.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
