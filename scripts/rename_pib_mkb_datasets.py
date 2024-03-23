#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rename pib, mkb, indic2indic datasets to have similar format as
other datasets from OPUS
"""

import os
import sys
from glob import glob
import argparse

def main():
    """ main method """

    args = parse_arguments()

    langs = sorted(["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"])

    for i, lang1 in enumerate(langs):

        for j, lang2 in enumerate(langs):

            if lang1 == lang2:
                continue

            if args.dset_name == "mkb":
                orig_file1 = os.path.join(args.in_dir + f"{lang1}-{lang2}/mkb.{lang1}")
                orig_file2 = os.path.join(args.in_dir + f"{lang1}-{lang2}/mkb.{lang2}")

            elif args.dset_name in ("pib-v1.3", "indic2indic"):
                orig_file1 = os.path.join(args.in_dir + f"{lang1}-{lang2}/train.{lang1}")
                orig_file2 = os.path.join(args.in_dir + f"{lang1}-{lang2}/train.{lang2}")

            if os.path.exists(orig_file1) and os.path.exists(orig_file2):

                tar_file1 = os.path.join(
                    args.in_dir,
                    f"{lang1}-{lang2}/{args.dset_name}.{lang1}-{lang2}.{lang1}"
                )
                tar_file2 = os.path.join(
                    args.in_dir,
                    f"{lang1}-{lang2}/{args.dset_name}.{lang1}-{lang2}.{lang2}"
                )

                os.system(f"cp {orig_file1} {tar_file1}")
                os.system(f"cp {orig_file2} {tar_file2}")

                print(orig_file1, '->', tar_file1, '   ', orig_file2, '->', tar_file2)




def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", help="path to pib-v1.3 or mkb or indic2indic dir")
    parser.add_argument("dset_name", choices=["pib-v1.3", "mkb","indic2indic"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
