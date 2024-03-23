#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 03 Feb 2022
# Last modified : 03 Feb 2022

"""
Create flist for each language. This created flist will be used to extract BoW features
"""

import os
import sys
import glob
import argparse
import numpy as np
from step_utils import load_dataset_file, read_lang_list


def main():
    """main method"""

    args = parse_arguments()

    if args.ovr:
        print("- Will overwrite any existing flist")

    parallel_base_dir = os.path.realpath(args.parallel_base_dir)

    dsets, kways = load_dataset_file(args.dataset_list_file)

    langs = read_lang_list(args)

    msl = []
    if len(args.msl) > 1:
        if len(args.msl) != len(dsets):
            print(
                "- Input arg error: -msl should have either 1 or the same number of args as the",
                "number of datasets, which is",
                len(dsets),
                ":",
                dsets,
                file=sys.stderr,
            )
            sys.exit()

        else:
            msl = args.msl

    else:
        msl = args.msl * len(dsets)

    if args.verbose:
        print(langs)
        print(dsets)
        print(kways)
        print(msl)

    os.makedirs(args.out_flist_dir, exist_ok=True)

    # create flist for each language by walking through every dataset
    for lang in langs:
        out_file = os.path.join(args.out_flist_dir, f"{lang}.flist")
        if os.path.exists(out_file):
            if args.ovr:
                if args.verbose:
                    print("- Will overwrite", out_file)
            else:
                print(
                    "- Error:",
                    out_file,
                    "already exists.",
                    "Use --ovr to overwrite or choose different out dir",
                )
                sys.exit()

    for i, lang in enumerate(langs):

        pfiles = []
        out_file = os.path.join(args.out_flist_dir, f"{lang}.flist")

        if args.verbose:
            print("-", lang)

        for j, dset in enumerate(dsets):
            pfiles_dset = []
            msl_patt = f"msl-{msl[j]}"

            if args.verbose:
                print("  - {:20s}".format(dset), end=" ")

            if kways[j] == "*":

                for k, lang2 in enumerate(langs):

                    patt1 = (
                        parallel_base_dir + f"/{dset}/{lang}-{lang2}/*{msl_patt}.{lang}"
                    )
                    patt2 = (
                        parallel_base_dir + f"/{dset}/{lang2}-{lang}/*{msl_patt}.{lang}"
                    )

                    files_found = glob.glob(patt1)
                    if files_found:
                        pfiles_dset.extend(files_found)
                        if args.verbose:
                            print(lang + "-" + lang2, len(pfiles_dset), end=" | ")
                    else:
                        files_found = glob.glob(patt2)
                        if files_found:
                            pfiles_dset.extend(files_found)
                            if args.verbose:
                                print(lang2 + "-" + lang, len(pfiles_dset), end=" | ")

            else:
                patt1 = parallel_base_dir + f"/{dset}/{kways[j]}/*{msl_patt}.{lang}"
                files_found = glob.glob(patt1)
                if files_found:
                    pfiles_dset.extend(files_found)
                    if args.verbose:
                        print(kways[j], len(pfiles_dset), end=" | ")


            pfiles.extend(pfiles_dset)
            if args.verbose:
                print()
        print(f"- {lang} text files found:", len(pfiles))

        np.savetxt(out_file, pfiles, fmt="%s")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parallel_base_dir", help="Base dir where all the parallel data lives."
    )
    parser.add_argument(
        "dataset_list_file",
        type=str,
        help="File with list of parallel datasets to use. One per line.",
    )

    # dataset_group = parser.add_mutually_exclusive_group(required=True)
    # dataset_group.add_argument("-dataset-list-file", type=str)
    # dataset_group.add_argument("-dataset-list", type=str, nargs="+")

    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument(
        "-lang_list_file",
        type=str,
        help="File with list of languages (two letter ISO codes) to use. Space separated.",
    )
    lang_group.add_argument(
        "-lang_list",
        type=str,
        nargs="+",
        help="List of languages (two letter ISO codes) to use. Space separated.",
    )

    parser.add_argument(
        "-msl",
        nargs="+",
        type=int,
        required=True,
        help="Min sentence length constraint, as used earlier to prepare the data from step 1. Just one value for all datasets or individual values for each dataset.",
    )
    parser.add_argument(
        "-out_flist_dir",
        type=str,
        required=True,
        help="Dir path to save lang.flist files",
    )
    parser.add_argument(
        "--ovr", action="store_true", help="overwrite existing flist files"
    )
    parser.add_argument("--verbose", action="store_true", help="Increased verbosity")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
