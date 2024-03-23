#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 14 Mar 2022
# Last modified : 14 Mar 2022

"""
Rename language codes in GlobalVoices. jp -> ja, zhs -> zh
"""

import os
import sys
import argparse
import glob


def main():
    """ main method """

    args = parse_arguments()

    in_files = glob.glob(args.in_dir + f"/*/*.{args.in_lang_code}")
    print("- Files found:", len(in_files))
    for in_file in in_files:
        out_file = in_file.replace(f".{args.in_lang_code}", f".{args.out_lang_code}")
        out_file = out_file.replace(f"{args.in_lang_code}.", f"{args.out_lang_code}.")
        os.system(f"cp -v {in_file} {in_file}.bak")
        os.system(f"mv -v {in_file} {out_file}")

    dirnames = list(set([os.path.dirname(in_file) for in_file in in_files]))
    for in_dir in dirnames:
        out_dir = in_dir.replace(f"-{args.in_lang_code}", f"-{args.out_lang_code}")
        out_dir = out_dir.replace(f"{args.in_lang_code}-", f"{args.out_lang_code}-")

        os.system(f"cp -rv {in_dir} {out_dir}")


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", help="path to GlobalVoices base dir")
    parser.add_argument("in_lang_code", help="input language code. eg: jp")
    parser.add_argument("out_lang_code", help="output language code. eg: ja")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
