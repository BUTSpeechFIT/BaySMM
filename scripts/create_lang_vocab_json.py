#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Create language ID to vocab_path mapping dictionary in json format
This is needed for training MultiBaySMM model.
"""

import os
import sys
import argparse
import json


def main():
    """main method"""

    args = parse_arguments()

    if not os.path.exists(args.in_vocab_file):
        print("- File not found:", args.in_vocab_file, file=sys.stderr)
        sys.exit()

    print("- ", args.lang, end=" ")
    lang_vocab = {}
    if os.path.exists(args.out_json_file):
        print(" Loading existing json file.. will be updated..")
        with open(args.out_json_file, "r") as fpr:
            lang_vocab = json.load(fpr)

    if args.lang in lang_vocab:
        print(f"  . Warning: {args.lang} already in {args.out_json_file}.")
        if os.path.realpath(args.in_vocab_file) != lang_vocab[args.lang]:
            print(f"  . Attention: Input vocab file does not match with the existing one",
                  args.in_vocab_file, lang_vocab[args.lang], file=sys.stderr)
        sys.exit()
    else:
        lang_vocab[args.lang] = os.path.realpath(args.in_vocab_file)
        with open(args.out_json_file, "w") as fpw:
            json.dump(lang_vocab, fpw, indent=2, sort_keys=True)
        print("  . Saved", args.out_json_file)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("in_vocab_file", help="path to a vocab file")
    parser.add_argument("lang", help="two letter language code")
    parser.add_argument("out_json_file", help="out json file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
