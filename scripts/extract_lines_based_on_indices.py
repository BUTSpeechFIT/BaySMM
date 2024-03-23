#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given input text file and a file with line numbers (indices),
selects the specific lines and saves to the output file
"""

import os
import sys
import codecs
import argparse
import numpy as np


def main():
    """ main method """

    args = parse_arguments()

    if os.path.exists(args.out_text_file):
        print(args.out_text_file, 'already exists.')
        sys.exit(0)


    indices = np.loadtxt(args.indices_file)
    print("- Loaded indices (line numbers):", indices.size, np.unique(indices).shape)

    lno = 0
    lines = []
    # if args.empty_lines > 0:
    #     if args.where == 'beg':
    #         print("Appending", args.empty_lines, "empty lines in the beginning")
    #         lines = [''] * args.empty_lines

    indices = set(indices.tolist())
    with open(args.in_text_file, 'r', encoding='utf-8') as fpr:
        # for doc_ix in indices:
        #    while lno < doc_ix:
        for line in fpr:
            line = line.strip()
            if lno in indices:
                lines.append(line)

            lno += 1

    # if args.empty_lines > 0:
    #     if args.where == 'end':
    #         print("Appending", args.empty_lines, "empty lines at the end")
    #         lines.extend([''] * args.empty_lines)

    with open(args.out_text_file, 'w', encoding='utf-8') as fpw:
        fpw.write("\n".join(lines) + "\n")
    print("- Num lines in out text:", len(lines))
    print("- Saved to", args.out_text_file)


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_text_file", help="input text file")
    parser.add_argument("indices_file", help="file with sorted indices (line numbers) to select")
    parser.add_argument("out_text_file", help="output file")
    # parser.add_argument("-empty_lines", type=int, default=0, help="num of empty lines to append")
    # parser.add_argument("-where", default="", type=str, choices=['beg', 'end'], help='where to append empty lines')

    args = parser.parse_args()

    # if args.empty_lines > 0:
    #     if not args.where:
    #         print("-where is required when -empty_lines > 0")
    return args

if __name__ == "__main__":
    main()
