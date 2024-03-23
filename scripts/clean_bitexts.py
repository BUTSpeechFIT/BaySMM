#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Clean bitexts especially from CCAligned.
Cleaning is based on en text only. Other lang is alinged based on the line numbers.
"""

import os
import sys
import argparse
import re
import string
from feature_xtr_from_text_spm_v2 import remove_punc

def is_it_a_numeral(tok):

    n = 0
    flag = False
    for c in tok:
        if ord('0') <= ord(c) <= ord('9'):
            n += 1
    if len(tok) * 0.3 < n:
        flag = True
    return flag

def is_it_eng(tok):

    n = 0
    flag = False
    for c in tok.lower():
        if ord('a') <= ord(c) <= ord('z'):
            n += 1
    if len(tok) == n:
        flag = True
    return flag


def main():
    """ main method """

    args = parse_arguments()

    if os.path.exists(args.file_1 + ".orig.bak"):
        print(args.file_1, 'already cleaned, replaced and backed up.')
        sys.exit()

    line_nums = []
    line_num = 0
    zero_lines = 0
    non_eng_lines = 0
    clean_lines = []
    non_uniq = 0
    with open(args.file_1, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line_num += 1
            print("\r {:9d}".format(line_num), end="")
            lower_line = remove_punc(line.strip())
            tokens = lower_line.strip().split()
            num_numerals = 0
            num_eng = 0

            uniq_tokens = set(tokens)
            if len(uniq_tokens) < len(tokens) * 0.65:
                non_uniq += 1
                continue

            for tok in tokens:
                if is_it_a_numeral(tok):
                    num_numerals += 1
                if is_it_eng(tok):
                    num_eng += 1
            if num_numerals > 0.3 * len(tokens):
                zero_lines += 1
                continue
            elif num_eng < len(tokens):
                non_eng_lines += 1
                continue
            else:
                line_nums.append(line_num)
                clean_lines.append(line.strip())
    print()
    print("numeral lines:", zero_lines)
    print("noneng  lines:", non_eng_lines)
    print("nonuniq lines:", non_uniq)

    print("clean   lines:", len(clean_lines))

    with open(args.file_1 + ".clean", "w", encoding="utf-8") as fpw:
        for line in clean_lines:
            fpw.write(line + "\n")

    line_num = 0
    i = 0
    lang2_lines = []
    with open(args.file_2, "r", encoding="utf-8") as fpr:
        for line_ix in line_nums:
            while line_num < line_ix:
                fpr.readline()
                line_num += 1
            lang2_lines.append(fpr.readline().strip())
            line_num += 1

    with open(args.file_2 + ".clean", "w", encoding="utf-8") as fpw:
        for line in lang2_lines:
            fpw.write(line + "\n")

    if args.replace:
        print("orig -> .orig.bak   clean -> orig")
        os.system(f"mv {args.file_1} {args.file_1}.orig.bak")
        os.system(f"mv {args.file_2} {args.file_2}.orig.bak")

        os.system(f"mv {args.file_1}.clean {args.file_1}")
        os.system(f"mv {args.file_2}.clean {args.file_2}")


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("file_1", help="path to file 1, should be ENGLISH only")
    parser.add_argument("file_2", help="path to file 2")
    parser.add_argument("--replace", action="store_true", help="replace orig files with clean ones. orig will have .orig.bak extension")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
