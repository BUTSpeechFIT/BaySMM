#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 03 Feb 2022
# Last modified : 03 Feb 2022

"""
Common utils for steps
"""

import os
import sys
import stat
import argparse
import numpy as np


def save_cmd_to_file(cmd, sh_file):

    with open(sh_file, 'w', encoding='utf-8') as fpw:
        fpw.write(cmd + "\n")
    os.chmod(sh_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IROTH | stat.S_IRGRP)


def load_dataset_file(dset_file):

    dsets = []
    kways = []
    with open(dset_file, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            if line.startswith("#"):
                continue
            else:
                parts = line.split()
                dsets.append(parts[0])
                if len(parts) == 2:
                    kways.append(parts[1])
                else:
                    kways.append("*")

    return dsets, kways

def read_lang_list(args):
    if args.lang_list_file:
        langs = np.loadtxt(args.lang_list_file, dtype=str).tolist()
    else:
        langs = args.lang_list

    return sorted(langs)


def main():
    """ main method """
    args = parse_arguments()

def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
