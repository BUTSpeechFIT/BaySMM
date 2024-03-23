#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 25 Jan 2022
# Last modified : 25 Jan 2022

"""
Compute the overlap between two files
"""

import os
import sys
import argparse
from pylibs.misc.io import read_simple_flist

def main():
    """ main method """

    args = parse_arguments()

    sents1 = set(read_simple_flist(args.in_file1))
    sents2 = set(read_simple_flist(args.in_file2))

    print('s1:', len(sents1), 's2:', len(sents2), 's1 and s2:', len(sents1 & sents2))


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_file1")
    parser.add_argument("in_file2")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
