#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 16 Aug 2021
# Last modified : 16 Aug 2021

"""
Check if two models have exactly same model params language-by-language
"""

import os
import sys
import argparse
import torch
from baysmm import MBaySMM, BaySMM


def main():
    """ main method """

    args = parse_arguments()

    cpu = torch.device("cpu")
    print("Loading model 1..")
    model_1 = torch.load(args.model_1, map_location=cpu)
    print("Loading model 2..")
    model_2 = torch.load(args.model_2, map_location=cpu)

    lids = set(model_1.T.keys()) | set(model_2.T.keys())

    m1 = False
    m2 = False
    # for lid in model_1.T:
    for lid in lids:
        if lid in model_1.T:
            print(lid, end=" ")
            m1 = True
        else:
            print(" -", end=" ")
            m1 = False

        if lid in model_2.T:
            print(lid, end=" ")
            m2 = True
        else:
            print(" -")
            m2 = False

        if m1 and m2:
            print(torch.allclose(model_1.T[lid].detach(), model_2.T[lid].detach()) and
                  torch.allclose(model_1.m[lid].detach(), model_2.m[lid].detach()))
        else:
            print("  -")


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_1", help="path to model 1")
    parser.add_argument("model_2", help="path to model 2")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
