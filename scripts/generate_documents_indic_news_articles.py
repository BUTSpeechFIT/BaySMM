#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate train/dev/test documents based on the random indices
given the original indicnlp-news-articles dataset
"""

import os
import re
import sys
import csv
import json
import glob
import argparse

import numpy as np

from indic_news_utils import LANGS, L2I


def main():
    """main method"""

    args = parse_arguments()

    l2i_subset = {}
    with open(args.label2int, "r", encoding="utf-8") as fpr:
        l2i_subset = json.load(fpr)
    print("- Loaded label2int subset:", l2i_subset)

    for lang in LANGS:

        all_sents = []
        all_labels = []

        for set_name in ["train", "valid", "test"]:
            csv_fname = os.path.join(
                args.input_data_dir, f"{lang}/{lang}-{set_name}.csv"
            )
            if not os.path.exists(csv_fname):
                print('- File not found:', csv_fname)
                sys.exit()

            with open(csv_fname, "r", encoding="utf-8") as fpr:
                csv_reader = csv.reader(fpr)
                for row in csv_reader:

                    # will split text and label to different lists while maintaining one-to-one
                    # correspondence
                    sent = row[1].replace("\n", " ")
                    all_sents.append(sent)
                    all_labels.append(L2I[row[0]])

        all_labels = np.asarray(all_labels, dtype=int)
        all_sents = np.asarray(all_sents, dtype=str)

        assert (all_labels.size == all_sents.size), "Number of sents != number of labels, probably  rows in contains new line"

        # read train/dev/test splits for all splits
        for i in range(1, args.num_splits + 1):
            for set_name in ["train", "dev", "test"]:
                ixs_file = os.path.join(
                    args.input_random_ixs_base_dir,
                    f"split_{i}/{lang}_{set_name}_row.ixs",
                )
                if not os.path.exists(ixs_file):
                    print(" File not found:", ixs_file)
                    sys.exit()

                ixs = np.loadtxt(ixs_file, dtype=int)
                print("  -", lang, i, set_name, ixs.shape, all_labels[ixs].shape, all_sents[ixs].shape)

                out_dir = os.path.join(args.output_base_dir, f"{lang}")
                os.makedirs(out_dir, exist_ok=True)

                out_txt_file = os.path.join(out_dir, f"{lang}_{set_name}_split_{i}.txt")
                np.savetxt(out_txt_file, all_sents[ixs], fmt="%s")

                out_label_file = os.path.join(out_dir, f"{lang}_{set_name}_split_{i}.labels")
                np.savetxt(out_label_file, all_labels[ixs], fmt="%d")

    print("- Done")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_data_dir", help="path to original indicnlp-news-articles dataset dir"
    )
    parser.add_argument(
        "input_random_ixs_base_dir", help="path to base dir with random indices splits"
    )
    parser.add_argument("label2int", help="path to label2int json file")
    parser.add_argument(
        "output_base_dir", help="path to output base dir to save the data"
    )
    parser.add_argument(
        "-num_splits",
        type=int,
        default=5,
        help="number of random splits, should be same as used for creating random indices",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
