#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Create label2int mapping for Indic news articles dataset
"""

import os
import sys
import csv
import json
import glob
import random
import argparse

from math import floor
from pprint import pprint
from indic_news_utils import LANGS
import numpy as np


def main():
    """main method"""

    args = parse_arguments()

    label2int = {}
    label2lang = {}

    lang_label_dist = {}


    for lang in LANGS:

        label_dist = {}  # {label_1: number of examples, label_2: num of examples}
        label2row_ix = {}

        all_labels = []
        all_rows = []

        row_num = 0
        for set_name in ["train", "valid", "test"]:
            csv_fname = os.path.join(args.in_dir, f"{lang}/{lang}-{set_name}.csv")
            if not os.path.exists(csv_fname):
                print("- File not found for", lang, set_name, "Skipping..")
                continue

            with open(csv_fname, "r", encoding="utf-8") as fpr:
                csv_reader = csv.reader(fpr)
                for row in csv_reader:
                    all_rows.append(row)
                    if row[0] not in label2int:
                        label2int[row[0]] = len(label2int)

                    label = row[0]
                    try:
                        label_dist[label] += 1
                        label2row_ix[label].append(row_num)
                    except KeyError:
                        label_dist[label] = 1
                        label2row_ix[label] = [row_num]

                    row_num += 1
                    all_labels.append(label2int[label])

                    if label not in label2lang:
                        label2lang[label] = set()
                    label2lang[label].add(lang)

        if label_dist:
            lang_label_dist[lang] = label_dist

    print(label2int)
    pprint(lang_label_dist)

    for label, langs in label2lang.items():
        print(label, len(langs), sorted(list(langs)))

    for lang in lang_label_dist:
        print(lang, sorted(list(lang_label_dist[lang].keys())))


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("in_dir", help="path to indic new articles base dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
