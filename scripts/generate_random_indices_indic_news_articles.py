#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Requires the original indic-new-articles dataset.
https://github.com/ai4bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset

Generate random indices for train/dev/test in 5 different splits.
Similar to MLDoc.

The generated indices shall be used with generate_documents_indic_news_articles.py
to generate the data.



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
from sklearn.model_selection import train_test_split


def save_text(ixs, all_rows, out_fname, write_mode):

    with open(out_fname, write_mode, encoding="utf-8") as fpw:
        writer = csv.writer(fpw)
        for i in ixs:
            writer.writerow(all_rows[i])


def divide_into_train_dev_test(rows_split_i, out_dir, lang, write_mode, sizes, all_rows):
    """Divide the subset into train/dev/test in 1:1:4 parts, like in MLDoc"""

    train_ixs = rows_split_i[: sizes["train"]]
    dev_ixs = rows_split_i[sizes["train"] : sizes["train"] + sizes["dev"]]
    test_ixs = rows_split_i[sizes["train"] + sizes["dev"] :]

    # Save row numbers
    with open(os.path.join(out_dir, f"{lang}_train_row.ixs"), write_mode) as fpw:
        np.savetxt(fpw, train_ixs, fmt="%d")

    with open(os.path.join(out_dir, f"{lang}_dev_row.ixs"), write_mode) as fpw:
        np.savetxt(fpw, dev_ixs, fmt="%d")

    with open(os.path.join(out_dir, f"{lang}_test_row.ixs"), write_mode) as fpw:
        np.savetxt(fpw, test_ixs, fmt="%d")

    save_text(train_ixs, all_rows, os.path.join(out_dir, f"{lang}_train.csv"), write_mode)
    save_text(dev_ixs, all_rows, os.path.join(out_dir, f"{lang}_dev.csv"), write_mode)
    save_text(test_ixs, all_rows, os.path.join(out_dir, f"{lang}_test.csv"), write_mode)


def main():
    """main method"""

    args = parse_arguments()


    labels_of_interest = set(args.labels_of_interest)
    print('- Label2int:', end=" ")
    for l, i in label2int.items():
        print(l, i, end=' ')
    print('\n- Labels of interest:', labels_of_interest)

    label2lang = {}

    lang_label_dist = {}

    desired_num_examples_per_label_per_split = (
        args.per_label_train + args.per_label_dev + args.per_label_test
    )
    train_factor = args.per_label_train / desired_num_examples_per_label_per_split
    dev_factor = args.per_label_dev / desired_num_examples_per_label_per_split
    test_factor = args.per_label_test / desired_num_examples_per_label_per_split

    print(
        "- Desired train dev test: {:.4f} {:.4f} {:.4f}".format(
            train_factor, dev_factor, test_factor
        )
    )

    for lang in LANGS:

        label_dist = {}  # {label_1: number of examples, label_2: num of examples}
        label2row_ix = {}

        all_rows = []

        for set_name in ["train", "valid", "test"]:
            csv_fname = os.path.join(args.in_dir, f"{lang}/{lang}-{set_name}.csv")
            if not os.path.exists(csv_fname):
                continue
            with open(csv_fname, "r", encoding="utf-8") as fpr:
                csv_reader = csv.reader(fpr)
                for row_num, row in enumerate(csv_reader):
                    assert (
                        len(row) == 2
                    ), "Each row must contain only two columns: LABEL, TEXT"

                    label = row[0]
                    all_rows.append(row)  # append only text

                    if label not in labels_of_interest:
                        continue

                    if label not in label2int:
                        label2int[label] = len(label2int)

                    try:
                        label_dist[label] += 1
                        label2row_ix[label].append(row_num)
                    except KeyError:
                        label_dist[label] = 1
                        label2row_ix[label] = [row_num]

                    if label not in label2lang:
                        label2lang[label] = set()
                    label2lang[label].add(lang)

        if label_dist:
            lang_label_dist[lang] = label_dist

        num_labels = len(label_dist)

        write_mode = "w"
        is_first_label = True

        for label, num_examples in label_dist.items():

            row_ixs_label = label2row_ix[label]

            num_examples_per_label_per_split = num_examples // num_labels

            x = min(
                desired_num_examples_per_label_per_split,
                num_examples_per_label_per_split,
            )

            if x < desired_num_examples_per_label_per_split:
                sizes = {
                    "train": floor(num_examples_per_label_per_split * train_factor),
                    "dev": floor(num_examples_per_label_per_split * dev_factor),
                    "test": floor(num_examples_per_label_per_split * test_factor),
                }
            else:
                sizes = {
                    "train": args.per_label_train,
                    "dev": args.per_label_dev,
                    "test": args.per_label_test,
                }

            print(
                "- {:2s} {:15s} {:5d} out of {:5d} per split ({:2d} splits in total).".format(
                    lang, label, x, len(row_ixs_label), args.num_splits
                )
            )

            for split_ix in range(1, args.num_splits + 1):

                if is_first_label:
                    write_mode = "w"

                random.seed(args.seed + split_ix)
                random.shuffle(row_ixs_label)

                rows_split_i = row_ixs_label[:x]

                out_dir = os.path.join(args.out_base_dir, f"split_{split_ix}")
                os.makedirs(out_dir, exist_ok=True)

                # divide into train/dev/test with 1:1:4 ratio
                divide_into_train_dev_test(
                    rows_split_i, out_dir, lang, write_mode, sizes, all_rows
                )

                write_mode = "a"

            if is_first_label:
                is_first_label = False

    print("-" * 50)
    print(label2int)
    pprint(lang_label_dist)

    pprint(label2lang)

    with open(
        os.path.join(args.out_base_dir, "label2int_indic_news_articles.json"),
        "w",
        encoding="utf-8",
    ) as fpw:
        json.dump(label2int, fpw, indent=2)

    l2i_subset = {}
    for label in args.labels_of_interest:
        l2i_subset[label] = label2int[label]
    with open(
        os.path.join(args.out_base_dir, "label2int_indic_news_articles_subset.json"),
        "w",
        encoding="utf-8",
    ) as fpw:
        json.dump(l2i_subset, fpw, indent=2)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("in_dir", help="path to indic new articles base dir")
    parser.add_argument(
        "out_base_dir",
        help="path to save the random indices for each splits and language",
    )
    parser.add_argument(
        "-per_label_train",
        type=int,
        default=250,
        help="number of samples per label in the training set. defaults to 250, similar to MLDoc",
    )
    parser.add_argument(
        "-per_label_dev",
        type=int,
        default=250,
        help="number of samples per label in the dev set. defaults to 250, similar to MLDoc",
    )
    parser.add_argument(
        "-per_label_test",
        type=int,
        default=1000,
        help="number of samples per label in the test set. defaults to 1000, similar to MLDoc",
    )
    parser.add_argument("-num_splits", default=5, help="make 5 random splits")

    parser.add_argument(
        "-labels_of_interest",
        type=str,
        nargs="+",
        default=["entertainment", "sports", "business"],
        help="train/dev/test splits will be created only for these labels",
    )

    parser.add_argument("-seed", default=256, help="random seed")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
