#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Get label proportion across all the splits for each language
1. RCV dataset
2. IndicNLP news articles dataset
"""

import os
import sys
import csv
import json
import glob
import argparse
import numpy as np


def load_labels_from_csv(csv_fname, l2i):

    labels_found = set()
    labels = []
    with open(csv_fname, "r", encoding="utf-8") as fpr:
        csv_reader = csv.reader(fpr)
        for row in csv_reader:

            # will split text and label to different lists while maintaining one-to-one
            # correspondence
            sent = row[1].replace("\n", " ")
            # all_sents.append(sent)
            labels.append(l2i[row[0]])
            labels_found.add(row[0])
    labels = np.asarray(labels, dtype=int)
    return labels, labels_found


def load_labels_from_file(fname, l2i):

    labels_found = set()
    labels = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            parts = line.strip().split()
            labels.append(l2i[parts[0]])
            labels_found.add(parts[0])
    labels = np.asarray(labels, dtype=int)
    return labels, labels_found


def get_ina_stats(args, l2i):

    stats_dict = {}
    lang2labels_found = {}

    for split in range(1, 6):
        csv_files = glob.glob(args.random_ixs_dir + f"/split_{split}/*.csv")
        print("- Split", split, len(csv_files))
        for csv_fname in csv_files:

            lang, set_name = os.path.basename(csv_fname).split(".")[0].split("_")
            labels, labels_found = load_labels_from_csv(csv_fname, l2i)

            if lang not in stats_dict:
                stats_dict[lang] = {'train': [], 'dev': [], 'test': []}
                lang2labels_found[lang] = {'train': [], 'dev': [], 'test': []}

            lab_ixs, lab_cnts = np.unique(labels, return_counts=True)
            lab_ratios = lab_cnts / lab_cnts.sum()
            stats_dict[lang][set_name].append(lab_cnts)
            lang2labels_found[lang][set_name].append(lab_ixs)

    return stats_dict, lang2labels_found


def get_rcv_stats(args, l2i):

    set_names = ['train_1000', 'dev', 'test']

    stats_dict = {}
    lang2labels_found = {}

    for split in range(1, 6):
        files = glob.glob(args.random_ixs_dir + f"/split_{split}/*")
        print("- Split", split, len(files))
        for fname in files:

            if len(fname.split(".")) == 2:
                continue

            lang, set_name = os.path.basename(fname).split("_", 1)

            if set_name not in set_names:
                continue

            labels, labels_found = load_labels_from_file(fname, l2i)

            if lang not in stats_dict:
                stats_dict[lang] = {'train_1000': [], 'dev': [], 'test': []}
                lang2labels_found[lang] = {'train_1000': [], 'dev': [], 'test': []}

            lab_ixs, lab_cnts = np.unique(labels, return_counts=True)
            lab_ratios = lab_cnts / lab_cnts.sum()
            stats_dict[lang][set_name].append(lab_cnts)
            lang2labels_found[lang][set_name].append(lab_ixs)

    return stats_dict, lang2labels_found


def main():
    """ main method """

    args = parse_arguments()

    l2i = {}
    with open(args.l2i, "r", encoding="utf-8") as fpr:
        l2i = json.load(fpr)
    i2l = {}
    for l, i in l2i.items():
        i2l[i] = l

    print("-", l2i)

    stats_dict = {}
    lang2labels_found = {}

    if args.dset == "ina":

        stats_dict, lang2labels_found = get_ina_stats(args, l2i)

    elif args.dset == "rcv":

        stats_dict, lang2labels_found = get_rcv_stats(args, l2i)

    else:
        print("dset", args.dset, "not understood")
        sys.exit()

    for lang in stats_dict:
        for s in stats_dict[lang]:
            print(
                lang,
                "{:5s}".format(s),
                lang2labels_found[lang][s],
                np.mean(stats_dict[lang][s], axis=0)
            )
        print("-" * 120)



def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("random_ixs_dir", help="path to dir with random splits")
    parser.add_argument("l2i", default="", help="path to label2int")
    parser.add_argument("dset", type=str, choices=["rcv", "ina"], help="choice of dataset. RCV for multilingual rcv (MLDoc). ina for IndicNLP news articles")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
