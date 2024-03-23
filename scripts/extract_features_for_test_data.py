#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

"""
Extract features (counts) for new/test data using existing
CountVectorizer object (vocab + config).
"""

import os
import sys
import csv
import json
import pickle
import argparse
from pprint import pprint
import numpy as np
from scipy import sparse
from pylibs.misc.io import read_simple_flist
from feature_xtr_from_text_spm_v2 import save_as_chunks


def jap_tokenizer(sent):
    """Japanese tokenizer"""

    try:
        tokens = JAP_TOK.tokenize(sent)
    except UnicodeEncodeError as err:
        print("- Set MECAB_CHARSET=utf8", file=sys.stderr)
        sys.exit()

    return [str(t) for t in tokens]


def zh_tokenizer(sent):
    """Chinese tokenizer"""

    words = ZH_TOK.cut(sent)
    return [w.strip() for w in words if w.strip()]


def load_l2i(l2i_file):
    """Load label2int"""

    l2i = {}
    if os.path.exists(l2i_file):
        with open(l2i_file, "r", encoding="utf-8") as fpr:
            l2i = json.load(fpr)
        print("Loaded label2int:", l2i)
    else:
        print(l2i_file, "NOT FOUND.", file=sys.stderr)
        sys.exit()

    return l2i


def load_docs_and_labels_from_csv(csv_file, l2i):
    """Load docs and labels from csv file"""

    docs = []
    labels = []
    with open(csv_file, "r", encoding="utf-8") as fpr:
        csvreader = csv.reader(fpr, delimiter=",")
        for row in csvreader:
            docs.append(row[1])
            try:
                label = l2i[row[0]]
            except KeyError:
                print(row[0], "not in label2int:\n", l2i, file=sys.stderr)
                sys.exit()
            labels.append(label)

    # print("# docs  :", len(docs))
    # print("# labels:", len(labels))
    return docs, labels


def main():
    """main method"""

    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    l2i = {}
    if args.l2i_file:
        l2i = load_l2i(args.l2i_file)

    with open(args.cvect_pkl_file, "rb") as fpr:
        cvect = pickle.load(fpr)
    parts = os.path.basename(args.cvect_pkl_file).split(".")[0].split("_")
    sfx = "_counts_" + "_".join(parts[2:])

    lid = parts[0]
    xtr_json = {}

    if args.out_xtr_json:
        if os.path.exists(args.out_xtr_json):
            with open(args.out_xtr_json, "r") as fpr:
                xtr_json = json.load(fpr)
        print("Loaded:", args.out_xtr_json)
        pprint(xtr_json)

    if lid not in xtr_json:
        xtr_json[lid] = []

    in_files = []
    if args.input_type == "text":
        in_files = [args.input_file]
    else:
        in_files = read_simple_flist(args.input_file)

    if args.verbose:
        print("Input files:", len(in_files))

    for i, in_file in enumerate(in_files):

        print(
            "- Loading {:3d}/{:3d} {:s}".format(i + 1, len(in_files), in_file),
        )

        if args.out_base:
            base = args.out_base
        else:
            base = os.path.basename(in_file).rsplit(".", 1)[0]

        if args.file_type == "plain":
            docs = read_simple_flist(in_file)
            toprk = int(args.topr * len(docs))
            if args.topr < 1:
                print(len(docs), end=" ")
                docs = docs[:toprk]
                print(len(docs))

        elif args.file_type == "csv":

            docs, labels = load_docs_and_labels_from_csv(in_file, l2i)
            toprk = int(args.topr * len(docs))
            docs = docs[:toprk]
            labels = labels[:toprk]

            if args.verbose:
                print(
                    "labels:",
                    len(labels),
                    "unique labels:",
                    np.unique(labels),
                    "l2i:",
                    len(l2i),
                )
            np.savetxt(os.path.join(args.out_dir, f"{base}.labels"), labels, fmt="%d")

        counts = cvect.transform(docs)

        # npz_file = os.path.join(os.path.realpath(args.out_dir), f"{base}{sfx}.npz")
        # sparse.save_npz(npz_file, counts)

        out_base_name = os.path.join(os.path.realpath(args.out_dir), f"{base}{sfx}")
        chunk_fnames = save_as_chunks(counts, out_base_name, args.chunk_size)

        if args.verbose:
            print("Counts:", counts.shape, end=" ")
        # print("->", npz_file, "saved.")

        xtr_json[lid].extend(chunk_fnames)

        if args.tfidf_pkl_file:
            with open(args.tfidf_pkl_file, "rb") as fpr:
                tfidf = pickle.load(fpr)
            tfidf_stats = tfidf.transform(counts)
            print("Tfidf:", tfidf_stats.shape, end=" ")
            parts = os.path.basename(args.tfidf_pkl_file).split(".")[0].split("_")
            sfx = "_" + "_".join(parts[1:])
            out_base = os.path.join(args.out_dir, f"{base}{sfx}")
            save_as_chunks(tidf_stats, out_base, args.chunk_size)
            print("Saved to:", out_base)

    if args.out_xtr_json:

        with open(args.out_xtr_json, "w") as fpa:
            json.dump(xtr_json, fpa, indent=2)
        print("\n" + args.out_xtr_json, "saved.")

    print("\n= Done =")


def parse_arguments():
    """Parse command line arguments"""

    global JAP_TOK, ZH_TOK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lang",
        type=str,
        choices=[
            "en",
            "de",
            "fr",
            "it",
            "es",
            "ru",
            "zh",
            "zh_cn",
            "pl",
            "ja",
            "ar",
            "ka",
            "rw",
            "te",
            "kn",
            "ur",
            "ta",
            "tr",
            "or",
            "ml",
            "mr",
            "bn",
            "gu",
            "hi",
            "pa",
        ],
        help="ISO 639-1 language code\
 (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)",
    )
    parser.add_argument("input_file", help="path to extract flist or file")
    parser.add_argument("input_type", choices=["text", "flist"], help="input type")
    parser.add_argument("file_type", choices=["plain", "csv"], help="file type")
    parser.add_argument("cvect_pkl_file", help="path to count vectorizer pkl")
    parser.add_argument("out_dir", help="path to out dir")
    parser.add_argument(
        "-topr",
        type=float,
        default=1.0,
        choices=np.arange(1, 11, 1) / 10.0,
        help="ratio of number of sentences per flist to be considered. Useful in preparing training sets of different sizes.",
    )

    parser.add_argument(
        "-out_base",
        default="",
        help="basename for output file. \
Will use the same base name as input file if not given. This option is valid only \
 when `input_type` is text. suffix will be inferred from cvect_pkl_file and appended.",
    )
    parser.add_argument(
        "-l2i_file",
        default="",
        help="required label2int file in case the file_type is csv",
    )
    parser.add_argument(
        "-tfidf_pkl_file", default="", help="path to tfidf vectorizer pkl"
    )

    parser.add_argument(
        "-out_xtr_json",
        default="",
        help="saves npz and vocab paths in the given json file. It will be useful for extracting embeddings.",
    )
    parser.add_argument(
        "-chunk_size",
        type=int,
        default=200000,
        help="save large files in xtr_flist in chunks for easy loading while training the multilingual model.",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.file_type == "csv":
        if args.l2i_file:
            if not os.path.exists(args.l2i_file):
                print(args.l2i_file, "NOT FOUND.", file=sys.stderr)
                sys.exit()
        else:
            print(
                "\nINFO: -l2i_file <path to label2int.json> is required in case the input is csv file with labels.\n"
            )
            sys.exit()

    if args.input_type == "flist" and args.out_base != "":
        print(
            "arg out_base can be used only when input_type is text and not flist",
            file=sys.stderr,
        )
        sys.exit()

    if args.lang == "ja":

        from konoha import WordTokenizer

        JAP_TOK = WordTokenizer("MeCab")

    elif args.lang in ("zh", "zh_cn"):
        import jieba

        ZH_TOK = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)

    return args


if __name__ == "__main__":

    main()
