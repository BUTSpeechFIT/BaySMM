#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Select sentence indices from the parallel text
which are atleast `certain words long`.
"""

import os
import sys
import codecs
import argparse
import numpy as np


def jap_tokenizer(sent):
    """Japanese tokenizer"""

    try:
        tokens = JAP_TOK.tokenize(sent)
    except UnicodeEncodeError as err:
        print("- export MECAB_CHARSET=utf8", file=sys.stderr)
        sys.exit()

    return [str(t) for t in tokens]


def zh_tokenizer(sent):
    """Chinese tokenizer"""

    words = ZH_TOK.cut(sent)
    return [w.strip() for w in words if w.strip()]


def main():
    """main method"""

    args = parse_arguments()

    if os.path.exists(args.out_file):
        print(args.out_file, 'already exists.')
        sys.exit(0)

    num_words = []
    with codecs.open(args.in_file, "r", "utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            tokens = []
            if args.lang in ("ja", "jp"):
                tokens = jap_tokenizer(line)
            elif args.lang in ("zh", "zhs"):
                tokens = zh_tokenizer(line)
            else:
                tokens = line.split()

            n_tokens = len(tokens)
            num_words.append(n_tokens)

    num_words = np.asarray(num_words).astype(int)
    print("- Total number of lines:", len(num_words))

    sent_ixs = np.where(num_words >= args.msl)[0]

    print(
        "- Num sentences that are atleast",
        args.msl,
        "tokens long:",
        sent_ixs.shape[0],
        "(%{:.1f})".format(sent_ixs.shape[0] * 100.0 / len(num_words)),
    )

    np.savetxt(args.out_file, sent_ixs, fmt="%d")


def parse_arguments():
    """parse arguments"""

    global JAP_TOK, ZH_TOK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_file", help="input parallel file")
    parser.add_argument(
        "lang",
        help="choice of lang. mecab tokenizer of ja/jp, jieba ofr zh/zhs",
    )
    parser.add_argument("msl", default=50, type=int, help="min sent len")
    parser.add_argument("out_file", help="path to output file")
    args = parser.parse_args()

    if args.lang in ("ja", "jp"):

        from konoha import WordTokenizer

        JAP_TOK = WordTokenizer("MeCab")
        print("- Using MeCab tokenizer")

    elif args.lang in ("zh", "zhs"):

        import jieba

        ZH_TOK = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)
        print("- Using jieba tokenizer")

    return args


if __name__ == "__main__":
    main()
