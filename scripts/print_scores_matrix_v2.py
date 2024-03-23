#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Get mean and std. of scores and print the results in
a matrix form.
"""

import os
import sys
import json
import argparse
from pprint import pprint
import numpy as np
from score_utils import (
    print_score_matrix_summary,
    print_transfer_summary,
    print_full_matrix,
    print_latex_table,
    print_latex_table_v2,
    BILSTM_EU,
    MULTICCA,
    LASER_GH,
)


def get_mean_std(scores):
    """ Get mean and std of scores across splits """

    return np.mean(scores, axis=0), np.std(scores, axis=0)


def get_scores(res_file, clf, langs, n_splits):
    """ Load scores from all splits and return mean and std. """

    dev_scores = np.zeros(shape=(n_splits, len(langs)), dtype=np.float32)
    test_scores = np.zeros_like(dev_scores)

    found_ixs = set()

    for i in range(1, n_splits + 1):
        res_file_1 = res_file.replace(f"_1_{clf}", f"_{i}_{clf}")

        if os.path.exists(res_file_1):

            res = {}
            try:
                with open(res_file_1, 'r') as fpr:
                    res = json.load(fpr)
            except json.decoder.JSONDecodeError as err:
                print("JSON decode error:", res_file_1)
                print(str(err))
                sys.exit()

            for lang, acc in res['acc']['dev'].items():
                if lang in langs:
                    dev_scores[i - 1, langs.index(lang)] = acc * 100.
                    found_ixs.add(langs.index(lang))

            for lang, acc in res['acc']['test'].items():
                if lang in langs:
                    test_scores[i - 1, langs.index(lang)] = acc * 100.

        else:
            print(res_file_1, "NOT FOUND.", file=sys.stderr)
            sys.exit()

    # found_ixs = sorted(list(found_ixs))

    # dev_scores = dev_scores[found_ixs, :][:, found_ixs]
    # test_scores = test_scores[found_ixs, :][:, found_ixs]

    return get_mean_std(dev_scores), get_mean_std(test_scores)


def save_scores(fname, scores, header):
    """ Save scores (mean / std) to file """

    np.savetxt(fname, scores, header=header, fmt="%.4f")


def get_score_summary(res_dir, clf, num_train, langs, lids, n_splits=5):
    """ Get score summary, mean, std of dev and test """

    dev_means = np.zeros(shape=(len(lids), len(lids)), dtype=np.float32)
    dev_stds = np.zeros_like(dev_means)
    test_means = np.zeros_like(dev_means)
    test_stds = np.zeros_like(dev_means)

    ntrain = ""

    sfx = ""
    if clf == "mclr":
        sfx = "_R0"

    elif clf in ("mclru", "mclru-0"):
        sfx = "_R32"

    if num_train > 1000 or "mclr" in clf:
        ntrain = f"_{num_train}"

    for i, src_lang in enumerate(langs):

        # dirty fix
        res_file = os.path.join(res_dir, f"{src_lang}{ntrain}_1_{clf}{sfx}.json")
        if not os.path.exists(res_file):
            res_file = os.path.join(res_dir, f"{src_lang}_{num_train}_1_{clf}{sfx}.json")

        # yet another dirty fix for INA dataset
        if not os.path.exists(res_file):
            res_file = os.path.join(res_dir, f"{src_lang}_1_{clf}{sfx}.json")

        if os.path.exists(res_file):
            [dev_means[i], dev_stds[i]], [test_means[i], test_stds[i]] = get_scores(
                res_file, clf, langs, n_splits
            )

        else:
            print(res_file, "NOT FOUND.")

    return dev_means, dev_stds, test_means, test_stds


def print_scores_for_ina(args):
    """Get average scores for IndicNLP-News-Artciles dataset and print them"""

    lid2int = {
        "bn": 0,
        "gu": 1,
        "kn": 2,
        "ml": 3,
        "mr": 4,
        "or": 5,
        "pa": 6,
        "ta": 7,
        "te": 8,
    }

    langs = args.langs
    lids = args.langs

    print(lids)

    ixs = [lid2int[lid] for lid in lids]

    dev_means, dev_stds, test_means, test_stds = get_score_summary(
        args.res_dir, args.clf, args.ntrain, langs, lids
    )

    if args.mean:

        print_score_matrix_summary(
            {
                f"{args.name}+{args.clf.upper()}(test)": test_means,
                f"{args.name}+{args.clf.upper()}( dev)": dev_means,
            }
        )

    if args.indv:
        print_transfer_summary(
            lids,
            {
                f"{args.name}+{args.clf.upper()} (test)": test_means,
                f"{args.name}+{args.clf.upper()} ( dev)": dev_means,
            },
        )

    else:

        if args.set_name == "dev":
            print_full_matrix(lids, dev_means, dev_stds)
        else:
            print_full_matrix(lids, test_means, test_stds)

    if args.latex:

        print_latex_table(
            test_means, lids, str(args.dim), "default", args.set_name
        )

        print_latex_table_v2(
            test_means,
            test_stds,
            lids,
            str(args.dim),
            "default",
            args.set_name,
            args.dev_lim,
        )



def print_scores_for_mldoc(args):
    """Compute average scores for MLDoc and print them"""

    global LASER_GH, MULTICCA

    lid2int = {"EN": 0, "DE": 1, "FR": 2, "IT": 3,
               "ES": 4, "RU": 5, "JA": 6, "ZH": 7}

    lang2lid = {
        "english": "EN",
        "german": "DE",
        "french": "FR",
        "italian": "IT",
        "spanish": "ES",
        "russian": "RU",
        "japanese": "JA",
        "chinese": "ZH",
    }

    lid2lang = {}
    for lang, lid in lang2lid.items():
        lid2lang[lid.lower()] = lang

    langs = [lid2lang[l] for l in args.langs]

    print(langs)

    lids = [lang2lid[l] for l in langs]

    ixs = [lid2int[lid] for lid in lids]

    # cartesian prod of indices
    y_ixs, x_ixs = np.meshgrid(ixs, ixs, indexing='xy')

    LASER_GH = LASER_GH[x_ixs, y_ixs]
    MULTICCA = MULTICCA[x_ixs, y_ixs]


    ntrain = ""

    sfx = ""
    if args.clf == "mclr":
        sfx = "_R0"

    elif args.clf in ("mclr", "mclru", "mclru-0"):
        sfx = "_R32"
        if "_mclr" not in args.res_dir:
            args.res_dir = args.res_dir.replace("results", "results_mclr")

    print("- Loading results from:", args.res_dir)

    if args.ntrain > 1000 or "mclr" in args.clf:
        ntrain = f"_{args.ntrain}"

    dev_means, dev_stds, test_means, test_stds = get_score_summary(
        args.res_dir, args.clf, args.ntrain, langs, lids
    )

    if args.mean:

        print_score_matrix_summary(
            {
                "BiLSTM-EU+MLP": BILSTM_EU,
                "MultiCCA+CNN": MULTICCA,
                "LASER-GH+MLP": LASER_GH,
                f"{args.name}+{args.clf.upper()}(test)": test_means,
                f"{args.name}+{args.clf.upper()}( dev)": dev_means,
            }
        )

    if args.i == "bilstm":
        test_means -= BILSTM_EU
    elif args.i == "cca":
        test_means -= MULTICCA
    elif args.i == "laser":
        test_means -= LASER_GH
    else:
        pass

    if args.indv:
        print_transfer_summary(
            lids,
            {
                "LASER-GH+MLP": LASER_GH,
                f"{args.name}+{args.clf.upper()} (test)": test_means,
                f"{args.name}+{args.clf.upper()} ( dev)": dev_means,
            },
        )
    else:

        if args.set_name == "dev":
            print_full_matrix(lids, dev_means, dev_stds)
        else:
            print_full_matrix(lids, test_means, test_stds)



    if args.i == "default":
        save_scores(
            args.res_dir + f"dev{ntrain}_{args.clf}{sfx}.mean",
            dev_means,
            "    ".join(lids),
        )
        save_scores(
            args.res_dir + f"dev{ntrain}_{args.clf}{sfx}.std",
            dev_stds,
            "   ".join(lids),
        )
        save_scores(
            args.res_dir + f"test{ntrain}_{args.clf}.mean",
            test_means,
            "    ".join(lids),
        )
        save_scores(
            args.res_dir + f"test{ntrain}_{args.clf}.std", test_stds, "   ".join(lids)
        )

    if args.latex:

        print_latex_table(test_means, lids, str(args.dim), args.i, args.set_name)

        print_latex_table_v2(
            test_means,
            test_stds,
            lids,
            str(args.dim),
            args.i,
            args.set_name,
            args.dev_lim,
        )


def main():
    """main method"""

    args = parse_arguments()

    if args.dset == "mldoc":
        print_scores_for_mldoc(args)

    elif args.dset == "ina":
        print_scores_for_ina(args)



def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("res_dir", help="directory with result json files")
    parser.add_argument("set_name", choices=["dev", "test"], help="which set to use?")
    parser.add_argument(
        "-ntrain",
        default=1000,
        type=int,
        choices=[1000, 2000, 5000, 10000],
        help="number of training examples for the classifier. Will pick results corresponding to this setting.",
    )
    parser.add_argument(
        "-clf",
        type=str,
        default="glcu-5",
        choices=["glcu-5", "glcu-0", "glc", "mclr", "mclru", "mclru-0"],
        help="Will pick results corresponding to this classifier"
    )
    parser.add_argument(
        "-name",
        type=str,
        default="MSM",
        help="Model name to be appended before clf name",
    )
    parser.add_argument("-dim", type=int, default=256)
    parser.add_argument(
        "-dev-lim",
        type=float,
        default=3,
        help="bolds the latex cell if the std.dev is higher than this number",
    )
    parser.add_argument(
        "-decimal", type=int, default=2, help="number of decimal digits for rounding"
    )
    parser.add_argument("--latex", action="store_true", help="print latex table")
    parser.add_argument("--mean", action="store_true", help="verbose")
    parser.add_argument("--indv", action="store_true", help="mean of transfer-directions fom each individual language to the rest")
    parser.add_argument("--verbose", action="store_true", help="verbose")

    sub_parsers = parser.add_subparsers(help="dataset name", dest="dset")
    sub_parsers.required = True

    mldoc_parser = sub_parsers.add_parser(
        "mldoc",
        help="MLDoc dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ina_parser = sub_parsers.add_parser(
        "ina",
        help="IndicNLP-News-Artciles dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mldoc_parser.add_argument(
        "-i",
        default="default",
        choices=["default", "bilstm", "cca", "93"],
        help="print absolute differences w.r.t to the published models.",
    )
    mldoc_parser.add_argument(
        "-langs",
        nargs="+",
        default=["en", "de", "fr", "it", "es", "ru", "ja", "zh"],
        help="list of language codes for which the results will be displayed.",
    )

    ina_parser.add_argument(
        "-langs",
        nargs="+",
        default=["bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te"],
        help="list of language codes for which the results will be displayed.",
    )

    args = parser.parse_args()

    if len(args.langs) < 2:
        print("Need atleast two languages.", file=sys.stderr)
        sys.exit()

    return args


if __name__ == "__main__":
    main()
