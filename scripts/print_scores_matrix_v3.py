#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Get mean and var. of scores and print the results in
a matrix form.
"""

import os
import sys
import json
import argparse
from pprint import pprint
import numpy as np

# import termtables as tt
np.set_printoptions(formatter={"all": lambda x: "{:6.2f}".format(x)})


def print_latex_table_v2(scores, var, lids, dim, impr, set_name, var_lim):
    """Print results in latex table format"""

    n_rows = "l" * len(lids)

    beg = (
        r"""
\begin{table}[h!]
 \def\arraystretch{1}
 \centering
 \begin{tabular}{c|"""
        + n_rows
        + r"""} \toprule
 & \multicolumn{"""
        + str(len(lids))
        + r"""}{c}{\textsc{Test language}} \\ """
    )

    info_1 = r""""""
    # & & \multicolumn{5}{c}{\textsc{~146K parallel sentences}} \\ \midrule """

    print(beg)
    print(r"  & ", end=" ")
    for lid in lids:
        print(f"{lid}", end=" ")
        if lid != lids[-1]:
            print("& ", end="")
        else:
            print(r"\\ \midrule", end="")
    print(info_1)
    # print(r"  \multirow{" + str(len(lids)) + "}{*}{" + dim + r"}")

    end_str = (
        r"""  \bottomrule
 \end{tabular}
 \caption{Average classification accuracy on the """
        + set_name
        + r""" set. Embedding dimension is $"""
        + dim
        + r"""$.}
 \label{tab:multi}
\end{table}
    """
    )

    add_sign = lambda x: "{:.2f}".format(x) if x > 0 else "{:.2f}".format(x)
    make_bold = (
        lambda x: "\\textbf{" + "{:.2f}".format(x) + "}"
        if x > 0
        else "{:.2f}".format(x)
    )
    fmt = {"float_kind": add_sign}
    for i, lid in enumerate(lids):
        if impr != "default":
            fmt = {"float_kind": make_bold}

        # print(
        #     r"   ",
        #     f"\{lid}" + r" &",
        #     np.array2string(scores[i, :], precision=2, formatter=fmt, separator=r" & ")[
        #         1:-1
        #     ],
        #     r"\\",
        # )
        print(
            r"  ",
            f"{lid}" + r" &", end=" ")
        for j, _ in enumerate(scores[i, :]):
            esym = r" &"
            if j == scores.shape[1]-1:
                esym = r" "

            if i == j:
                print(
                    r"\cellcolor{lightgray}{"+"{:.1f} ({:.1f})".format(scores[i, j], var[i, j]) + r"}" + esym, end=""
                )
            else:
                if var[i, j] > var_lim:
                    print(
                        r"\textbf{"+"{:.1f} ({:.1f})".format(scores[i, j], var[i, j]) + r"}" + esym, end=""
                    )
                else:
                    print(
                        " {:.1f} ({:.1f})".format(scores[i, j], var[i, j]) + esym, end=""
                    )

        print(r"\\")
    print(end_str)


def print_latex_table(scores, lids, dim, impr, set_name):
    """ Print results in latex table format """

    n_rows = "r" * len(lids)

    beg = r"""
\begin{table}[h!]
 \def\arraystretch{1.4}
 \centering
 \begin{tabular}{c|""" + n_rows + r"""} \toprule
 & \multicolumn{""" + str(len(lids)) + r"""}{c}{\textsc{Test language}} \\ """

    info_1 = r""""""
    # & & \multicolumn{5}{c}{\textsc{~146K parallel sentences}} \\ \midrule """

    print(beg)
    print(r"  & ", end=" ")
    for lid in lids:
        print(rf"\{lid}",  end=" ")
        if lid != lids[-1]:
            print("& ", end="")
        else:
            print(r"\\ \midrule", end="")
    print(info_1)
    # print(r"  \multirow{" + str(len(lids)) + "}{*}{" + dim + r"}")

    end_str = (
        r"""  \bottomrule
 \end{tabular}
 \caption{Average classification accuracy on the """ + set_name + r""" set. Embedding dimension is $"""
        + dim
        + r"""$.}
 \label{tab:multi}
\end{table}
    """
    )

    add_sign = lambda x: "{:.2f}".format(x) if x > 0 else "{:.2f}".format(x)
    make_bold = (
        lambda x: "\\textbf{" + "{:.2f}".format(x) + "}"
        if x > 0
        else "{:.2f}".format(x)
    )
    fmt = {"float_kind": add_sign}
    for i, lid in enumerate(lids):
        if impr != "default":
            fmt = {"float_kind": make_bold}

        print(
            r"   ",
            f"\{lid}" + r" &",
            np.array2string(scores[i, :], precision=2, formatter=fmt, separator=r" & ")[
                1:-1
            ],
            r"\\",
        )

    print(end_str)


def get_mean_var(scores):
    """ Get mean and var of scores across splits """

    return np.mean(scores, axis=0), np.var(scores, axis=0)


def get_scores(res_file, clf, langs, n_splits):
    """ Load scores from all splits and return mean and variance """

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

    return get_mean_var(dev_scores), get_mean_var(test_scores)


def save_scores(fname, scores, header):
    """ Save scores (mean / var) to file """

    np.savetxt(fname, scores, header=header, fmt="%.4f")


def get_score_summary(res_dir, clf, num_train, langs, lids, n_splits=5):
    """ Get score summary, mean, var of dev and test """

    dev_means = np.zeros(shape=(len(lids), len(lids)), dtype=np.float32)
    dev_vars = np.zeros_like(dev_means)
    test_means = np.zeros_like(dev_means)
    test_vars = np.zeros_like(dev_means)

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
            [dev_means[i], dev_vars[i]], [test_means[i], test_vars[i]] = get_scores(
                res_file, clf, langs, n_splits
            )

        else:
            print(res_file, "NOT FOUND.")

    return dev_means, dev_vars, test_means, test_vars


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

    n_dashes = 16 + (len(langs) * 8)

    dev_means, dev_vars, test_means, test_vars = get_score_summary(
        args.res_dir, args.clf, args.ntrain, langs, lids
    )

    if args.mean:
        print("-" * 58)
        print("Mean of   \t    Full    \t  Off   \t   Diag ")
        print("-" * 58)

        off_ixs = np.where(~np.eye(dev_means.shape[0], dtype=bool))
        print(
            "{:17s}:".format(args.name + "+" + args.clf.upper() + " dev"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(dev_means),
                np.var(dev_means),
                np.mean(dev_means[off_ixs]),
                np.var(dev_means[off_ixs]),
                np.mean(np.diag(dev_means)),
                np.var(np.diag(dev_means)),
            ),
        )
        print(
            "{:17s}:".format(args.name + "+" + args.clf.upper() + " test"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(test_means),
                np.var(test_means),
                np.mean(test_means[off_ixs]),
                np.var(test_means[off_ixs]),
                np.mean(np.diag(test_means)),
                np.var(np.diag(test_means)),
            ),
        )
        print("-" * 58, "\n")

    print("-" * n_dashes)
    if args.indv:
        print("{:17s}".format(args.set_name.upper()), "      ".join(lids))
        print("-" * n_dashes)
        print("{:17s}".format(args.name + "+" + args.clf.upper()), end="")

    else:
        print("{:6s}".format(args.set_name.upper()), "      ".join(lids))
        print("-" * n_dashes)

    for i, lid in enumerate(lids):

        if args.set_name == "dev":
            if args.indv:
                ixs_2 = np.delete(np.arange(len(lids), dtype=int), i)
                print(
                    np.array2string(
                        np.mean(dev_means[i, ixs_2]), precision=2, separator=" "
                    ),
                    end="  ",
                )
            else:
                print(
                    lid
                    + "  "
                    + np.array2string(
                        dev_means[i, :], precision=2, sign="+", separator="  "
                    )
                )
        else:
            if args.indv:
                ixs_2 = np.delete(np.arange(len(lids), dtype=int), i)
                print(
                    np.array2string(
                        np.mean(test_means[i, ixs_2], axis=0),
                        precision=2,
                        separator="  ",
                    ),
                    end="  ",
                )
            else:
                print(
                    lid
                    + "  "
                    + np.array2string(
                        test_means[i, :], precision=2, sign="+", separator="  "
                    )
                )
    if args.indv:
        print("    ")
    print("-" * n_dashes)

    # tt.print(test_means, header=lids, style=tt.styles.thin, alignment="r", precision=2)

    if args.latex:
        cfg_file = args.res_dir + "/../config.json"
        config = {}
        with open(cfg_file, "r") as fpr:
            config = json.load(fpr)
        print_latex_table(
            test_means, lids, str(config["hyper"]["K"]), "default", args.set_name
        )

def print_scores_for_mldoc(args):
    """Compute average scores for MLDoc and print them"""

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

    # seq: EN, DE, FR, IT, ES
    # This is taken from MLDoc paper Schwenk and Li 2018, (BiLSTM-EU)
    # http://www.lrec-conf.org/proceedings/lrec2018/summaries/658.html
    bilstm = np.asarray(
        [
            [88.40, 71.83, 72.80, 60.73, 66.65],
            [71.52, 92.03, 75.45, 56.45, 75.50],
            [76.00, 78.42, 89.75, 63.70, 70.70],
            [67.15, 66.22, 65.07, 82.88, 67.07],
            [62.70, 71.05, 62.67, 57.93, 88.28],
        ]
    )

    # This is taken from MLDoc paper Schwenk and Li 2018
    # http://www.lrec-conf.org/proceedings/lrec2018/summaries/658.html
    # seq: EN, DE, FR, IT, ES, RU, JA, ZH
    cca = np.asarray(
        [
            [92.20, 81.20, 72.38, 69.38, 72.50, 60.80, 67.63, 74.73],
            [55.95, 93.70, 71.55, 63.98, 73.23, 44.83, 60.18, 55.45],
            [64.80, 53.70, 92.50, 61.15, 65.40, 40.75, 37.75, 38.35],
            [53.70, 49.20, 62.25, 85.55, 58.68, 35.38, 45.30, 32.13],
            [74.00, 55.80, 65.63, 58.35, 94.45, 45.53, 43.40, 41.63],
            [72.50, 40.30, 44.60, 42.70, 41.03, 85.65, 39.68, 42.38],
            [54.90, 52.70, 48.30, 44.33, 54.28, 40.85, 85.35, 44.78],
            [56.00, 48.70, 53.58, 47.18, 35.53, 40.45, 50.63, 87.30],
        ]
    )


    # This is LASER+MLP. Results are taken from the github page
    # https://github.com/facebookresearch/LASER/tree/main/tasks/mldoc
    # seq: EN, DE, FR, IT, ES, RU, JA, ZH
    bilstm_93 = np.asarray(
        [
            [90.73, 86.25, 78.03, 70.20, 79.30, 67.25, 60.95, 70.98],
            [80.75, 92.70, 82.83, 73.25, 79.60, 68.18, 56.80, 72.90],
            [80.08, 87.03, 90.80, 71.08, 78.40, 67.55, 53.60, 66.12],
            [74.15, 80.73, 78.35, 85.93, 82.60, 68.83, 55.15, 56.10],
            [69.58, 79.73, 75.30, 71.10, 88.75, 59.83, 59.65, 61.70],
            [72.60, 79.62, 71.28, 67.00, 68.18, 84.65, 59.23, 65.62],
            [68.45, 81.90, 67.95, 57.98, 67.95, 53.70, 85.15, 66.12],
            [77.95, 83.38, 75.83, 70.33, 78.38, 66.62, 55.25, 88.98],
        ]
    )

    bilstm_93 = bilstm_93[x_ixs, y_ixs]
    cca = cca[x_ixs, y_ixs]

    ntrain = ""

    sfx = ""
    if args.clf == "mclr":
        sfx = "_R0"

    elif args.clf in ("mclr", "mclru", "mclru-0"):
        sfx = "_R32"
        if "_mclr" not in args.res_dir:
            args.res_dir = args.res_dir.replace("results", "results_mclr")

    print(args.res_dir)

    if args.ntrain > 1000 or "mclr" in args.clf:
        ntrain = f"_{args.ntrain}"

    dev_means, dev_vars, test_means, test_vars = get_score_summary(
        args.res_dir, args.clf, args.ntrain, langs, lids
    )

    n_dashes = 16 + (len(langs) * 8)

    if args.mean:
        off_ixs = np.where(~np.eye(bilstm.shape[0], dtype=bool))
        print("-" * 58)
        print("Mean of   \t    Full  \t  Off  \t    Diag ")
        print("-" * 58)
        print(
            "{:17s}".format("BiLSTM-EU"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(bilstm),
                np.var(bilstm),
                np.mean(bilstm[off_ixs]),
                np.var(bilstm[off_ixs]),
                np.mean(np.diag(bilstm)),
                np.var(np.diag(bilstm)),
            ),
        )
        off_ixs = np.where(~np.eye(bilstm_93.shape[0], dtype=bool))
        print(
            "{:17s}".format("LASER+MLP"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(bilstm_93),
                np.var(bilstm_93),
                np.mean(bilstm_93[off_ixs]),
                np.var(bilstm_93[off_ixs]),
                np.mean(np.diag(bilstm_93)),
                np.var(np.diag(bilstm_93)),
            ),
        )
        off_ixs = np.where(~np.eye(cca.shape[0], dtype=bool))
        print(
            "{:17s}".format("MultiCCA+CNN"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(cca),
                np.var(cca),
                np.mean(cca[off_ixs]),
                np.var(cca[off_ixs]),
                np.mean(np.diag(cca)),
                np.var(np.diag(cca)),
            ),
        )
        off_ixs = np.where(~np.eye(dev_means.shape[0], dtype=bool))
        print(
            "{:17s}".format(args.clf.upper() + " dev"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(dev_means),
                np.var(dev_means),
                np.mean(dev_means[off_ixs]),
                np.var(dev_means[off_ixs]),
                np.mean(np.diag(dev_means)),
                np.var(np.diag(dev_means)),
            ),
        )
        print(
            "{:17s}".format(args.clf.upper() + " test"),
            "{:5.2f} ({:5.2f})  {:5.2f} ({:5.2f}) {:5.2f} ({:5.2f})".format(
                np.mean(test_means),
                np.var(test_means),
                np.mean(test_means[off_ixs]),
                np.var(test_means[off_ixs]),
                np.mean(np.diag(test_means)),
                np.var(np.diag(test_means)),
            ),
        )
        print("-" * 58, "\n")

    if args.i == "bilstm":
        dev_means -= bilstm
        test_means -= bilstm
    elif args.i == "cca":
        dev_means -= cca
        test_means -= cca
    elif args.i == "93":
        dev_means -= bilstm_93
        test_means -= bilstm_93
    else:
        pass

    print("-" * n_dashes)
    print("{:5s} ".format(args.set_name.upper()), "      ".join(lids))
    print("-" * n_dashes)

    if args.indv:
        print("{:15s}".format("LASER+MLP"), end="")
        for i, lid in enumerate(lids):
            ixs_2 = np.delete(np.arange(len(lids), dtype=int), i)
            print(np.array2string(np.mean(bilstm_93[i, ixs_2]), precision=2, separator=" "), end="  ")

        print("\n{:15s}".format(args.name+"+"+args.clf.upper()), end="")

    for i, lid in enumerate(lids):

        if args.set_name == "dev":
            if args.indv:
                ixs_2 = np.delete(np.arange(len(lids), dtype=int), i)
                print(np.array2string(np.mean(dev_means[i, ixs_2]), precision=2, separator=" "), end="  ")
            else:
                print(
                    lid
                    + "  "
                    + np.array2string(
                        dev_means[i, :], precision=2, sign="+", separator="  "
                    )
                )
        else:
            if args.indv:
                ixs_2 = np.delete(np.arange(len(lids), dtype=int), i)
                print(np.array2string(np.mean(test_means[i, ixs_2], axis=0), precision=2, separator="  "), end="  ")
            else:
                print(
                    lid
                    + "  "
                    + np.array2string(
                        test_means[i, :], precision=2, sign="+", separator="  "
                    )
                )
    if args.indv:
        print("    ")
    print("-" * n_dashes)

    # tt.print(test_means, header=lids, style=tt.styles.thin, alignment="r", precision=2)

    if args.i == "default":
        save_scores(
            args.res_dir + f"dev{ntrain}_{args.clf}{sfx}.mean",
            dev_means,
            "    ".join(lids),
        )
        save_scores(
            args.res_dir + f"dev{ntrain}_{args.clf}{sfx}.var",
            dev_vars,
            "   ".join(lids),
        )
        save_scores(
            args.res_dir + f"test{ntrain}_{args.clf}.mean",
            test_means,
            "    ".join(lids),
        )
        save_scores(
            args.res_dir + f"test{ntrain}_{args.clf}.var", test_vars, "   ".join(lids)
        )
        # print(
        #    f"[dev,test]{ntrain}_{args.clf}.[mean,var] scores saved in {args.res_dir}"
        # )

    if args.latex:
        cfg_file = args.res_dir + "/../config.json"
        config = {}
        with open(cfg_file, "r") as fpr:
            config = json.load(fpr)
        print_latex_table(test_means, lids, str(args.dim), args.i, args.set_name)

        print_latex_table_v2(
            test_means, test_vars, lids, str(args.dim), args.i, args.set_name, args.var_lim
        )

    # print()


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
        default="MODEL",
        help="Model name to be appended before clf name",
    )
    parser.add_argument("-dim", type=int, default=1024)
    parser.add_argument("-var-lim", type=float, default=6, help="bolds the latex cell if the variance is higher than this number")
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
