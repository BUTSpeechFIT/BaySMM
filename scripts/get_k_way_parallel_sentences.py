#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Get K-way parallel sentences from bitext pairs
Works for Europarl, MultiUN
"""

import os
import re
import sys
import argparse
import numpy as np


def load_lines_to_dict(fname):
    """Load lines and return line to int mapping"""

    dup_lines = 0
    line2int = {}
    lno = 0
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            if line and line not in line2int:
                line2int[line] = lno
            else:
                dup_lines += 1
            lno += 1
    print("-", fname, ": unique lines:", len(line2int), "duplicate lines:", dup_lines)
    return line2int


def load_lines(fname):
    """Load lines and return int to line mapping"""

    lines = []
    empty_lnos = set()
    lno = 0
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            lines.append(line)
            if not line:
                empty_lnos.add(lno)
            lno += 1
    print("-", fname, "lines:", len(lines), "- empty lines:", len(empty_lnos))
    return lines, empty_lnos


def get_int2line(line2int):
    int2line = {}
    for line, idx in line2int.items():
        int2line[idx] = line
    return int2line


def save_matching_lines(ixs, lines, empty_ixs, out_file):
    with open(out_file, "w", encoding="utf-8") as fpw:
        for i in ixs:
            if i in empty_ixs:
                continue
            else:
                fpw.write(lines[i] + "\n")
    print("-", out_file, "saved.")


def main():
    """main method"""

    args = parse_arguments()

    base = args.dataset

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pivot_lang in args.other_langs:
        args.other_langs.remove(args.pivot_lang)

    # pivot language
    pivot = args.pivot_lang
    common_lang_ext = "." + pivot

    # the language of the mono file should be same across bitext pairs
    mono_files_from_bitext_pairs = []
    oth_lang_exts = []
    for oth in args.other_langs:
        if oth < pivot:
            fname = os.path.join(
                args.in_dir, f"{oth}-{pivot}/{base}.{oth}-{pivot}.{pivot}"
            )
        else:
            fname = os.path.join(
                args.in_dir, f"{pivot}-{oth}/{base}.{pivot}-{oth}.{pivot}"
            )

        if os.path.exists(fname):
            mono_files_from_bitext_pairs.append(fname)
        else:
            print("-", fname, "not found.")
            sys.exit()
        oth_lang_exts.append("." + oth)

    n_langs = len(oth_lang_exts) + 1
    header = ""
    for oth in oth_lang_exts:
        header += common_lang_ext[1:] + "-" + oth[1:] + " "

    # line_num to text mapping for every mono file from bitext pair
    line2text = {}
    for i, fname in enumerate(mono_files_from_bitext_pairs):
        line2text[i] = load_lines_to_dict(fname)
    print("- line2text:", len(line2text))

    text_lines = {}
    all_empty = set()

    cmn_lines, cmn_empty = load_lines(mono_files_from_bitext_pairs[0])
    all_empty |= cmn_empty

    for i, fname in enumerate(mono_files_from_bitext_pairs):
        txt_lines, emp_lnos = load_lines(
            re.sub(common_lang_ext + r"$", oth_lang_exts[i], fname)
        )
        text_lines[i] = txt_lines
        all_empty |= emp_lnos

    print("- All empty lines to be ignored:", len(all_empty))

    ixs = []

    for line in line2text[0]:
        flag = True
        oth_ixs = []
        for i in line2text:
            if line in line2text[i]:
                oth_ixs.append(line2text[i][line])
                flag = True
            else:
                flag = False
                break

        if (
            flag
            and len(set(oth_ixs) & all_empty) == 0
            and len(oth_ixs) == len(line2text)
        ):
            ixs.append(oth_ixs)

        print(
            "\r  - {:2d}-way matching lines found: {:7d}".format(
                len(oth_lang_exts) + 1, len(ixs)
            ),
            end="",
        )

    print()
    ixs = np.asarray(ixs).astype(np.int64)
    np.savetxt(
        os.path.join(args.out_dir, str(n_langs) + "-way-ixs.txt"),
        ixs,
        fmt="%d",
        header=header,
    )

    save_matching_lines(
        ixs[:, 0],
        cmn_lines,
        all_empty,
        os.path.join(args.out_dir, f"{base}-{n_langs}-way.{pivot}"),
    )
    for i in line2text:
        save_matching_lines(
            ixs[:, i],
            text_lines[i],
            all_empty,
            os.path.join(args.out_dir, f"{base}-{n_langs}-way.{args.other_langs[i]}"),
        )

    print("- Saved in", args.out_dir)


def parse_arguments():
    """parse arguments"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_dir",
        required=True,
        help="path to dir where the parallel corpus exists (eg: Europarl/)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="dataset basename (assumes every file name has the pattern:\
lang1-lang2/dataset.lang1-lang2.langX)",
    )
    parser.add_argument("--out_dir", required=True, help="path to out dir")
    parser.add_argument(
        "--pivot_lang",
        choices=["en", "de", "fr", "es", "it", "hi", "bn", "ta", "kn", "ar"],
        help="pivot language ID (two char ISO) to which bitexts to other langs exist.",
    )
    parser.add_argument(
        "--other_langs",
        nargs="+",
        required=True,
        type=str,
        help="other lang IDs (two char ISO)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
