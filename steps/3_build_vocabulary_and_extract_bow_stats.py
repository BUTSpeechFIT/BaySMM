#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date created : 03 Feb 2022
# Last modified : 03 Feb 2022

"""
Builds vocabulary for each language and extracts BoW stats for parallel data
and also for any other data (RCV, MLDoc, INA) for downstream tasks.
"""

import os
import sys
import stat
import argparse
import numpy as np
from step_utils import read_lang_list, save_cmd_to_file


assert os.environ.get("MBAY_PRE"), "Please set envionment variable MBAY_PRE"
MBAY_PRE = os.environ.get("MBAY_PRE")


def main():
    """main method"""

    args = parse_arguments()

    hash_bang = r"#!/usr/bin/bash"

    langs = read_lang_list(args)

    input_flist_dir = os.path.realpath(args.input_flist_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = os.path.realpath(args.out_dir)

    input_type = args.input_type
    ng = " ".join([str(n) for n in args.ngram_range])
    mdf = args.mdf
    ana = args.ana
    mv = args.mv
    topr = args.topr

    xtr_flist = ""
    if args.xtr_flist_dir:
        args.xtr_flist_dir = os.path.realpath(args.xtr_flist_dir)

    tsks_dir = os.path.join(out_dir, "tsks/")
    os.makedirs(tsks_dir, exist_ok=True)

    sh_files = []

    all_cmds = []

    for lang in langs:
        sh_file = os.path.join(tsks_dir, f"{lang}_bow_xtr.sh")
        log_file = os.path.join(tsks_dir, f"{lang}_bow_xtr.log")

        input_flist = os.path.join(input_flist_dir, f"{lang}.flist")
        xtr_out_dir1 = os.path.join(out_dir, f"{lang}/parallel")
        os.makedirs(xtr_out_dir1, exist_ok=True)

        if args.xtr_flist_dir:
            xtr_out_dir2 = os.path.join(out_dir, f"{lang}/{args.xtr_tag}")
            os.makedirs(xtr_out_dir2, exist_ok=True)
            xtr_flist = os.path.join(args.xtr_flist_dir, f"{lang}.flist")
            if not os.path.exists(xtr_flist):
                xtr_flist = ""

        cmd_pfx = ""
        cmd_sfx = f" > {log_file}"

        if lang in args.pivot_lang:
            idx = args.pivot_lang.index(lang)
            cvect_pkl_file = os.path.realpath(args.pivot_cvect_pkl[idx])
            cmd_core = f"""python3 {MBAY_PRE}/src/extract_features_for_test_data.py \
{lang} {input_flist} {input_type} plain {cvect_pkl_file} {xtr_out_dir1} -topr {topr}"""

        else:
            cmd_core = f"""python3 {MBAY_PRE}/src/feature_xtr_from_text_spm_v2.py \
{input_flist} {input_type} {lang} {out_dir}/ -extract_list {xtr_flist} \
-xtr_out_dir {xtr_out_dir2} -ng {ng} -mdf {mdf} -ana {ana} -mv {mv} -topr {topr}"""

        if args.local:
            all_cmds.append(cmd_core)
        else:
            cmd = hash_bang + "\n" + cmd_pfx + cmd_core + cmd_sfx
            save_cmd_to_file(cmd, sh_file)
            sh_files.append(sh_file)

    if args.local:
        import subprocess

        for cmd in all_cmds:
            process = subprocess.Popen(cmd.split())
            output, err = process.communicate()

    else:
        tsk_file = os.path.join(tsks_dir, f"bow-xtr.tsk")
        with open(tsk_file, "w", encoding="utf-8") as fpw:
            fpw.write("\n".join(sh_files) + "\n")

        print(f"\n- Run {tsk_file} on local machine or", "submit it to a cluster.")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_flist_dir",
        help="path to flist dir using which vocab will be built. this dir was the output of step 2",
    )

    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument(
        "-lang_list_file",
        type=str,
        help="File with list of languages (two letter ISO codes) to use. Space separated.",
    )
    lang_group.add_argument(
        "-lang_list",
        type=str,
        nargs="+",
        help="List of languages (two letter ISO codes) to use. Space separated.",
    )

    parser.add_argument(
        "out_dir", help="out dir where vocab and BoW stats will be saved."
    )

    parser.add_argument(
        "-xtr_flist_dir",
        type=str,
        default="",
        help="path to flist dir for which BoW stats will be extracted. Vocabulary will not be built on this data.",
    )
    parser.add_argument(
        "-xtr_tag",
        type=str,
        default="",
        help="name tag for the xtr flist. this is required when xtr-flist-dir is given",
    )

    parser.add_argument(
        "-pivot_lang",
        type=str,
        nargs="+",
        default="",
        help="two letter ISO lang code for pivot language(s), for which vocab and countvectorizer already exists",
    )
    parser.add_argument(
        "-pivot_cvect_pkl",
        type=str,
        nargs="+",
        default="",
        help="path to CountVectorizer pkl file for the pivot languages(s)",
    )

    parser.add_argument(
        "-mdf", type=int, default=2, help="min doc frequency constraint"
    )
    parser.add_argument(
        "-ngram-range",
        type=int,
        nargs=2,
        default=[1, 1],
        choices=[1, 2, 3],
        help="ngram range",
    )

    parser.add_argument(
        "-ana",
        default="word",
        choices=["word", "char_wb"],
        help="analyzer for count vectorizer or BoW",
    )

    parser.add_argument(
        "-mv",
        type=int,
        default=100000,
        help="max vocab size in case of bag-of-words extracted using sklearn CountVectorizer",
    )

    parser.add_argument(
        "-chunk_size",
        type=int,
        default=200000,
        help="save large files in chunks (num rows) for easy loading while training the multilingual model.",
    )

    parser.add_argument("-input_type", type=str, default="flist", choices=["flist"])

    parser.add_argument(
        "-topr",
        type=float,
        default=1.0,
        choices=np.arange(1, 11, 1) / 10.0,
        help="ratio of number of sentences per flist to be considered. Useful in preparing training sets of different sizes.",
    )

    parser.add_argument(
        "--local", action="store_true", help="Creates tsk files to submit to a cluster"
    )

    parser.add_argument("--verbose", action="store_true", help="increase verbosity")

    args = parser.parse_args()

    if args.xtr_flist_dir:
        if not args.xtr_tag:
            print("-xtr_tag is required when -xtr_flist_dir is given")
            sys.exit()

    if len(args.pivot_lang) != len(args.pivot_cvect_pkl):
        print(
            "- Input arg error: Number of args for -pivot-lang ({:d}) should match the number of \
args for -pivot-cvect-pkl ({:d})".format(
                len(args.pivot_lang), len(args.pivot_cvect_pkl)
            )
        )
        sys.exit()

    return args


if __name__ == "__main__":
    main()
