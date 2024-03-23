#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Util functions for Indic News Articles dataset
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pylibs.misc.io import get_ivectors_from_h5


LANGS = ["bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te"]


L2I = {
    "entertainment": 0,
    "sports": 1,
    "business": 2,
    "lifestyle": 3,
    "technology": 4,
    "crime": 5,
    "politics": 6,
}


def check_for_existing_results_file(args, sfx=""):
    """Check for existing results file"""

    scores_dict = {"acc": {}, "xen": {}}
    proba_dict = {}

    res_base = f"{args.in_lang}_{args.split_ix}_{args.clf}"
    # res_base = f"{args.in_lang}_{args.clf}"
    if args.clf == "glcu":
        res_base += f"-{args.trn}"

    res_file = os.path.join(args.out_dir, f"{res_base}{sfx}.json")
    proba_file = os.path.join(args.out_dir, f"{res_base}{sfx}_proba.json")

    if os.path.exists(res_file):
        if not args.ovr and not args.append:
            print("- Results file exists:", res_file, "")
            sys.exit(0)

        if args.append:
            print("- Loading results file from:", res_file)
            with open(res_file, "r") as fpr:
                scores_dict = json.load(fpr)
            proba_file = res_file.replace(".json", "_proba.json")
            print("- Loading proba   file from:", proba_file)
            with open(proba_file, "r") as fpr:
                proba_dict = json.load(fpr)

    return res_file, proba_file, scores_dict, proba_dict


def infer_sfx(emb_dir, lang):
    """Infer common suffix from files"""

    fnames = glob.glob(emb_dir + f"/{lang}*.h5")
    if fnames:
        fname = fnames[0]
        base = os.path.basename(fname).split(".")[0]
        parts = base.split("_")
        try:
            cnt_ix = parts.index("counts")
        except ValueError:
            cnt_ix = parts.index("bow")

        if parts[cnt_ix - 1] == "bpe":
            cnt_ix -= 1
        model_ix = parts.index("model")
        sfx = "_".join(parts[cnt_ix : model_ix + 1])
        return "_" + sfx + "_"

    else:
        return None


def get_feat_fname(emb_dir, lang, config, set_name, args):
    """Return filename of embeddings h5 file or None if file does not exists."""

    sfx = infer_sfx(emb_dir, lang)
    # print('lang:', lang, 'inner_sfx:', sfx)

    fname = None
    if sfx:
        model_iters = args.model_iters if args.model_iters else config["trn_done"]
        xtr_iters = args.xtr if args.xtr else config["xtr"]

        fname = os.path.join(
            emb_dir,
            f"{lang}_{set_name}_split_{args.split_ix}{sfx}T{model_iters}_e{xtr_iters}.h5",
        )

        if not os.path.exists(fname):
            print("- Embedding file not found:", fname, file=sys.stderr)
            fname = None

    return fname


def get_labels_fname(lang, set_name, labels_dir, split_ix):
    """Get labels file name for current lang, set and split"""

    fname = os.path.join(
        os.path.realpath(labels_dir), f"{lang}/{lang}_{set_name}_split_{split_ix}.labels"
    )
    if not os.path.exists(fname):
        print("- ERROR: Label file not found:", fname, file=sys.stderr)
        sys.exit()

    return fname


def load_feats(in_feat_file, dim="half"):
    """Load features"""

    if not in_feat_file:
        print(
            "indic_news_utils (load_feats): in_feat_file should be a file path",
            file=sys.stderr,
        )
        sys.exit()

    if os.path.exists(in_feat_file):
        ext = os.path.basename(in_feat_file).split(".")[-1]
        if ext == "npy":
            feats = np.load(in_feat_file)
        elif ext == "mtx":
            feats = sio.mmread(in_feat_file).tocsr()
        elif ext == "h5":
            feats = get_ivectors_from_h5(in_feat_file, dim_load=dim)
        else:
            print(
                "Input feats file ext not understood.",
                "Should be either .npy or .mtx or .h5",
            )
            sys.exit(1)
    else:
        print(in_feat_file, "NOT FOUND.")
        sys.exit(1)

    return feats


def load_labels(input_labels_file, label2int):
    """Load labels from the label file, and use label2int mapping to convert the
    label names to integers if not done already. Also make the labels contiguous.

        Args:
        -----
            input_labels_file (str): File path to label file, where labels are class names (string)
            label2int (dict or str): Label2int mapping dict or a json file containing the mapping

        Returns:
        --------
            np.ndarray: Labels in integer format.
    """

    if isinstance(label2int, str):
        if os.path.exits(label2int):
            with open(label2int, "r", encoding="utf-8") as fpr:
                label2int = json.load(fpr)
        else:
            print("FILE NOT FOUND:", label2int, file=sys.stderr)
            sys.exit()

    labels = []
    with open(input_labels_file, "r") as fpr:
        for line in fpr:
            line = line.strip()
            try:
                labels.append(int(line))
            except ValueError:
                labels.append(label2int[line])

    labels = np.asarray(labels).astype(int)

    uniq_labels = np.unique(labels)
    if np.where(np.diff(np.sort(uniq_labels)) != 1)[0].any():
        # print("- Making labels contiguous:", uniq_labels.tolist(), end=" -> ")

        l2i_cont = {}
        for l in np.sort(uniq_labels):
            l2i_cont[l] = len(l2i_cont)

        labels_cont = []
        for l in labels:
            labels_cont.append(l2i_cont[l])
        # print(np.unique(labels_cont))

        labels = np.asarray(labels_cont, dtype=int)

    return labels


def check_feats_and_labels(feats, labels, labels_to_use: list, verbose: bool = False):
    """Check features and labels compatibility"""

    if labels.shape[0] != feats.shape[0]:
        print(
            "- Error: number of docs with labels (%d) should",
            labels.shape[0],
            "match number of docs (%d) in feats.",
            feats.shape[0],
        )
        sys.exit()

    uniq_labels = np.unique(labels)
    if np.where(np.diff(np.sort(uniq_labels)) != 1)[0].any():
        print(
            "- Error: Labels are not given consecutive numbers:", uniq_labels.tolist(), file=sys.stderr
        )
        sys.exit()


    if uniq_labels.size < len(labels_to_use):
        print("- Error: Found {:d} unique labels < labels_to_use ({:d})".format(uniq_labels.size, len(labels_to_use)), file=sys.stderr)
        sys.exit()

    li_to_use = np.asarray([L2I[l] for l in labels_to_use])
    li_ixs = np.concatenate([np.where(labels == l)[0] for l in li_to_use])
    labels_subset = labels[li_ixs]
    feats_subset = feats[li_ixs]
    assert (
        feats_subset.shape[0] == labels_subset.shape[0]
    ), "feats and labels subset shapes are not equal"

    feats = feats_subset
    labels = labels_subset
    if verbose:
        print(
            " - selecting feats and labels corresponding to label names:",
            labels_to_use,
    )
        print(" - created feats subset:", feats.shape)
        print(
            " - created labels subset [{:d}]".format(labels.shape[0]),
        np.unique(labels, return_counts=True),
        )

    return feats, labels


def main():
    """main method"""

    args = parse_arguments()


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
