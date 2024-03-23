#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Common utils of RCV data
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import h5py
import scipy.io as sio


LANGS = [
    "english",
    "german",
    "french",
    "italian",
    "spanish",
    "russian",
    "japanese",
    "chinese",
]


L2I = {
    "CCAT": 0,
    "ECAT": 1,
    "GCAT": 2,
    "MCAT": 3
}


def check_for_existing_results_file(args, sfx=""):
    """Check for existing results file"""

    scores_dict = {'acc': {}, 'xen': {}}
    proba_dict = {}

    res_base = f"{args.in_lang}_{args.ntrain}_{args.split_ix}_{args.clf}"
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
            with open(res_file, 'r') as fpr:
                scores_dict = json.load(fpr)
            proba_file = res_file.replace(".json", "_proba.json")
            print("- Loading proba   file from:", proba_file)
            with open(proba_file, 'r') as fpr:
                proba_dict = json.load(fpr)

    return res_file, proba_file, scores_dict, proba_dict


def infer_sfx(emb_dir, lang):
    """Infer common suffix from files"""

    # print('emb dir:', emb_dir)
    fnames = glob.glob(emb_dir + "/" + f"{lang}*.h5")
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

    fname = None
    if sfx:
        model_iters = args.model_iters if args.model_iters else config["trn_done"]
        xtr_iters = args.xtr if args.xtr else config["xtr"]
        fname = os.path.join(
            emb_dir,
            f"{lang}_{set_name}_split_{args.split_ix}{sfx}T{model_iters}_e{xtr_iters}.h5"
        )

        if not os.path.exists(fname):
            print("Embedding file not found:", fname, file=sys.stderr)
            fname = None

    return fname


def get_labels_fname(lang, set_name, labels_dir, split_ix):
    """Get labels file name for current lang, set and split"""

    fname = os.path.join(labels_dir, f"{lang}/{lang}_{set_name}_split_{split_ix}.labels")
    if not os.path.exists(fname):
        print("Label file not found:", fname, file=sys.stderr)
        sys.exit()

    return fname


def load_feats(in_feat_file, dim='half'):
    """ Load features """

    if not in_feat_file:
        print("rcv_utils: in_feat_file should be a file path", file=sys.stderr)
        sys.exit()

    if os.path.exists(in_feat_file):
        ext = os.path.basename(in_feat_file).split(".")[-1]
        if ext == "npy":
            feats = np.load(in_feat_file)
        elif ext == "mtx":
            feats = sio.mmread(in_feat_file).tocsr()
        elif ext == "h5":
            feats = get_ivectors_from_h5(in_feat_file, dim=dim)
        else:
            print("Input feats file ext not understood.",
                  "Should be either .npy or .mtx or .h5")
            sys.exit(1)
    else:
        print(in_feat_file, "NOT FOUND.")
        sys.exit(1)

    return feats


def get_ivectors_from_h5(ivecs_h5_file, iter_num=-1, dim='half', config=None):
    """ Load ivectors from h5 file and return them in numpy array
        for a given iter num. If its -1, then the final iteration
        i-vectors are returned. """

    max_iters = int(os.path.splitext(os.path.basename(
        ivecs_h5_file))[0].split("_")[-1][1:])
    # print('get_ivectors_from_h5: max_iters:', max_iters)
    ivecs_h5f = h5py.File(ivecs_h5_file, 'r')
    ivecs_h5 = ivecs_h5f.get('ivecs')

    if not config:
        cfg_file = os.path.dirname(ivecs_h5_file) + "/../config.json"
        with open(cfg_file, 'r') as fpr:
            config = json.load(fpr)

    cur_dim = config['hyper']['K']
    # print("embedding dim:", cur_dim)

    if iter_num == -1:
        ivecs = ivecs_h5.get(str(max_iters))[()]
    else:
        ivecs = ivecs_h5.get(str(iter_num))[()]

    if ivecs.shape[0] == (2 * cur_dim):
        ivecs = ivecs.T

    assert ivecs.shape[1] == (2 * cur_dim)

    if dim == 'half':
        ivecs = ivecs[:, :cur_dim]

    ivecs_h5f.close()
    return ivecs


def load_labels(input_labels_file, label2int):
    """ Load labels from the label file, and use label2int mapping to convert the
label names to integers.

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
            with open(label2int, 'r', encoding='utf-8') as fpr:
                label2int = json.load(fpr)
        else:
            print("FILE NOT FOUND:", label2int, file=sys.stderr)
            sys.exit()

    labels = []
    with open(input_labels_file, 'r') as fpr:
        labels = [label2int[line.strip()] for line in fpr if line.strip()]
    labels = np.asarray(labels).astype(int)

    return labels


def check_feats_and_labels(feats, labels):
    """ Check features and labels compatibility """

    if labels.shape[0] != feats.shape[0]:
        print('Error: number of docs with labels (%d) should', labels.shape[0],
              'match number of docs (%d) in feats.', feats.shape[0])
        sys.exit()


def main():
    """ main method """
    args = parse_arguments()


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
