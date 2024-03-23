#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Train a classifier on embeddings from source language and test on all the languages from
IndicNLP News articles dataset
"""

import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from indic_news_utils import (
    load_feats,
    load_labels,
    get_feat_fname,
    get_labels_fname,
    check_feats_and_labels,
    check_for_existing_results_file,
    infer_sfx,
    LANGS,
    L2I,
)

from pylibs.clf.glc_models import GLC, GLCU


def train_classifier(feats, labels, args):
    """Train a classifier on training data"""

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    clf_pkl = ""
    if args.clf != "glc":
        clf_pkl = os.path.join(
            out_dir,
            f"{args.clf}-{args.trn}_{args.in_lang}_{args.split_ix}.pkl",
        )
    else:
        clf_pkl = os.path.join(out_dir, f"{args.clf}_{args.in_lang}_{args.split_ix}.pkl")

    if os.path.exists(clf_pkl) and not args.ovr:
        print("- Loading classifier from:", clf_pkl)
        with open(clf_pkl, "rb") as fpr:
            clf = pickle.load(fpr)

    else:

        if args.clf == "glc":

            clf = GLC(est_prior=True)
            clf.train(feats, labels)

        else:

            clf = GLCU(trn_iters=args.trn, est_prior=True)
            clf.train(feats, labels)

        if not args.no_save:
            # saving classifiers
            with open(clf_pkl, "wb") as fpw:
                pickle.dump(clf, fpw)
            if args.verbose:
                print("- Classifier saved to:", clf_pkl)

    return clf


def classify_test_data(clf, emb_dir, config, langs, args, scores_dict, proba_dict):
    """Classify dev / test data"""

    if args.verbose:
        print("- Classifying dev / test data:")

    for i, set_name in enumerate(["dev", "test"]):
        proba_sub_dict = {}
        acc_sub_dict = {}
        xen_sub_dict = {}

        if set_name in scores_dict["acc"]:
            acc_sub_dict = scores_dict["acc"][set_name]
            xen_sub_dict = scores_dict["xen"][set_name]
            proba_sub_dict = proba_dict[set_name]

        for j, lang in enumerate(langs):

            if lang in acc_sub_dict:
                print(
                    "- Skipping classification of {:8s} {:4s} set.".format(
                        lang, set_name
                    ),
                    "Existing results (acc, xen): ({:.4f}, {:.4f})".format(
                        acc_sub_dict[lang], xen_sub_dict[lang]
                    ),
                )
                continue

            feats_fname = get_feat_fname(emb_dir, lang, config, set_name, args)
            if feats_fname:
                labels_fname = get_labels_fname(
                    lang, set_name, args.labels_dir, args.split_ix
                )
            else:
                continue

            if args.clf in ("glcu", "glcu-0"):
                feats = load_feats(feats_fname, dim="full")
            else:
                feats = load_feats(feats_fname, dim="half")

            labs = load_labels(labels_fname, L2I)
            if labs.shape[0] == feats.shape[1]:
                feats = feats.T

            feats, labs = check_feats_and_labels(feats, labs, args.labels_to_use, args.verbose)

            if args.verbose:
                print(
                    "-",
                    os.path.basename(feats_fname),
                    ":",
                    feats.shape,
                    os.path.basename(labels_fname),
                    ":",
                    labs.shape,
                )

            pred_proba = clf.predict_proba(feats)
            pred = np.argmax(pred_proba, axis=1)

            proba_sub_dict[lang] = pred_proba.tolist()

            acc_sub_dict[lang] = accuracy_score(labs, pred)
            xen_sub_dict[lang] = log_loss(labs, pred_proba)

        proba_dict[set_name] = proba_sub_dict
        scores_dict["acc"][set_name] = acc_sub_dict
        scores_dict["xen"][set_name] = xen_sub_dict

    return scores_dict, proba_dict


def main():
    """main method"""

    args = parse_arguments()

    langs = args.langs

    # Load config file for the MBaySMM model that generated the embeddings
    config = {}
    with open(args.cfg_file, "r") as fpr:
        config = json.load(fpr)
    if not args.emb_dir:
        args.emb_dir = config['emb_dir']
    emb_dir = os.path.realpath(args.emb_dir)

    nl = len(args.labels_to_use)

    if not args.out_dir:
        args.out_dir = os.path.join(os.path.realpath(args.emb_dir), f"../results_ina_{nl}classes/")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.clf == "glcu-0":
        args.trn = 0

    res_file, proba_file, scores_dict, proba_dict = check_for_existing_results_file(
        args
    )

    train_feats_file = get_feat_fname(emb_dir, args.in_lang, config, "train", args)
    train_labels_file = get_labels_fname(
        args.in_lang, f"train", args.labels_dir, args.split_ix
    )

    if args.verbose:
        print("- Loading training embeddings:", train_feats_file)

    if args.clf in ("glcu", "glcu-0"):
        feats = load_feats(train_feats_file, dim="full")
    else:
        feats = load_feats(train_feats_file, dim="half")

    # Load labels,
    labels = load_labels(train_labels_file, L2I)

    if labels.shape[0] == feats.shape[1]:
        feats = feats.T

    feats, labels = check_feats_and_labels(feats, labels, args.labels_to_use, args.verbose)

    if args.verbose:
        print("- Input feats:", feats.shape, "labels:", labels.shape)

    clf = train_classifier(feats, labels, args)

    scores_dict, proba_dict = classify_test_data(
        clf, emb_dir, config, langs, args, scores_dict, proba_dict
    )

    with open(res_file, "w") as fpw:
        json.dump(scores_dict, fpw, indent=4)

    with open(proba_file, "w") as fpw:
        json.dump(proba_dict, fpw)

    l2i_file = os.path.join(args.out_dir, "label2int.json")
    if not os.path.exists(l2i_file):
        with open(l2i_file, "w") as fpw:
            json.dump(L2I, fpw, indent=4)

    print("- Results saved in", res_file)

    if args.verbose:
        import pprint

        pprint.pprint(scores_dict["acc"])


def parse_arguments():
    """parse arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("clf", choices=["glc", "glcu"], help="choice of classifier")
    parser.add_argument(
        "cfg_file", help="path to config json file corresponding model that is used to extract the embeddings"
    )
    parser.add_argument("labels_dir", help="path where all the true labels are present")
    parser.add_argument("split_ix", type=int, choices=range(1, 6), help="split index")
    parser.add_argument(
        "in_lang",
        type=str,
        help="source language for traning classifier",
        choices=LANGS,
    )
    parser.add_argument("-emb_dir", help="path to dir with embeddings (by default it will be taken from the config file")
    parser.add_argument(
        "-labels_to_use",
        type=str,
        nargs="+",
        default=["entertainment", "sports"],
        help="list of labels to use from IndicNLP-news-classif dataset. 2 labels for 9 languages, or 3 labels (+business) for 5 languages. others have much less overlap (2 langs).",
    )
    parser.add_argument("-langs", type=str, nargs='+', default=LANGS, help='langs to use')
    parser.add_argument("-model_iters", default=None, help="model iters")
    parser.add_argument("-xtr", default=None, help="xtr iters")
    parser.add_argument(
        "-out_dir",
        default="",
        help="path to output dir to save results and classifier.\
The default one will be emb_dir/../results/",
    )
    # parser.add_argument(
    #    "-ntrain",
    #    default=1000,
    #    type=int,
    #    choices=[1000, 2000, 5000, 10000],
    #    help="number of training examples for the classifier",
    # )
    parser.add_argument("-nj", default=1, type=int, help="number of jobs")
    parser.add_argument(
        "-trn",
        default=5,
        type=int,
        help="EM iterations for GLCU. If trn=0, model is GLC and \
                        scoring at test time is done with GLCU",
    )
    parser.add_argument("--no_save", action="store_true", help="do not save classifier")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append results for new languages to the existing file",
    )
    parser.add_argument("--ovr", action="store_true", help="overwrite results file")
    parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
