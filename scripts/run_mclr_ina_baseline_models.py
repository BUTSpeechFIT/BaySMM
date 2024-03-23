#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train MCLR / MCLRU and predict labels or posterior probabilities
"""

import os
import sys
import glob
import json
import shutil
import platform
from time import time
import logging
import tempfile
import argparse
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score
from pylibs.clf.mclr_models import MCLR, MCLRU
from safe_gpu import safe_gpu
import mclr_utils
from indic_news_utils import (
    infer_sfx,
    check_for_existing_results_file,
    load_feats,
    load_labels,
    check_feats_and_labels,
    get_feat_fname,
    get_labels_fname,
    LANGS,
    L2I,
)


TMP_DIR = tempfile.mkdtemp() + "/"


def create_file_suffix(args):
    sfx = f"{args.clf}_{args.in_lang}_{args.split_ix}"
    return sfx


def load_feats_and_labels(emb_dir, in_lang, set_name, args):
    """Load features and labels"""

    logger = logging.getLogger()

    feats_f = os.path.join(emb_dir, f"{in_lang}_{set_name}_split_{args.split_ix}.{args.ext}")
    # feats_f = get_feat_fname(emb_dir, in_lang, config, set_name, args)
    labels_f = get_labels_fname(in_lang, set_name, args.labels_dir, args.split_ix)

    # logger.info(feats_f)
    # logger.info(labels_f)

    if not feats_f:
        print(
            "- No Embedding file found for {:s} ({:s}) in {:s}".format(
                in_lang, set_name, emb_dir
            ),
            file=sys.stderr,
        )
        # sys.exit()

    if L2I:
        labels = load_labels(labels_f, L2I)
    else:
        labels = np.loadtxt(labels_f)

    if min(labels) == 1:
        labels -= 1

    if feats_f:
        if args.ext in ("raw", "enc"):
            feats = np.fromfile(feats_f, dtype=np.float32, count=-1)
            feats = feats.reshape(feats.shape[0] // args.dim, args.dim)
        else:
            feats = np.load(feats_f)

        feats, labels = check_feats_and_labels(feats, labels, args.labels_to_use, args.verbose)
        return torch.from_numpy(feats).float(), torch.from_numpy(labels).long()

    else:
        return None, None


def create_config(args, dim, n_classes):
    """Create config file"""

    config = {}
    for k, v in vars(args).items():
        config[k] = v

    config["dim"] = dim
    config["n_classes"] = n_classes
    config["dev_metric"] = args.dev_metric

    return config


def create_model_and_optim(config):
    """Create model and optimizer"""

    if config["clf"] == "mclr":
        model = MCLR(
            dim=config["dim"],
            n_classes=config["n_classes"],
            lam_w=config["lw"],
            cuda=config["cuda"]
        )

    elif config["clf"] == "mclru":
        model = MCLRU(
            dim=config["dim"],
            n_classes=config["n_classes"],
            lam_w=config["lw"],
            R=config["R"],
            cuda=config["cuda"],
            use_uncert=True
        )

    elif config["clf"] == "mclru-0":
        model = MCLRU(
            dim=config["dim"],
            n_classes=config["n_classes"],
            lam_w=config["lw"],
            R=config["R"],
            cuda=config["cuda"],
            use_uncert=False,
        )

    if config["cuda"]:
        model.cuda()
        model.device = torch.device("cuda")

    optim = torch.optim.Adam([model.W, model.b], lr=config["lrate"])

    return model, optim


def classify_dev_test_data(
    model, emb_dir, langs, args, scores_dict, proba_dict
):
    """Classify dev and test data"""

    logger = logging.getLogger()

    model.compute_grads(False)

    logger.info("Classifying dev and test data:")

    for i, set_name in enumerate(["dev", "test"]):
        proba_sub_dict = {}
        acc_sub_dict = {}
        xen_sub_dict = {}

        for j, lang in enumerate(langs):

            x_test, y_test = load_feats_and_labels(
                emb_dir, lang, set_name, args
            )

            if x_test is None:
                continue

            x_test, y_test = check_feats_and_labels(x_test, y_test, args.labels_to_use, args.verbose)

            if args.verbose:
                print(set_name, lang, ':', x_test.shape, y_test.shape)

            labs = y_test.cpu().numpy()

            pred_proba = model.predict_proba(x_test)
            pred = torch.argmax(pred_proba, dim=1).cpu().numpy()

            proba_sub_dict[lang] = pred_proba.cpu().numpy().tolist()

            acc_sub_dict[lang] = accuracy_score(labs, pred)
            xen_sub_dict[lang] = log_loss(labs, pred_proba.cpu().numpy())

        proba_dict[set_name] = proba_sub_dict
        scores_dict["acc"][set_name] = acc_sub_dict
        scores_dict["xen"][set_name] = xen_sub_dict

    return scores_dict, proba_dict


def train_classifier(x_train, y_train, x_dev, y_dev, args):
    """Train classifier and obtain best hyper-param by tuning on dev data"""

    logger = logging.getLogger()

    if args.verbose:
        print("- Temp dir:", TMP_DIR)

    # hyper-params
    lam_ws = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e01, 5e-4, 5e-3, 5e-2, 5e-1, 5, 5e01]

    n_classes = np.unique(y_train).shape[0]
    logger.info("Num of classes %d", n_classes)
    dim = x_train.shape[1] if args.clf == "mclr" else x_train.shape[1] // 2

    mclr_config = create_config(args, dim, n_classes)

    vix = 2 if args.dev_metric == "acc" else 3
    init_score = 0.0 if args.dev_metric == "acc" else 999.0
    best_dev = [init_score, 0, 0, None]  # [acc, xen, lw, model_file]

    print("- Training and dev sets are from:", args.in_lang)
    print("- Finding best hyper-parameter by tracking", args.dev_metric, "on dev set.")
    for lam_w in lam_ws:

        sfx = create_file_suffix(args)
        sfx += f"_{lam_w}"

        mclr_config["lw"] = lam_w

        logger.info("R: %d, lam_w: %2.0e", args.R, lam_w)

        model, optim = create_model_and_optim(mclr_config)

        best_model_file, scores = mclr_utils.train_and_validate(
            model,
            optim,
            x_train,
            y_train,
            x_dev,
            y_dev,
            TMP_DIR + f"temp_clf_{sfx}",
            mclr_config["trn"],
            mclr_config["val_iters"],
            dev_metric=args.dev_metric,
        )

        best_ix = np.argmax(scores[:, 2])

        print_str = (
            "R: {:3d} lw: {:2.0e} | Train acc: {:.4f} xen: {:.4f} | "
            "Dev acc: {:.4f} xen: {:.4f}".format(args.R, lam_w, *scores[best_ix, :])
        )
        if args.verbose or args.v >= 2:
            print("  - ", print_str)

        if args.dev_metric == "acc":
            if scores[best_ix, vix] > best_dev[0]:
                best_dev = [scores[best_ix, vix], lam_w, best_model_file]
        else:
            if scores[best_ix, vix] < best_dev[0]:
                best_dev = [scores[best_ix, vix], lam_w, best_model_file]

    if args.verbose or args.v >= 1:
        print(
            "- Best Dev {:3s}: {:.4f} with lw: {:2.0e} | Model: {:s}".format(
                args.dev_metric, *best_dev
            )
        )

    best_model = torch.load(best_dev[-1])

    return best_model


def main():
    """main method"""

    args = parse_arguments()

    langs = args.langs

    scores_dict = {"acc": {}, "xen": {}}
    proba_dict = {}

    emb_dir = os.path.realpath(args.emb_dir)

    stime = time()

    nl = len(args.labels_to_use)

    if not args.out_dir:
        args.out_dir = os.path.join(os.path.realpath(args.emb_dir), f"results_ina_mclr_{nl}classes/")

    os.makedirs(args.out_dir, exist_ok=True)

    sfx = create_file_suffix(args)

    log_dir = os.path.join(args.out_dir, "logs/")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=os.path.join(log_dir, f"run_{sfx}.log"),
        filemode="w",
        level=logging.INFO,
    )
    print("- Log file:", os.path.join(log_dir, f"run_{sfx}.log"))
    logger = logging.getLogger()
    if args.safe_gpu:
        gpu_owner = safe_gpu.GPUOwner(logger=logger)
        args.cuda = True
    else:
        args.cuda = False

    logger.info("Node : %s", platform.node())
    if args.verbose:
        logger.addHandler(logging.StreamHandler())
    logger.info("PyTorch version: %s", str(torch.__version__))

    res_file, proba_file, scores_dict, proba_dict = check_for_existing_results_file(
        args, f"_R{args.R}"
    )

    x_train, y_train = load_feats_and_labels(
        emb_dir, args.in_lang, f"train", args
    )

    logger.info("Training feats (%d, %d) Labels: %d", *x_train.shape, y_train.shape[0])

    x_dev, y_dev = load_feats_and_labels(emb_dir, args.in_lang, "dev", args)

    logger.info("Dev feats (%d, %d) Labels: %d", *x_dev.shape, y_dev.shape[0])

    ## Train classifier
    best_model = train_classifier(x_train, y_train, x_dev, y_dev, args)

    ## Classify dev and test data of all the languages
    scores_dict, proba_dict = classify_dev_test_data(
        best_model, args.emb_dir, langs, args, scores_dict, proba_dict
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

    clf_name = os.path.join(args.out_dir, f"{sfx}.pt")
    torch.save(best_model, clf_name)
    print("- Best classifier model saved as", clf_name)

    if args.verbose or args.v >= 1:
        import pprint

        pprint.pprint(scores_dict["acc"])

    if args.v >= 2:
        print("- Removing temp dir:", TMP_DIR)
        shutil.rmtree(TMP_DIR)
        print("- Done {:.4f} sec".format(time() - stime))

    logger.info("Done")


def parse_arguments():
    """parse command line args"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("emb_dir", help="path to dir with embeddings")
    parser.add_argument("ext", type=str, choices=["raw", "enc", "npy"], help="file ext for embeddings")
    parser.add_argument("dim", type=int, help="embedding dimension")
    parser.add_argument("labels_dir", help="path to dir with random_data_splits/ that also contains labels in sub dirs")
    parser.add_argument("split_ix", type=int, choices=range(1, 6), help="split index")
    parser.add_argument(
        "in_lang",
        type=str,
        help="source language for traning classifier",
        choices=LANGS,
    )

    parser.add_argument(
        "-labels_to_use",
        type=str,
        nargs="+",
        default=["entertainment", "sports"],
        help="list of labels to use from IndicNLP-news-classif dataset. 2 labels for all 9 languages, or 3 labels (+business) for 5 languages (gu ml or pa te). Other categories have much less overlap (2 langs).",
    )
    parser.add_argument("-langs", type=str, nargs='+', default=LANGS, help='langs to use')
    parser.add_argument(
        "-dev_metric",
        type=str,
        default="acc",
        choices=["acc", "xen"],
        help="which eval metric should be used for choosing best hyper-parameters",
    )
    parser.add_argument("-model_iters", default=None, help="model iters")
    parser.add_argument("-xtr", default=None, help="xtr iters")

    parser.add_argument(
        "-out_dir", default="", help="path to output dir to save results and classifier"
    )
    # parser.add_argument(
    #    "-ntrain",
    #    default=1000,
    #    type=int,
    #    choices=[1000, 5000, 10000],
    #    help="number of training examples for the classifier",
    # )

    parser.add_argument("-trn", default=100, type=int, help="training iters")
    parser.add_argument(
        "-lrate", default=5e-2, type=float, help="learning rate for adam"
    )
    # parser.add_argument(
    #     "-R",
    #     default=32,
    #     type=int,
    #     help="number of samples for Monte Carlo approx for MCLRU",
    # )
    parser.add_argument(
        "-val_iters", type=int, default=10, help="compute val acc for every `val_iters`"
    )

    parser.add_argument(
        "-threads",
        default=1,
        type=int,
        help="num of CPU threads to use if GPU is not available",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append results for new languages to the existing file",
    )

    parser.add_argument("-seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "-v",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="verbosity level, low to high",
    )
    parser.add_argument("--safe-gpu", action="store_true", help="use safe-gpu package to acquire a free GPU")
    parser.add_argument("--ovr", action="store_true", help="over write")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display everything, same as setting -v to 5.",
    )

    args = parser.parse_args()

    args.clf = "mclr"
    args.R = 0

    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.v == 5:
        args.verbose = True

    return args


if __name__ == "__main__":
    main()
