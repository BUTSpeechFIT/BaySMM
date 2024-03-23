#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multilingual Bayesian subspace multinomial model.
1. Training the model
2. Extend the model to newer languages
3. Extrtact/infer posterior distribution of document embeddings for
   any of the seen languages
"""

import os
import sys
import json
import logging
import argparse
from time import time
import copy

from scipy import sparse
import numpy as np
import torch

from baysmm import BaySMM
from data_utils import SMMDataset, MBaySMMDataset
import train_utils
import extract_utils
import utils_multi
from pylibs.misc.io import get_ivectors_from_h5

RESERVE = torch.ones(size=(10, 10))


def load_par2emb(par2emb_file):
    """Load par to embedding from file"""

    par2emb = {}
    with open(par2emb_file, "rb") as fpr:
        par2emb = torch.load(fpr)
    return par2emb


@torch.no_grad()
def compute_ppl(args):
    """Compute PPL"""

    with open(args.cfg_file, "r") as fpr:
        config = json.load(fpr)

    if not args.ppl_dir:
        args.ppl_dir = os.path.join(config["exp_dir"], "ppl/")

    os.makedirs(args.ppl_dir, exist_ok=True)

    lid = args.lid
    device = torch.device("cuda"if config["cuda"] else "cpu")
    print("- Device      :", device)

    if lid not in config["lang2vocab"]:
        print("-", lid, "not in config.")
        sys.exit()

    vocab_size = config["lang2vocab"][args.lid]
    config["vocab_size"] = vocab_size

    sbase = os.path.basename(args.stats_file).split(".")[0]

    data_npz = sparse.load_npz(args.stats_file)
    if data_npz.shape[0] == vocab_size:
        data_npz = data_npz.T
    assert (
        data_npz.shape[1] == vocab_size
    ), "Dim 1 in stats npz ({:d}) does not match vocab size ({:d})".format(
        data_npz.shape[1], vocab_size
    )

    config["n_docs"] = data_npz.shape[0]
    print("- Stats       :", data_npz.shape)

    embs = get_ivectors_from_h5(args.emb_file, dim_load="full")
    print("- Embeddings  :", embs.shape)
    model_iter = os.path.basename(args.emb_file).rsplit(".", 1)[0].split("_")[-2][1:]

    ppl_file = os.path.join(args.ppl_dir, f"{sbase}_ppl_r_{args.R}_{model_iter}.txt")
    pd_ppl_file = os.path.join(
        args.ppl_dir, f"{sbase}_ppl_pd_r_{args.R}_{model_iter}.txt"
    )
    if os.path.exists(ppl_file) and not args.ovr:
        print(ppl_file, "already exists. Skipping.")
        with open(ppl_file, "r") as fpr:
            print(fpr.read())
            sys.exit()

    cpu = torch.device("cpu")
    mbay_model = torch.load(args.model_file, map_location=cpu)

    if lid not in mbay_model.T:
        print("-", lid, "not in mbay_model")
        sys.exit()

    bay_model = BaySMM(mbay_model.m[lid].data, config, config["cuda"])
    bay_model.T.data = mbay_model.T[lid].data
    bay_model.m.data = mbay_model.m[lid].data

    bay_model.T.requires_grad_(False)
    bay_model.m.requires_grad_(False)

    bay_model.config["R"] = args.R
    bay_model.Q = torch.nn.Parameter(torch.from_numpy(embs), requires_grad=False)
    bay_model.R = torch.nn.Parameter(torch.Tensor([args.R]).int(), requires_grad=False)
    bay_model.eps = torch.nn.Parameter(
        torch.randn(args.R, data_npz.shape[0], config["hyper"]["K"]),
        requires_grad=False,
    )

    bay_model.to(bay_model.device)

    bay_model.sample()

    dset = SMMDataset(data_npz, None, vocab_size, "unsup")
    dset.to_device(device)

    ppl_per_doc = torch.Tensor([]).to(device=device, dtype=torch.float)

    total_loss = torch.Tensor([0]).to(device=device, dtype=torch.float)

    for data_dict in dset.yield_batches(data_npz.shape[0], shuffle=False):
        loss, _ = bay_model.compute_loss(data_dict, "Q")

        total_loss += loss

        ppl_per_doc = torch.cat((ppl_per_doc, loss / sum(data_dict["counts"])))

    print("- Loss         :", total_loss.cpu().numpy()[0])

    avg_doc_ppl = np.exp(ppl_per_doc.sum().item() / data_npz.shape[0])
    print("- Avg. Doc. PPL: {:.0f}".format(avg_doc_ppl))

    corpus_ppl = np.exp(total_loss.item() / data_npz.sum())
    print("- Corpus    PPL: {:.0f}".format(corpus_ppl))

    np.savetxt(
        ppl_file,
        np.asarray([avg_doc_ppl, corpus_ppl]).reshape(1, -1),
        fmt="%.1f",
        header="avg.doc.ppl,corpus_ppl",
        delimiter=",",
    )
    np.savetxt(pd_ppl_file, np.exp(ppl_per_doc), fmt="%.1f")

    print("- PPL saved in:", ppl_file)
    print("- Per doc. PPL saved in:", pd_ppl_file)


@torch.no_grad()
def compute_elbo(dset, model, par2emb, config, R=32) -> torch.Tensor:
    """Compute Evidence Lower Bound with 32 MonteCarlo samples.

    Args:
        dset (MBaySMMDataset): MBaySMMDataset object
        model (MBaySMM): MBaySMM Model
        par2emb (dict): Parallel dataset ID to GauEmb mapping
        config (dict): Experiment configuration
        R (int): Number of MonteCarlo samples for approximating E[log-sum-exp]

    Returns:
        torch.Tensor: [expected_log_likelihood, KL divergence]
    """

    # Set R=32 in order to get a good estimate of ELBO
    for p_num in par2emb:
        par2emb[p_num].set_R(R)

    elbo = torch.zeros(2).to(device=model.device, dtype=torch.float)
    for p_num, doc_ixs, batch_dict in dset.yield_batches(
        config["elbo_bsize"], False, model.device
    ):
        elbo[0] += model.compute_exp_llh(doc_ixs, batch_dict, par2emb[p_num])
        elbo[1] += par2emb[p_num].compute_kld(doc_ixs)

    # Set R to the original value
    for p_num in par2emb:
        par2emb[p_num].set_R(config["hyper"]["R"])

    return elbo


def extend_model(args):
    """Extend the model to newer languages.

    Will create new model with the required language-specific parameters copied
    from an existing model (or latest_trn_model). Once the training has finished,
    the new language-specific parameters will be added to old orignal model and saved.
    """

    dset = MBaySMMDataset(args.data_json, args.new_lang_vocab_json)

    # -- configuration --

    with open(args.cfg_file, "r") as fpr:
        config = json.load(fpr)

    extend_config = copy.deepcopy(config)
    extend_config["latest_trn_model"] = ""
    extend_config["latest_trn_emb"] = ""
    extend_config["trn_done"] = 0
    extend_config["lang2vocab"] = {}
    extend_config["train_data_json"] = args.data_json

    extend_config["model_dir"] = os.path.join(config["model_dir"], args.ext_id)
    extend_config["emb_dir"] = os.path.join(
        config["exp_dir"], f"embeddings_{args.ext_id}"
    )

    extend_config["cfg_file"] = os.path.join(
        extend_config["exp_dir"], f"config_{args.ext_id}.json"
    )
    os.makedirs(extend_config["model_dir"], exist_ok=True)
    os.makedirs(extend_config["emb_dir"], exist_ok=True)

    extend_config["bsize"] = args.bsize
    extend_config["bsize_dec"] = args.bsize_dec
    extend_config["min_bsize"] = args.min_bsize

    for lid, vocab_size in dset.lang2vocab.items():
        if lid in config["lang2vocab"]:
            ex_vocab_size = config["lang2vocab"][lid]
            if ex_vocab_size != vocab_size:
                print(
                    "- Existing vocab size ({:d}) for language {:s} should match the vocab size ({:d}) from newer parallel data".format(
                        ex_vocab_size, lid, vocab_size
                    ),
                    file=sys.stderr,
                )
                sys.exit()
        else:
            config["lang2vocab"][lid] = vocab_size

        extend_config["lang2vocab"][lid] = vocab_size

    os.makedirs(extend_config["tmp_dir"] + "embs/", exist_ok=True)

    # -- end of configuration --

    logger = utils_multi.create_log_file(
        extend_config["log_dir"], "extend_training_" + args.ext_id
    )
    if args.verbose:
        logger.addHandler(logging.StreamHandler())

    logger.info("PyTorch version: %s", str(torch.__version__))
    logger.info("Loaded config file from %s", args.cfg_file)

    # -- end of log file creation --

    cpu = torch.device("cpu")
    if args.model_f == "latest_trn_model":
        args.model_f = config["latest_trn_model"]
    model = torch.load(args.model_f, map_location=cpu)

    # new_model, par2emb = train_utils.create_model(extend_config, dset)

    trainable_languages = set()
    for p_num in dset.info:
        trainable_languages |= set(dset.info[p_num]["train"])
    print("Parameters of these languages will be updated:", trainable_languages)

    lang2ubm_all = train_utils.get_lang2ubm(dset)
    lang2vocab = {}
    lang2ubm = {}
    for lid in trainable_languages:
        lang2vocab[lid] = extend_config["lang2vocab"][lid]
        lang2ubm[lid] = lang2ubm_all[lid]

    model.add_params(lang2vocab, lang2ubm)

    for lid in config["lang2vocab"]:
        if lid in trainable_languages:
            model.T[lid].requires_grad_(True)
            model.m[lid].requires_grad_(True)
        else:
            model.T[lid].requires_grad_(False)
            model.m[lid].requires_grad_(False)

        if lid in extend_config["lang2vocab"]:
            if config["cuda"]:
                model.T[lid].data = model.T[lid].data.cuda()
                model.m[lid].data = model.m[lid].data.cuda()

    if config["cuda"]:
        model.lam_t.data = model.lam_t.cuda()

    par2emb = train_utils.create_par2emb(dset, extend_config)

    logger.info(
        "Trainable parameters %.1f M", utils_multi.get_num_trainable_params(model)
    )

    logger.info(
        "Extended training on {:d} langauges from {:d} parallel set(s), totalling {:d} docs (sents).".format(
            len(trainable_languages), len(dset.par2docs), dset.n_docs
        )
    )

    train_utils.try_train(dset, model, par2emb, extend_config)


def main():
    """main method"""

    global RESERVE

    args = parse_arguments()

    stime = time()

    if not args.safe_gpu:
        args.cuda = False
        # if args.cuda and os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set") != "Not Set":
        #    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        #    print("Using GPU ID:", gpu_id)
        #    RESERVE = RESERVE.to(device=torch.device("cuda"))

    if args.phase == "train":
        train_utils.train(args)

    elif args.phase == "extend":
        extend_model(args)

    elif args.phase == "extract":
        extract_utils.extract(args)

    elif args.phase == "ppl":
        compute_ppl(args)

    else:
        print("- phase:", args.phase, "not understood/implemented.")
        sys.exit()

    print(".. done {:.2f} sec".format(time() - stime))


def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    sub_parsers = parser.add_subparsers(help="phase (train or extract)", dest="phase")
    sub_parsers.required = True

    train_parser = sub_parsers.add_parser(
        "train",
        help="train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    extend_parser = sub_parsers.add_parser(
        "extend",
        help="extend an existing model to newer languages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    xtr_parser = sub_parsers.add_parser(
        "extract",
        help="extract embeddings for the given stats ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ppl_parser = sub_parsers.add_parser(
        "ppl",
        help="compute PPL for the given stats using the given embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- sub parser for training

    train_parser.add_argument(
        "train_data_json", help="path to json file containing parallel npz files"
    )
    train_parser.add_argument(
        "lang_vocab_json",
        help="path to json file containing lang_id to vocab file mappings",
    )

    train_parser.add_argument("out_dir", help="path to output directory")

    train_parser.add_argument(
        "-var_p",
        type=float,
        default=1e01,
        help="initial precision of var. dist (default: %(default)s)",
    )
    train_parser.add_argument(
        "-lw",
        default=1e01,
        type=float,
        help="prior precision of embeddings",
    )
    train_parser.add_argument(
        "-R",
        default=8,
        type=int,
        help="no. of Monte Carlo samples for approximating ELBO via the re-parametrization trick.",
    )

    train_parser.add_argument(
        "-K",
        default=128,
        type=int,
        help="embedding (subspace) dimension.",
    )
    train_parser.add_argument(
        "-rt",
        default="l2",
        choices=["l2"],
        help="l2 regularization for bases T",
    )
    train_parser.add_argument(
        "-lt",
        default=5e-3,
        type=float,
        help="l2 reg. constant for bases T (same for all the languages)",
    )
    train_parser.add_argument(
        "-optim",
        default="adam",
        choices=["adam"],
        help="optimizer",
    )
    train_parser.add_argument("-eta", type=float, default=0.005, help="learning rate")
    train_parser.add_argument(
        "-trn",
        type=int,
        default=100,
        help="number of training iterations.",
    )
    train_parser.add_argument("-bsize", default=4096, type=int, help="batch size")
    train_parser.add_argument("-min_bsize", default=64, type=int, help="min batch size")
    train_parser.add_argument(
        "-bsize_dec", type=int, default=64, help="batch size decrement step"
    )

    train_parser.add_argument(
        "-save",
        default=50,
        type=int,
        help="save every nth intermediate model.",
    )

    train_parser.add_argument(
        "--save_as_npz",
        action="store_true",
        help="saves model parameters in compressed npz format",
    )

    train_parser.add_argument(
        "-log",
        choices=["info", "debug", "warning"],
        default="INFO",
        help="logging level",
    )

    train_parser.add_argument(
        "-mkl",
        default=1,
        type=int,
        help="number of MKL threads.",
    )

    train_parser.add_argument("-seed", type=int, default=0, help="random seed")

    train_parser.add_argument(
        "--weight",
        action="store_true",
        help="weigh the objective in relative to the data size of each language.",
    )

    train_parser.add_argument(
        "--ovr",
        action="store_true",
        help="over-write the exp dir.",
    )

    train_parser.add_argument("--nocuda", action="store_true", help="Do not use GPU.")
    train_parser.add_argument("--verbose", action="store_true", help="verbose")

    # -- sub parser for extending an existing model to newer languages

    extend_parser.add_argument(
        "data_json", help="path to parallel data json file with newer languages"
    )
    extend_parser.add_argument(
        "new_lang_vocab_json",
        help="path to json file containing lang_id to vocab file mappings",
    )

    extend_parser.add_argument("cfg_file", help="path to config json file")

    extend_parser.add_argument(
        "ext_id", type=str, help="unique string to represent this model extension"
    )

    extend_parser.add_argument(
        "-model_f",
        default="latest_trn_model",
        type=str,
        help="path to trained model file.",
    )

    extend_parser.add_argument(
        "-log",
        choices=["info", "debug", "warning"],
        default="INFO",
        help="logging level",
    )

    extend_parser.add_argument(
        "-mkl",
        default=1,
        type=int,
        help="number of MKL threads.",
    )

    extend_parser.add_argument("-bsize", default=4096, type=int, help="batch size")
    extend_parser.add_argument(
        "-min_bsize", default=64, type=int, help="min batch size"
    )
    extend_parser.add_argument(
        "-bsize_dec", type=int, default=64, help="batch size decrement step"
    )

    extend_parser.add_argument("-seed", type=int, default=0, help="random seed")

    extend_parser.add_argument("--nocuda", action="store_true", help="do not use cuda")
    extend_parser.add_argument("--verbose", action="store_true", help="verbose")

    # -- sub parser for extracting embeddings

    xtr_parser.add_argument(
        "extract_data_json",
        help="path to json file in a specific format.",
    )

    xtr_parser.add_argument("cfg_file", help="path to config json file")

    xtr_parser.add_argument(
        "-model_f",
        default="latest_trn_model",
        type=str,
        help="path to trained model file.",
    )
    xtr_parser.add_argument(
        "-R", type=int, default=8, help="number of MonteCarlo samples"
    )
    xtr_parser.add_argument(
        "-xtr",
        type=int,
        default=50,
        help="number of extraction iterations.",
    )
    xtr_parser.add_argument("-eta", type=float, default=5e-3, help="Learning rate")
    xtr_parser.add_argument(
        "-nth",
        type=int,
        default=100,
        help="save every nth extracted embedding.",
    )
    xtr_parser.add_argument("-bsize", default=8192, type=int, help="batche size.")
    xtr_parser.add_argument("-mkl", default=1, type=int, help="number of MKL threads.")
    xtr_parser.add_argument("-seed", type=int, default=0, help="random seed")

    xtr_parser.add_argument(
        "-log",
        choices=["info", "debug", "warning"],
        default="INFO",
        help="logging level",
    )
    xtr_parser.add_argument(
        "-out_dir",
        default="",
        type=str,
        help="output dir to save embeddings. Default is taken from config[emb_dir]",
    )

    xtr_parser.add_argument("--nocuda", action="store_true", help="do not use cuda")
    xtr_parser.add_argument("--verbose", action="store_true", help="verbose.")

    # PPL parser
    ppl_parser.add_argument("stats_file", help="path to stats npz file")
    ppl_parser.add_argument("lid", type=str, help="language ID - ISO two letter code")
    ppl_parser.add_argument("cfg_file", help="path to config file")
    ppl_parser.add_argument("emb_file", help="path to embeddings file")
    ppl_parser.add_argument("model_file", help="path to model file")

    ppl_parser.add_argument(
        "-ppl-dir", type=str, default="", help="dir to save ppl values"
    )
    ppl_parser.add_argument("-R", type=int, default=32, help="no. of samples")

    ppl_parser.add_argument(
        "-mkl",
        default=1,
        type=int,
        help="number of MKL threads",
    )
    ppl_parser.add_argument("-seed", default=0, type=int, help="random seed")
    ppl_parser.add_argument(
        "--ovr", action="store_true", help="Overwrite the PPL scores in the file."
    )
    ppl_parser.add_argument("--nocuda", action="store_true", help="do not use cuda")
    ppl_parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    args.prog = parser.prog

    args.cuda = torch.cuda.is_available() and not args.nocuda

    torch.set_num_threads(args.mkl)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == "__main__":
    version = torch.__version__.split(".")
    if int(version[0]) < 1 and int(version[1]) < 7:
        print(
            "Requires PyTorch version >= 1.8",
            "Current version is:",
            torch.__version__,
            file=sys.stderr,
        )
        sys.exit()

    main()
