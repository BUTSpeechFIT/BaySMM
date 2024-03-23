#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utils for creating and training model
"""

import os
import re
import sys
import json
import pickle
import shutil
import traceback
from time import time
import tempfile
import platform
import logging

import numpy as np
from scipy import sparse
import torch
from torch._C import device

from baysmm import MBaySMM, GaussianEmbeddings
from data_utils import MBaySMMDataset
from utils_multi import (
    print_model_params,
    get_num_trainable_params,
    save_config,
    create_log_file,
)


def create_mbay_config(args) -> dict:
    """Create config file for MultiBaySMM"""

    logger = logging.getLogger()

    exp_dir = os.path.realpath(args.out_dir) + "/"
    exp_dir += "multi_r_{:d}_vp_{:.0e}_lw_{:.0e}_{:s}_{:.0e}".format(
        args.R, args.var_p, args.lw, args.rt, args.lt
    )
    exp_dir += "_{:d}_{:s}/".format(args.K, args.optim)

    if args.weight:
        exp_dir += "_weight"

    if args.ovr:
        if os.path.exists(exp_dir):
            print("- Overwriting existing output dir:", exp_dir)
            shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_file = os.path.join(exp_dir, "config.json")
    config = {}

    try:
        config = json.load(open(cfg_file, "r"))
        print("- Config:", cfg_file, "loaded.")
        os.makedirs(config["tmp_dir"], exist_ok=True)

        if config["trn"] < args.trn:
            config["trn"] = args.trn
            logger.info("Updated config.trn to %d", args.trn)

    except FileNotFoundError:

        config["model_dir"] = os.path.join(exp_dir, "models/")
        config["emb_dir"] = os.path.join(exp_dir, "embeddings/")
        config["log_dir"] = os.path.join(exp_dir, "logs/")
        config["res_dir"] = os.path.join(exp_dir, "results/")

        os.makedirs(config["model_dir"], exist_ok=True)
        os.makedirs(config["emb_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)
        os.makedirs(config["res_dir"], exist_ok=True)

        config["cfg_file"] = cfg_file  # this file
        config["exp_dir"] = exp_dir
        config["ubm_file"] = os.path.join(config["model_dir"], "ubm.pkl")
        config["tmp_dir"] = tempfile.mkdtemp() + "/"

        config["prog"] = args.prog
        config["pytorch_ver"] = torch.__version__

        config["hyper"] = {}

        for arg in vars(args):
            if arg in ("R", "lw", "var_p", "K", "rt", "lt"):
                config["hyper"][arg] = getattr(args, arg)
            else:
                config[arg] = getattr(args, arg)

        config["trn_done"] = 0
        config["latest_trn_model"] = ""
        config["latest_trn_emb"] = ""

    return config


def estimate_ubm(stats: sparse.csr_matrix) -> torch.FloatTensor:
    """Given the stats (scipy.sparse), estimate UBM (ML log-unigram probs)

    Args:
        stats (scipy.sparse): Doc by Word sparse matrix of counts

    Returns:
        torch.Tensor of size (n_words x 1)
    """
    # universal background model or log-unigram dist. over vocabulary
    return torch.from_numpy(
        np.log((stats.sum(axis=0) / stats.sum()).reshape(-1, 1))
    ).float()


def get_lang2ubm(dset: MBaySMMDataset) -> dict:
    """Get UBMs in a dictionary"""

    lang2ubm = {}
    for p_num in dset.data:
        for lang, stats in dset.data[p_num].items():
            try:
                lang2ubm[lang] += stats.sum(axis=0).reshape(-1, 1).A.astype(np.float32)
            except KeyError:
                lang2ubm[lang] = stats.sum(axis=0).reshape(-1, 1).A.astype(np.float32)

    for lang, wcounts in lang2ubm.items():
        # to avoid divison by zero, which will not occur if stats are obtained properly
        wcounts += 1e-03
        lang2ubm[lang] = np.log(wcounts / wcounts.sum())
    return lang2ubm


def create_model(config: dict, dset: MBaySMMDataset):
    """Creates and initializes the model.

    Model is created using the configuration, and UBM estimated using the dataset.

    Args:
        config (dict): Dictionary with model configuration
        dset (MBaySMMDataset object): MBaySMMDataset object

    Returns:
        torch.nn.Module (MBaySMM), dict (par_num to torch.nn.Module)
    """

    logger = logging.getLogger(__name__)

    if os.path.exists(config["latest_trn_model"]):
        model = torch.load(config["latest_trn_model"])
        logger.info("Loading latest trained model: %s", config["latest_trn_model"])

        par2emb = torch.load(config["latest_trn_emb"])

    else:
        # Create new model
        lang2ubm = get_lang2ubm(dset)
        with open(config["ubm_file"], "wb") as fpw:
            pickle.dump(lang2ubm, fpw)

        logger.info("Saved UBM to %s", config["ubm_file"])

        logger.info("Creating a new model.")

        model = MBaySMM(
            dset.lang2vocab,
            lang2ubm,
            config["hyper"],
            cuda=config["cuda"],
            seed=config["seed"],
        )

        par2emb = create_par2emb(dset, config)

    if config["cuda"]:
        model.cuda()

    return model, par2emb


def create_par2emb(dset, config):
    """Create parallel dataset ID to GaussianEmbedding mapping.

    Args:
        dset (MBaySMMDataset): MBaySMM Dataset
        config (dict): Configuration dictionary

    Returns:
        dict: Parallel dataset ID to GaussianEmbedding (torch.nn.Module) mapping dictionary
    """
    par2emb = {}
    for p_num, n_docs in dset.par2docs.items():
        par2emb[p_num] = GaussianEmbeddings(
            n_docs,
            config["hyper"]["K"],
            prior_mean=0.0,
            prior_precision=config["hyper"]["lw"],
            cov_type="diag",
            R=config["hyper"]["R"],
            cuda=config["cuda"],
        )

    if config["cuda"]:
        for p_num in par2emb:
            par2emb[p_num].cuda()

    return par2emb


def create_optimizers(config, model, par2emb):
    """Create optimiziers for the model and embeddings"""

    opt_emb = {}
    for p_num, gau_emb in par2emb.items():
        opt_emb[p_num] = torch.optim.Adam(gau_emb.parameters(), lr=config["eta"])
    opt_model = torch.optim.Adam(
        [par for par in model.parameters() if par.requires_grad], lr=config["eta"]
    )

    return opt_emb, opt_model


def clear_gradients(opt_emb, opt_model):
    """Clear gradients"""

    for _, opt in opt_emb.items():
        opt.zero_grad()
    opt_model.zero_grad()


def train_model(dset, model, par2emb, config):
    """Train the model

    Args:
    -----
        dset (MBaySMMDataset): MBaySMM dataset object
        model (MBaySMM): MBaySMM model object
        par2emb (dict): Dict with Gaussian Embeddings
        config (dict): Config dict

    Returns:
    --------
        MBaySMM object: Trained model
    """

    logger = logging.getLogger("train_utils")

    opt_emb, opt_model = create_optimizers(config, model, par2emb)

    total_loss = torch.tensor(0.0).to(dtype=torch.float, device=model.device)
    batch_loss = torch.tensor(0.0).to(dtype=torch.float, device=model.device)
    n_steps = torch.tensor(0.0).to(dtype=torch.float, device=model.device)

    # j = 0
    total_steps = "NA"
    for i in range(config["trn_done"], config["trn"]):

        btime = time()

        total_loss.zero_()
        batch_loss.zero_()
        n_steps.zero_()

        sno = 0
        for p_num, doc_ixs, batch_dict in dset.yield_batches(config["bsize"], True):
            sno += 1
            clear_gradients(opt_emb, opt_model)

            if config["cuda"]:
                for lid in batch_dict:
                    batch_dict[lid]["counts"] = batch_dict[lid]["counts"].to(
                        device=model.device
                    )
                    batch_dict[lid]["ixs"] = batch_dict[lid]["ixs"].to(
                        device=model.device
                    )

            exp_llh = model.compute_exp_llh(doc_ixs, batch_dict, par2emb[p_num])

            batch_loss = -1.0 * exp_llh

            for lid in batch_dict:
                t_pen = model.t_penalty(lid)
                batch_loss += t_pen

            kld = par2emb[p_num].compute_kld(doc_ixs)

            batch_loss += kld

            batch_loss.backward()

            total_loss += batch_loss.detach().item()
            n_steps += 1

            opt_model.step()
            opt_emb[p_num].step()

            if sno % 100 == 0 or sno == total_steps:
                logger.info(
                    "Iter: %4d/%4d  | PNum: %3s | Steps per Iter: %6d/%6s | %s: %.1f  |  %s: %.1f s",
                    i + 1,
                    config["trn"],
                    str(p_num),
                    sno,
                    str(total_steps),
                    "Avg. loss",
                    (total_loss / n_steps).cpu().numpy(),
                    "Time taken",
                    (time() - btime),
                )
                total_loss.zero_()
                n_steps.zero_()
                btime = time()

        total_steps = sno
        config["trn_done"] = i + 1

        if (i + 1) % config["save"] == 0 or (i + 1 == config["trn"]):

            config["latest_trn_model"] = os.path.join(
                config["model_dir"], f"model_T{i+1}.pt"
            )
            torch.save(model, config["latest_trn_model"])

            config["latest_trn_emb"] = os.path.join(
                config["model_dir"], f"par2emb_T{i+1}.pt"
            )
            torch.save(par2emb, config["latest_trn_emb"])

            save_config(config)

            logger.info(
                "%s and %s saved.", config["latest_trn_model"], config["latest_trn_emb"]
            )

    return model


def train(args):
    """Parse the arguments and prepare to train the model"""

    dset = MBaySMMDataset(args.train_data_json, args.lang_vocab_json)

    config = create_mbay_config(args)
    config["lang2vocab"] = dset.lang2vocab

    logger = create_log_file(config["log_dir"], "training")

    if args.verbose:
        logger.addHandler(logging.StreamHandler())

    logger.info("Node: %s", platform.node())
    logger.info("PyTorch version: %s", str(torch.__version__))

    save_config(config)

    model, par2emb = create_model(config, dset)

    logger.info("Trainable parameters %.1f M", get_num_trainable_params(model))

    if args.verbose:
        print_model_params(model)
        print(dset)

    logger.info(
        "Training on {:d} parallel set(s), totalling {:d} docs (sents).".format(
            len(dset.par2docs), dset.n_docs
        )
    )

    try_train(dset, model, par2emb, config)


def try_train(dset, model, par2emb, config):
    """The training scheme in try-except block.
    In case of CUDA OOM RuntimeException, batch size is decreased automatically and
    training will be attempted again.

        Args:
        -----
            dset (MBaySMMDataset): An object of MBaySMMDataset object
            model (MBaySMM): Initialized or pre-trained MBaySMM model
            par2emb (dict): Parallel dataset number to embedding mapping. \
Embeddigs within a parallel dataset are shared by the respective langauges.
            config (dict): Experiment config with paths, hyper-params, settings, etc.
    """

    logger = logging.getLogger("train_utils")

    stime = time()

    while True:

        try:

            model = train_model(dset, model, par2emb, config)

            break

        except RuntimeError as err:

            if re.search(r"out of memory", str(err)):

                torch.cuda.empty_cache()
                config["bsize"] -= config["bsize_dec"]
                print("- CUDA OOM. Decreasing training batch size to", config["bsize"])

                if config["bsize"] < config["min_bsize"]:
                    print(
                        "- Batch size < {:d} (min_bsize). Exiting ..".format(
                            config["min_bsize"]
                        ),
                        file=sys.stderr,
                    )
                    sys.exit(1)

            else:
                print("{0}".format(str(err)))
                traceback.print_tb(err.__traceback__)
                sys.exit(1)

            continue

    save_config(config)

    ttaken = time() - stime
    logger.info("Time taken:{:.2f} sec (={:.2f} min)".format(ttaken, ttaken/60.))
