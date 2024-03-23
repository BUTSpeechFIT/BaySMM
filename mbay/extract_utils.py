#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Commom utils for extracting embeddings
"""

import os
import re
import sys
import json
import math

import platform
import logging

import traceback

from datetime import datetime

from scipy import sparse
import torch

from baysmm import BaySMM
import utils
import utils_multi
import data_utils
from safe_gpu import safe_gpu


def extract_and_save(npz_file, lid, mbay_model, config, mbase):
    """Extract embeddings for the given stats, and save.


    Args:
    ----
        npz_file (str): Path to BoW stats in npz format
        lid (str): Language ID
        mbay_model (MBaySMM): Multi BaySMM model
        config (dict): Model configuration
        mbase (str): Model basename (eg: T200)
    """

    logger = logging.getLogger("extract_and_save")

    if lid not in config["lang2vocab"]:
        print(lid, "not in the model represented by:", config["cfg_file"])
        return True

    vocab_size = config["lang2vocab"][lid]
    data_npz = sparse.load_npz(npz_file).tocsc()
    if data_npz.shape[0] == vocab_size:
        data_npz = data_npz.T

    assert (
        data_npz.shape[1] == config["lang2vocab"][lid]
    ), f"{npz_file} dim 1 != {vocab_size}"

    sbase = os.path.basename(npz_file).rsplit(".", 1)[0]

    out_file = os.path.join(
        config["emb_dir"], f"{sbase}_{mbase}_e" + str(config["xtr"]) + ".h5"
    )

    if not os.path.exists(out_file):

        logger.info(
            "Extracting embeddings for %d %s (%s)", data_npz.shape[0], "docs", npz_file
        )

        config["vocab_size"] = vocab_size
        config["n_docs"] = data_npz.shape[0]

        if config["xtr_bsize"] > config["n_docs"]:
            config["xtr_bsize"] = config["n_docs"]

        bay_model = BaySMM(mbay_model.m[lid].data, config, config["cuda"])
        bay_model.T.data = mbay_model.T[lid].data
        bay_model.m.data = mbay_model.m[lid].data

        bay_model.T.requires_grad_(False)
        bay_model.m.requires_grad_(False)

        bay_model.to(bay_model.device)

        dset = data_utils.SMMDataset(data_npz, None, vocab_size, "unsup")

        dset.to_device(bay_model.device)

        # Reset embedding posterior parameters
        bay_model.init_var_params(data_npz.shape[0])

        # move model to device (CUDA if available)
        bay_model.to_device(bay_model.device)

        # Create optimizer
        if config["optim"] == "adam":
            opt_e = torch.optim.Adam([bay_model.Q], lr=config["xtr_eta"])
        else:
            opt_e = torch.optim.Adagrad([bay_model.Q], lr=config["xtr_eta"])

        n_batches = math.ceil(data_npz.shape[0] / config["xtr_bsize"])
        loss_iters = bay_model.extract_ivector_posteriors(dset, opt_e, sbase, n_batches)

        utils.save_loss(
            loss_iters,
            bay_model.config,
            "xtr_" + sbase,
            "_" + mbase + "_e{:d}".format(config["xtr"]),
        )

        utils.merge_ivecs_v2(config, sbase, mbase, config["xtr"], n_batches)
    else:
        logger.info(
            "Embedding were already extracted: %s",
            out_file,
        )

    return True


def extract(args):
    """Extract embeddings"""

    # -- configuration --

    if not os.path.exists(args.cfg_file):
        print("- Config file not found:", args.cfg_file, out=sys.stderr)
        sys.exit()

    with open(args.cfg_file, "r") as fpr:
        config = json.load(fpr)

    os.makedirs(config["tmp_dir"], exist_ok=True)

    config["xtr_R"] = args.R
    config["xtr_done"] = 0
    config["xtr"] = args.xtr
    config["nth"] = args.nth
    config["xtr_eta"] = config["eta"] if args.eta == -1 else args.eta

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        config["emb_dir"] = os.path.realpath(args.out_dir)

    # -- end of configuration

    logger = utils_multi.create_log_file(config["log_dir"], "extraction")
    if args.safe_gpu:
        gpu_owner = safe_gpu.GPUOwner(logger=logger)
        args.cuda = True
        config["cuda"] = True
    logger.info("Node : %s", platform.node())

    utils_multi.save_config(config)

    if args.verbose:
        logger().addHandler(logging.StreamHandler())
    logger.info("Loaded config file from %s", args.cfg_file)
    logger.info("PyTorch version: %s", str(torch.__version__))

    # -- end of log file creation

    xtr_data = {}
    with open(args.extract_data_json, "r") as fpr:
        xtr_data = json.load(fpr)

    cpu = torch.device("cpu")
    if args.model_f == "latest_trn_model":
        args.model_f = config["latest_trn_model"]

    if args.model_f is None or not os.path.exists(args.model_f):
        print("- Error: Model not found", args.model_f, file=sys.stderr)
        sys.exit()

    mbay_model = torch.load(args.model_f, map_location=cpu)

    mbase = os.path.basename(args.model_f).rsplit(".", 1)[0]

    for lid, npz_list in xtr_data.items():

        if lid not in mbay_model.T:
            print(
                "Language",
                lid,
                "not used in training the model. Hence, embeddings cannot be extracted."
            )
            continue

        config["xtr_bsize"] = args.bsize
        config["cuda"] = args.cuda

        for i, npz_file in enumerate(npz_list):

            print(" - {:3s} {:3d}/{:3d} {:s}".format(lid, i+1, len(npz_list), npz_file))

            extracted = False
            while not extracted:

                try:

                    extracted = extract_and_save(
                        npz_file, lid, mbay_model, config, mbase
                    )

                except RuntimeError as err:

                    if re.search(r"out of memory", str(err)):
                        config["xtr_bsize"] -= config["bsize_dec"]
                        logger.warning("Decreasing batch size to %d", config["xtr_bsize"])

                        if config["xtr_bsize"] < 1:
                            logger.error("Batch size < 1. Exiting ..")
                            print("Batch size < 1. Exiting ..", file=sys.stderr)
                            sys.exit()
                    else:
                        print("{0}".format(str(err)))
                        traceback.print_tb(err.__traceback__)
                        sys.exit()

        print()
