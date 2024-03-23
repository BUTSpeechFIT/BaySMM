#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common util functions
"""

import os
import sys
import json
import shlex
import codecs
import logging
import tempfile
import subprocess
from datetime import datetime
from time import sleep
import math
from collections import OrderedDict
# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import h5py
import torch


def create_log_file(log_dir, phase):
    """ Create log file """

    now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(log_dir, f"{phase}_{now_str}.log")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=log_file,
        level=logging.INFO,
    )
    print("Log file:", log_file)
    return logging.getLogger()


def print_model_params(model):
    """Print info about model parameters"""

    print("============== Model parameters ==============")
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.dtype, param.size())
    print("===============================================\n")


def get_num_trainable_params(model):
    """Return the number of trainable params in Millions"""
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    return num_params / 1e6


def load_multi_labels(label_file):
    """ Load labels from file and convert them into multi-label format.

        Args:
        ----
            label_file (str): Path to label file

        Returns:
        --------
            np.ndarray (mutli_labels): N x L multi label binary matrix
    """

    max_label = 0
    min_label = np.inf
    labels = []
    with open(label_file, 'r') as fpr:
        for line in fpr:
            labs = [int(i) for i in line.strip().split(",") if i]
            if np.max(labs) > max_label:
                max_label = np.max(labs)
            if np.min(labs) < min_label:
                min_label = np.min(labs)
            labels.append(labs)

    multi_labels = np.zeros(shape=(len(labels), max_label+1), dtype=int)
    for i, labs in enumerate(labels):
        labs = np.asarray(labs, dtype=int)
        if min_label == 1:
            labs -= 1
        multi_labels[i, np.asarray(labs, dtype=int)] = 1

    return multi_labels


def read_simple_flist(fname):
    """ Load a file into list. Should be called from smaller files only. """

    lst = []
    with codecs.open(fname, 'r') as fpr:
        lst = [line.strip() for line in fpr if line.strip()]
    return lst


def create_smm_config(args):
    """ Create configuration for SMM """

    exp_dir = os.path.realpath(args.out_dir) + "/"
    exp_dir += "lw_{:.0e}_{:s}_{:.0e}".format(args.lw, args.rt, args.lt)
    exp_dir += "_{:d}".format(args.K)

    if args.model_type in ("super", "hybrid"):
        exp_dir += "_lc_{:.0e}_{:.0e}_{:.0e}".format(args.lc, args.ag, args.ad)
        exp_dir += "_lper_{:.2f}".format(args.lper)

    exp_dir += "_{:s}_{:s}/".format(args.model_type, args.optim)

    if args.ovr:
        if os.path.exists(exp_dir):
            print('Overwriting existing output dir:', exp_dir)
            subprocess.check_call("rm -rf " + exp_dir, shell=True)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_file = exp_dir + "config.json"
    config = OrderedDict()

    try:
        config = json.load(open(cfg_file, 'r'))
        print('Config:', cfg_file, 'loaded.')
        os.makedirs(config['tmp_dir'], exist_ok=True)

    except IOError:

        ivecs_d = exp_dir + "ivecs/"
        os.makedirs(ivecs_d, exist_ok=True)
        os.makedirs(exp_dir + "results/", exist_ok=True)

        config['cfg_file'] = cfg_file  # this file
        config['exp_dir'] = exp_dir
        config['ivecs_dir'] = ivecs_d
        config['tmp_dir'] = tempfile.mkdtemp() + "/"

        try:
            config['training_stats_file'] = os.path.realpath(args.npz_file)
        except AttributeError:
            pass

        config['vocab_file'] = os.path.realpath(args.vocab_file)

        hyper = {'K': args.K, 'lam_w': args.lw, 'reg_t': args.rt,
                 'lam_t': args.lt, 'lam_c': args.lc,
                 'alf_g': args.ag, 'alf_d': args.ad}

        config['hyper'] = hyper
        config['eta_w'] = args.eta_w
        config['eta_t'] = args.eta_t
        config['eta_c'] = args.eta_c
        config['cuda'] = args.cuda
        config['optim'] = args.optim
        config['disc'] = args.model_type in ("super", "hybrid")
        config['update_ubm'] = args.update_ubm
        config['pos'] = args.pos
        config['mlabels'] = args.mlabels
        config['model_type'] = args.model_type
        config['trn_iters'] = args.trn
        config['save'] = args.save
        config['pytorch_ver'] = torch.__version__

        # useful to continue training or extracting from the latest model
        config['latest_trn_model'] = None

        config['trn_done'] = 0

        # print("Config file created.")
        json.dump(config, open(cfg_file, "w"), indent=2)

    return config


def create_baysmm_config(args):
    """ Create configuration for BaySMM """

    exp_dir = os.path.realpath(args.out_dir) + "/"
    exp_dir += "s_{:.2f}_rp_{:d}_".format(args.s, args.R)
    exp_dir += "lw_{:.0e}_{:s}_{:.0e}".format(args.lw, args.rt, args.lt)
    exp_dir += "_{:d}_{:s}".format(args.K, args.optim)

    if args.stoc:
        exp_dir += "_stoc"

    exp_dir += "/"

    if args.ovr:
        if os.path.exists(exp_dir):
            print('Overwriting existing output dir:', exp_dir)
            subprocess.check_call("rm -rf " + exp_dir, shell=True)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_file = exp_dir + "config.json"
    config = OrderedDict()

    try:
        config = json.load(open(cfg_file, 'r'))
        print('Config:', cfg_file, 'loaded.')
        os.makedirs(config['tmp_dir'], exist_ok=True)

    except IOError:

        ivecs_d = exp_dir + "ivecs/"
        os.makedirs(ivecs_d, exist_ok=True)
        os.makedirs(exp_dir + "results/", exist_ok=True)

        config['cfg_file'] = cfg_file  # this file
        config['exp_dir'] = exp_dir
        config['ivecs_dir'] = ivecs_d
        config['tmp_dir'] = tempfile.mkdtemp() + "/"
        # print("Temp dir:", config['tmp_dir'])

        try:
            config['training_stats_file'] = os.path.realpath(args.npz_file)
        except AttributeError:
            pass

        config['vocab_file'] = os.path.realpath(args.vocab_file)

        hyper = {'K': args.K, 'lam_w': args.lw, 'reg_t': args.rt,
                 'lam_t': args.lt, 'R': args.R}

        config['scale'] = args.s
        config['hyper'] = hyper
        config['eta_q'] = args.eta_q
        config['eta_t'] = args.eta_t
        config['cuda'] = args.cuda
        config['update_ubm'] = args.update_ubm
        config['trn_iters'] = args.trn
        config['pytorch_ver'] = torch.__version__
        config['save'] = args.save
        config['optim'] = args.optim
        config['var_p'] = args.var_p

        # useful to continue training or extracting from the latest model
        config['latest_trn_model'] = None

        config['trn_done'] = 0

        # print("Config file created.")
        json.dump(config, open(cfg_file, "w"), indent=2)

    return config


def create_baysmm_config_v2(args):
    """ Create configuration for BaySMM """

    exp_dir = os.path.realpath(args.out_dir) + "/"
    exp_dir += "s_{:.2f}_rp_{:d}_".format(args.s, args.R)
    exp_dir += "lw_{:.0e}_{:s}_{:.0e}".format(args.lw, args.rt, args.lt)
    # exp_dir += "_{:d}_map_{:d}_{:s}".format(args.K, args.optim)
    exp_dir += "_{:d}_map_{:d}_{:s}".format(args.K, args.init, args.optim)

    # if args.model_type in ("super", "hybrid"):
    #    exp_dir += "_lc_{:.0e}_{:.0e}_{:.0e}".format(args.lc, args.ag, args.ad)
    #    exp_dir += "_lper_{:.2f}".format(args.lper)

    # exp_dir += "_{:s}".format(args.model_type)

    exp_dir += "/"

    if args.ovr:
        if os.path.exists(exp_dir):
            print('Overwriting existing output dir:', exp_dir)
            subprocess.check_call("rm -rf " + exp_dir, shell=True)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_file = exp_dir + "config.json"
    config = OrderedDict()

    try:
        config = json.load(open(cfg_file, 'r'))
        print('Config:', cfg_file, 'loaded.')
        os.makedirs(config['tmp_dir'], exist_ok=True)

    except IOError:

        ivecs_d = exp_dir + "ivecs/"
        os.makedirs(ivecs_d, exist_ok=True)

        config['cfg_file'] = cfg_file  # this file
        config['exp_dir'] = exp_dir
        config['ivecs_dir'] = ivecs_d
        config['tmp_dir'] = tempfile.mkdtemp() + "/"
        print("Temp dir:", config['tmp_dir'])

        try:
            config['training_stats_file'] = os.path.realpath(args.npz_file)
        except AttributeError:
            pass

        config['vocab_file'] = os.path.realpath(args.vocab_file)

        hyper = {'K': args.K, 'lam_w': args.lw, 'reg_t': args.rt,
                 'lam_t': args.lt, 'R': args.R, 'alf_g': args.ag,
                 'alf_d': 0.}

        config['init'] = args.init
        config['disc'] = args.disc
        config['scale'] = args.s
        config['hyper'] = hyper
        config['eta_q'] = args.eta_q
        config['eta_t'] = args.eta_t
        config['cuda'] = args.cuda
        config['update_ubm'] = args.update_ubm
        config['trn_iters'] = args.trn
        config['pytorch_ver'] = torch.__version__
        config['save'] = args.save
        config['optim'] = args.optim

        # useful to continue training or extracting from the latest model
        config['latest_trn_model'] = None

        config['trn_done'] = 0

        # print("Config file created.")
        json.dump(config, open(cfg_file, "w"), indent=2)

    return config




def save_config(config):
    """ Save config file """

    json.dump(config, open(config['cfg_file'], "w"), indent=2)


def estimate_ubm(stats):
    """ Given the stats (scipy.sparse), estimate UBM (ML)

    Args:
        stats (scipy.sparse): Word by Doc sparse matrix of counts

    Returns:
        torch.Tensor of size (n_words x 1)
    """
    # universal background model or log-average dist. over vocabulary
    return torch.from_numpy(np.log((stats.sum(axis=1) /
                                    stats.sum()).reshape(-1, 1))).float()


def get_torch_dtype(dtype_str):
    """ Get torch data type """

    dtype = torch.float
    if dtype_str == 'double':
        dtype = torch.double
    return dtype


def write_info(config, info_str):
    """ Write / append information into file """

    if not isinstance(info_str, str):
        print("utils: `info_str` must be a string.")
        sys.exit()

    mode = "w"
    info_file = config['exp_dir'] + "info.txt"
    if os.path.exists(info_file):
        mode = "a"

    with open(info_file, mode) as fpw:
        fpw.write(info_str + "\n")


def save_model(model):
    """ Save model parameters and config. """

    sfx = str(model.config['trn_done']) + ".h5"
    model.config['latest_trn_model'] = model.config['exp_dir'] + 'model_T' + sfx

    h5f = h5py.File(model.config['latest_trn_model'], 'w')
    params = h5f.create_group('params')
    ivecs = h5f.create_group('ivecs')

    if model.__class__.__name__ == "SMM":
        # In case of simple SMM, the model parameters are
        # matrix T and vector m
        # i-vectors / embeddings are W
        ivecs.create_dataset('W', data=model.W.data.cpu().numpy())
        if model.config['disc']:
            disc_params = h5f.create_group('disc_params')
            c_params = disc_params.create_group('C')
            b_params = disc_params.create_group('b')
            for l_ix, _ in enumerate(model.C):
                c_params.create_dataset(str(l_ix),
                                        data=model.C[l_ix].data.cpu().numpy())
                b_params.create_dataset(str(l_ix),
                                        data=model.b[l_ix].data.cpu().numpy())
        params.create_dataset('T', data=model.T.data.cpu().numpy())
        params.create_dataset('m', data=model.m.data.cpu().numpy())

    elif model.__class__.__name__ == "BaySMM":
        # In case of Bayesian SMM, the model parameters are
        # matrix T and vector m
        # i-vector posterior distribution params ae Q
        ivecs.create_dataset('Q', data=model.Q.data.cpu().numpy())
        params.create_dataset('T', data=model.T.data.cpu().numpy())
        params.create_dataset('m', data=model.m.data.cpu().numpy())

    elif model.__str__() == 'MDSMM':

        ivecs.create_dataset('W', data=model.W.data.cpu().numpy())

        t_params = params.create_group('T_list')
        m_params = params.create_group('m_list')

        for idx, T in enumerate(model.T):
            t_params.create_dataset(str(idx),
                                    data=T.data.cpu().numpy())
            m_params.create_dataset(str(idx),
                                    data=model.m[idx].data.cpu().numpy())
    else:
        logging.error("Unkown model. Cannot save parameters.")
        sys.exit()

    h5f.close()

    save_config(model.config)

    logging.info("Model parameters saved: %s", model.config['latest_trn_model'])


def save_loss(loss_iters, config, base, sfx):
    """ Save loss over iterations into file """

    loss_file = config['exp_dir'] + base + sfx + "_loss.txt"
    mode = "w"

    if os.path.exists(loss_file):
        mode = "a"
        logging.info('Appending to existing loss file: %s', loss_file)

    with open(loss_file, mode) as fpw:
        np.savetxt(fpw, loss_iters, fmt="%.4f")

    logging.info("Loss over iters saved: %s", loss_file)


def load_params(model_f):
    """ Load parameters of existing model """

    params = {}

    # import pdb
    # pdb.set_trace()

    try:
        h5f = h5py.File(model_f, 'r')
        param_grp = h5f.get('params')
        ivecs_grp = h5f.get('ivecs')

        if 'm' in param_grp:
            params['m'] = torch.from_numpy(param_grp.get('m')[()])
        if 'T' in param_grp:
            params['T'] = torch.from_numpy(param_grp.get('T')[()])

        if 'm_list' in param_grp:
            m_list = []
            t_list = []
            m_grp = param_grp.get('m_list')
            t_grp = param_grp.get('T_list')
            for i, _ in enumerate(t_grp):
                m_list.append(torch.from_numpy(m_grp.get(str(i))[()]))
                t_list.append(torch.from_numpy(t_grp.get(str(i))[()]))

            params['m'] = m_list
            params['T'] = t_list

        if 'W' in ivecs_grp:
            params['W'] = torch.from_numpy(ivecs_grp.get('W')[()])
        elif 'Q' in ivecs_grp:
            params['Q'] = torch.from_numpy(ivecs_grp.get('Q')[()])

        else:
            logging.warning("i-vectors not found in: %s", model_f)
            sys.exit()

        if 'disc_params' in h5f:
            disc_grp = h5f.get('disc_params')
            c_grp = disc_grp.get('C')
            b_grp = disc_grp.get('b')
            c_params = []
            b_params = []
            for i, _ in enumerate(c_grp):
                c_params.append(torch.from_numpy(c_grp.get(str(i))[()]))
                b_params.append(torch.from_numpy(b_grp.get(str(i))[()]))

            params['C'] = c_params
            params['b'] = b_params

    except IOError as err:
        raise IOError("Cannot load model:", model_f, err)

    except AttributeError as err:
        raise AttributeError("Cannot find model parameters.", err)

    finally:
        h5f.close()

    return params


def load_stats(stats_f, vocab_f):
    """ Validate and load the input stats """

    stats = scipy.sparse.load_npz(stats_f)

    if vocab_f:
        vocab = read_simple_flist(vocab_f)

        # Check the compatibility of stats
        if stats.shape[1] == len(vocab):
            stats = stats.T
            print("Transposed the stats to make them word-by-doc.")
            scipy.sparse.save_npz(os.path.realpath(stats_f), stats)

        if stats.shape[0] != len(vocab):
            print("Number of rows in stats should match with length of vocabulary.")
            print("Given stats:", stats.shape[0], "vocab. length:", len(vocab))
            sys.exit()

    return stats.tocsc()


def merge_ivecs(config, sbase, mbase, xtr_iters):
    """ Merge batches of i-vectors and corresponding LLH

    Args:
        config (dict): configuration dict
        sbase (str): base name of stats file
        mbase (str): base name of model file
        xtr_iter (int): number of extraction iterations
        n_batches (int): number of batches
    """

    out_file = config['ivecs_dir'] + sbase + "_" + mbase + "_e" + str(xtr_iters) + ".h5"

    h5f = h5py.File(out_file, 'w')
    g_ivecs = h5f.create_group('ivecs')
    g_llh = h5f.create_group('llh')

    data = []
    # llh = []
    for i in range(1, xtr_iters+1):
        # for bix in range(n_batches):
        fname = config['tmp_dir'] + "ivecs_" + sbase + "_"
        fname += str(i) + ".npy"
        # ename = fname.replace("ivecs_", "llh_")
        if os.path.exists(fname):
            # data.append(np.load(fname))
            # llh.append(np.load(ename))
            data = np.load(fname)
            # if data:
            g_ivecs.create_dataset(str(i), data=data,
                                   compression='gzip')
            # g_llh.create_dataset(str(i), data=np.concatenate(llh, axis=0),
            #                      compression='gzip')

        # else:
        #    print(fname, 'not found.')

    h5f.close()

    print("All i-vectors saved to:", out_file)
    subprocess.check_call("rm -rf " + config['tmp_dir'] + "*.npy", shell=True)
    print("Deleted npy files.")


def merge_ivecs_v2(config, sbase, mbase, xtr_iters, n_batches):
    """ Merge batches of i-vectors and corresponding LLH

    Args:
        config (dict): configuration dict
        sbase (str): base name of stats file
        mbase (str): base name of model file
        xtr_iter (int): number of extraction iterations
        n_batches (int): number of batches
    """

    out_file = config['ivecs_dir'] + sbase + "_" + mbase + "_e" + str(xtr_iters) + ".h5"

    h5f = h5py.File(out_file, 'w')
    g_ivecs = h5f.create_group('ivecs')

    data = []
    for i in range(1, xtr_iters+1):
        for bix in range(n_batches):
            fname = config['tmp_dir'] + "ivecs_" + sbase + "_"
            fname += str(i) + "_b_" + str(bix) + ".npy"
            if os.path.exists(fname):
                data.append(np.load(fname))

        if data:
            g_ivecs.create_dataset(str(i), data=np.concatenate(data, axis=1),
                                   compression='gzip')
            data = []

    h5f.close()

    logging.info("All i-vectors saved to: %s", out_file)
    subprocess.check_call("rm -rf " + config['tmp_dir'] + "*.npy", shell=True)
    logging.info("Deleted npy files from: %s", config['tmp_dir'])


def merge_ivecs_llh(config, sbase, mbase, xtr_iters, n_batches):
    """ Merge batches of i-vectors and corresponding LLH

    Args:
        config (dict): configuration dict
        sbase (str): base name of stats file
        mbase (str): base name of model file
        xtr_iter (int): number of extraction iterations
        n_batches (int): number of batches
    """

    out_file = config['ivecs_dir'] + sbase + "_" + mbase + "_e" + str(xtr_iters) + ".h5"

    h5f = h5py.File(out_file, 'w')
    g_ivecs = h5f.create_group('ivecs')
    # g_llh = h5f.create_group('llh')

    data = []
    llh = []
    for i in range(1, xtr_iters+1):
        for bix in range(n_batches):
            fname = config['tmp_dir'] + "ivecs_" + sbase + "_b" + str(bix) + "_" + mbase + "_e"
            fname += str(i) + ".npy"
            # ename = fname.replace("ivecs_", "llh_")
            if os.path.exists(fname):
                data.append(np.load(fname))
                # llh.append(np.load(ename))

        if data:

            g_ivecs.create_dataset(str(i), data=np.concatenate(data, axis=0),
                                   compression='gzip')
            # g_llh.create_dataset(str(i), data=np.concatenate(llh, axis=0),
            #                     compression='gzip')

            data = []
            # llh = []

    h5f.close()

    print("All ivectors saved to", out_file)
    # subprocess.check_call("rm -rf " + config['tmp_dir'] + "*.npy", shell=True)
    # print("Deleted npy files.")bn


def save_ivecs_to_h5(in_ivecs_dir, mbase, xtr_iters):
    """ Save all ivecs to h5 format to save disk space. """

    print("Compressing i-vectors..")
    in_files = os.listdir(in_ivecs_dir)

    out_file = in_ivecs_dir + "ivecs_" + mbase + "_e" + str(xtr_iters) + ".h5"
    h5f = h5py.File(out_file, 'w')

    grp_train = h5f.create_group('train')
    grp_test = h5f.create_group('test')

    for in_file in in_files:

        base, ext = os.path.splitext(os.path.basename(in_file))

        if ext == ".npy":

            data = np.load(in_ivecs_dir + in_file)

            if "train" in base:
                grp_train.create_dataset(base, data=data, compression='gzip')
            else:
                grp_test.create_dataset(base, data=data, compression='gzip')

    h5f.close()

    print("All ivectors saved to", out_file)
    subprocess.check_call("rm -rf " + in_ivecs_dir + "*.npy", shell=True)
    print("Deleted npy files.")


def t_sparsity(model):
    """ Compute sparsity percentage in T """

    if isinstance(model.T, torch.nn.parameter.Parameter):
        V, K = model.T.size()
        nnz = np.count_nonzero(model.T.data.cpu().clone().numpy())
        s_rat = ((V * K) - nnz) / (V * K)
    else:
        nnz = 0
        total_params = 0
        for T in model.T:
            V, K = T.size()
            total_params += (V * K)
            nnz += np.count_nonzero(T.data.cpu().clone().numpy())
        s_rat = (total_params - nnz) / total_params
    return s_rat * 100.


def compute_avg_doc_ppl(elbo, X):
    """ Given LLH or ELBO per doc and corresponding stats,
    computes average document perplexity """

    ppl = torch.exp(-(torch.sum(elbo / X.sum(dim=0)))/X.size()[1])
    return ppl.cpu().numpy()


def compute_unigram_ppl(elbo, X):
    """ Given LLH or ELBO per doc and corresponding stats,
    computes unigram perplexity across corpus """

    ppl = torch.exp(-torch.sum(elbo) / X.sum())
    return ppl.cpu().numpy()


def get_free_gpu(ignore_ids=None):
    """ Get free GPU device ID """

    if ignore_ids is None:
        ignore_ids = []

    num_retry = 0
    max_retry = 5

    while num_retry < max_retry:

        cmd = "nvidia-smi --query-gpu=count,index,memory.used,utilization.gpu --format='csv'"
        args = shlex.split(cmd)

        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()

        out = out.decode('ascii').strip()
        err = err.decode('ascii').strip()
        if err:
            print(err)
            sys.exit()

        lines = out.split("\n")[1:]
        gpu_index = -1
        for line in lines:
            vals = line.split(",")
            if int(vals[2].split()[0]) < 15 and int(vals[3].split()[0]) == 0:
                gpu_index = int(vals[1])
                if gpu_index in ignore_ids:
                    gpu_index = -1
                else:
                    break

        if gpu_index == -1:
            num_retry += 1
            sleep(np.random.randint(30, 120))
            print("* Retrying to get a free GPU. Attempt {:d}/{:d}".format(num_retry,
                                                                           max_retry))
            print(out)

        else:
            break

    if gpu_index == -1:
        print("* No free GPUs. *\n")
        sys.exit()

    print("* GPU ID:", gpu_index)
    return gpu_index


def get_gpu_total_memory(gpu_id=0):
    """ Get GPU total memory """

    cmd = "nvidia-smi --query-gpu=memory.total --format=csv"
    args = shlex.split(cmd)

    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()

    out = out.decode('ascii').strip()
    err = err.decode('ascii').strip()
    if err:
        print(str(err))
        sys.exit()
    line = out.split("\n")[1+gpu_id]

    mem = line.split()[0]

    print('mem:', mem)

    return int(mem)


def print_config_info(config):
    """ Print information in configuration """

    from pprint import pprint

    pprint(config)


def estimate_approx_num_batches(model, stats):
    """ Estimate the number of batches approximately """

    const = float(32 / 1e+6)
    stats_size = stats.nnz * 3
    params_size = (model.T.size()[0] * model.T.size()[1]) + model.m.size()[0]

    if model.__class__.__name__ == 'SMM':
        ivecs_size = (model.W.size()[0] * model.W.size()[1])
    elif model.__class__.__name__ == 'BaySMM':
        ivecs_size = (model.Q.size()[0] * model.Q.size()[1])
        ivecs_size += (model.eps.size()[0] * model.eps.size()[1] * model.eps.size()[2])

    gpu_mem = get_gpu_total_memory()
    den = gpu_mem - (const * (stats_size + (4 * params_size) + ivecs_size))
    num_b = math.ceil((3 * ivecs_size * const) / den)
    tot_size = (stats_size + ((4 * (params_size + ivecs_size))) * 3) * const
    print('estimated total size:', tot_size)
    print('estimated number of batches:', num_b)


def plot_loss(loss_iters, config, base):
    """ Plot and save the loss (LLH) over iterations """

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.style.use('presentation')

    plt.figure(1)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    title_str = "Weight of disc. loss $\\alpha$ "
    title_str += "={:.1f}, K={:d}".format(config['hyper']['alf_d'],
                                          config['hyper']['K'])

    plt.title(title_str)

    loss_name = 'Total loss'

    if loss_iters.shape[1] == 1:
        loss_name = 'Gen. loss'

        plt.plot(np.arange(loss_iters.shape[0]), loss_iters[:, 0], '.-',
                 color='C0', label=loss_name)

    else:

        gloss_iters = loss_iters[:, 0] - loss_iters[:, 1]

        plt.plot(np.arange(loss_iters.shape[0]), loss_iters[:, 0], '-',
                 alpha=0.5, color='C0', label=loss_name)

        plt.plot(np.arange(gloss_iters.shape[0]), gloss_iters, '-',
                 alpha=1, color='C1', label='Gen. loss')

        plt.plot(np.arange(loss_iters.shape[0]), loss_iters[:, 1], '-',
                 color='C2', label='Disc. loss')

    plt.grid()
    plt.legend(loc='best')

    os.makedirs(config['exp_dir'] + "plots/", exist_ok=True)
    img_file = config['exp_dir'] + "plots/" + base
    plt.savefig(img_file + ".png", dpi=300)
    plt.savefig(img_file + ".pdf", dpi=300)
    # plt.show()
    print("Image saved as", img_file, "{.png .pdf}")
