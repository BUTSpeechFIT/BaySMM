#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 22 Mar 2021
# Last modified : 22 Mar 2021

"""
Cluster with k-means and get top-K representative words for each cluster
"""

import os
import sys
import argparse
import json
import h5py
import sklearn
import torch
import numpy as np
from baysmm import BaySMM
import pickle
import utils
from sklearn.cluster import KMeans, MiniBatchKMeans


def get_ivectors_from_h5(h5_fname, config):
    """Load ivectors from h5 file"""

    try:
        h5f = h5py.File(h5_fname, "r")
        h5_ivecs = h5f.get("ivecs")

        # get max_iters (final iter) from h5
        # since we even save intermediate embeddings during inference
        max_iters = 0
        for key in h5_ivecs:
            if int(key) > max_iters:
                max_iters = int(key)

        embs = h5_ivecs.get(str(max_iters))[()]
        if embs.shape[0] == config["hyper"]["K"] * 2:
            embs = embs.T

        embs = embs[
            :, : config["hyper"]["K"]
        ]  # take only the mean as the document embedding

    except IOError as err:
        print(err)

    finally:
        h5f.close()

    return embs


def load_json(fname):
    """Load json into dict"""
    data = {}
    with open(fname, "r", encoding="utf-8") as fpr:
        data = json.load(fpr)
    return data


def load_model(config):
    """Load trained model"""

    params = utils.load_params(config["latest_trn_model"])

    model = BaySMM(params["m"], config, config["cuda"])
    model.T.data = params["T"]
    model.Q.data = params["Q"]

    print("Loaded model from:", config["latest_trn_model"])

    return model


def main():
    """main method"""

    args = parse_arguments()

    config = load_json(args.config_file)
    config["cuda"] = False
    print("Loaded config.")

    vocab = load_json(args.vocab_file)
    int2vocab = {}
    for word, idx in vocab.items():
        int2vocab[idx] = word
    print("Loaded vocab:", len(vocab))

    model = load_model(config)
    model.requires_grad_(False)

    embs = None
    if args.ivecs_h5:
        print("Loaded embeddings.")
        embs = get_ivectors_from_h5(args.ivecs_h5, config)
    else:
        print("Using embeddings from model.")
        embs = model.Q.detach().numpy()[:, : config["hyper"]["K"]]

    print("Doc embeddings", embs.shape)

    if embs.shape[0] > 1e4:
        print(
            "Large number of samples {:d}. Using MiniBatchKmeans".format(embs.shape[0])
        )
        kmeans = MiniBatchKMeans(n_clusters=args.k, n_init=10, random_state=args.seed)
    else:
        kmeans = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed)

    pred_ixs = kmeans.fit_predict(embs)

    cluster_ixs, cluster_strength = np.unique(pred_ixs, return_counts=True)

    cluster_strength = cluster_strength / pred_ixs.shape[0]

    sorted_clusters = np.argsort(cluster_strength)[::-1]
    # print(sorted_clusters, cluster_ixs[sorted_clusters])

    global_mean = np.mean(embs, axis=0)

    centroids = kmeans.cluster_centers_

    scores = model.T.detach().numpy() @ (centroids - global_mean).T
    # scores = model.T.detach().numpy() @ centroids.T

    print()
    for k in sorted_clusters[: args.topn]:
        k_ixs = np.argsort(scores[:, k])[::-1]
        print("Cluster {:3d}:".format(k), end=" ")
        for i in k_ixs[: args.topk]:
            print(int2vocab[i], end=", ")
        print("\n")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("vocab_file", help="path to vocab.json file")
    parser.add_argument(
        "config_file",
        help="path to config.json file, embeddings will be taken from here",
    )
    parser.add_argument(
        "-ivecs_h5",
        default="",
        help="path to ivectors / embeddings h5 file. If not given, will take embeddings from trained model",
    )
    parser.add_argument(
        "-k", type=int, default=20, help="number of clusters for k-means"
    )
    parser.add_argument("-topn", type=int, default=10, help="top n cluster to use")
    parser.add_argument("-topk", type=int, default=10, help="top k words per cluster")
    parser.add_argument("-seed", type=int, default=2, help="random seed")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("sklearn version", sklearn.__version__)
    main()
