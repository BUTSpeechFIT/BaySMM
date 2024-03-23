#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Data utililty functions
"""

import os
import sys
import json
import codecs
import random
import argparse
import math
import numpy as np
import scipy.sparse
import torch


class SMMDataset:
    """SMM dataset.

    A dataset class for training document i-vector model
    (Subspace Multinomial Model).
    """

    def __init__(
        self,
        data_npz,
        labels,
        vocab_len,
        dset_type="super",
        multi_label=False,
        pos="upper",
    ):
        """Initialize a dataset for training an SMM or
        Discriminative SMM model.

        Args:
        -----
            data_npz_file (scipy.sparse or str): scipy.sparse matrix
                or path to scipy.sparse data matrix file
            label_file (numpy.ndarray or str): numpy.ndarray of labels
                or path label file, where every row is a label.
            vocab_len (int): Vocabulary length
            dset_type (str): Dataset type can be `super` - supervised i.e.,
                all the documents has labels. `unsup` - unsupervised i.e.,
                none of the documents have labels, in this case `labels`
                can be `None`. `hybrid` - it means that the `data_npz` has both
                labelled and unlabelled data. The labelled data is in the
                upper part (by default) of the matrix with the corresponding labels in
                `labels` and the unlabelled data is at the lower part of
                `data_npz`.
            multi_label (bool): Multiple-labels per document ?
            pos (str): Position of labelled data in data_npz (default is upper),
                can be `lower`

        Returns:
        --------
            SMMDataset object
        """

        self.data_npz = None
        self.labs = None
        self.n_labels = None
        self.n_docs = None
        self.pos = pos
        self.vocab_len = vocab_len
        self.dset_type = dset_type
        self.device = torch.device("cpu")

        self.__load_npz(data_npz)

        if dset_type in ("super", "hybrid"):
            self.__load_labels(labels)

        elif dset_type == "unsup":
            pass

        else:
            raise ValueError("Dataset:" + dset_type + " not understood.")

        self.random_data_batches = []

    def __len__(self):
        """Length of the dataset"""
        return self.n_docs

    def __load_npz(self, data_npz_file):
        """Load scipy.sparse matrix (npz) file"""

        if scipy.sparse.issparse(data_npz_file):
            self.data_npz = data_npz_file
        elif isinstance(data_npz_file, str):
            try:
                self.data_npz = scipy.sparse.load_npz(data_npz_file).tocsc()
            except IOError as err:
                raise IOError(
                    "SMMDataset: Unable to load data npz file", data_npz_file, err
                )

        # convert to Doc-by-Word shape
        if self.vocab_len == self.data_npz.shape[0]:
            self.data_npz = self.data_npz.T

        if self.vocab_len != self.data_npz.shape[1]:
            print(
                "SMMDataset: Vocabulary length ({:d})".format(self.vocab_len),
                "should match dim 1 of data_npz",
                "({:d})".format(self.data_npz.shape[1]),
                file=sys.stderr,
            )
            sys.exit()

        self.n_docs = self.data_npz.shape[0]

    def __load_labels(self, labels):
        """Load labels"""

        if isinstance(labels, np.ndarray):
            pass

        elif isinstance(labels, str):
            try:
                labels = np.loadtxt(labels).astype(int)
            except IOError as err:
                raise IOError("Dataset: Unable to load label file:", labels, err)

        self.n_labels = np.unique(labels).shape[0]

        if np.min(labels) == 1:
            print("Dataset: Adjusting labels to start from 0")
            labels -= 1

        if self.dset_type == "super":
            if labels.shape[0] != self.n_docs:
                print(
                    "Dataset: Number of documents ({:d})".format(self.n_docs),
                    "should match number of labelled documents",
                    "({:d})".format(labels.shape[0]),
                )
                sys.exit()

        self.labs = labels

    def to_device(self, device):
        """To CPU or CUDA"""
        self.device = device

    def get_total_batches(self, n_batches: int) -> int:
        """Get total number of batches"""

        if self.dset_type == "super":
            n_docs_w_labels = self.labs.shape[0]
            bsize = math.ceil(n_docs_w_labels / n_batches)
            data_n_batches = n_batches

        elif self.dset_type == "unsup":
            bsize = math.ceil(self.n_docs / n_batches)
            data_n_batches = n_batches

        else:
            n_docs_w_labels = self.labs.shape[0]
            bsize = math.ceil(n_docs_w_labels / n_batches)
            data_n_batches = math.ceil(self.n_docs / bsize)

        return data_n_batches, bsize

    def create_batches(self, n_batches):
        """Create batches, where every batch will have start, end
        info corresponding to data chunk"""

        if self.dset_type == "unsup":
            n_docs_w_labels = 0
        else:
            n_docs_w_labels = self.labs.shape[0]

        data_n_batches, bsize = self.get_total_batches(n_batches)

        six = 0
        eix = bsize
        leix = bsize

        for _ in range(data_n_batches):
            label_rng = None  # batch_labs = torch.Tensor([])

            if self.pos == "upper":
                if leix < n_docs_w_labels:
                    label_rng = [six, leix]

                elif six < n_docs_w_labels <= leix:
                    leix = n_docs_w_labels
                    eix = n_docs_w_labels
                    label_rng = [six, leix]

                elif eix > self.n_docs:
                    eix = self.n_docs

                else:
                    pass

            else:
                off = self.n_docs - n_docs_w_labels
                if six < off < eix:
                    eix = off
                    leix = off

                elif six >= off:
                    label_rng = [six - off, leix - off]

                elif eix > self.n_docs:
                    eix = self.n_docs

                else:
                    pass

            self.random_data_batches.append(
                {"six": six, "eix": eix, "label_rng": label_rng}
            )
            six = eix
            eix += bsize
            leix += bsize

    def yield_batches(self, n_batches, shuffle=True):
        """Yield data in batches, where `n_batches` are minimum number of
        batches.

        Args:
        -----
            n_batches (int): The minimum number of batches.
        """

        if not self.random_data_batches:
            self.create_batches(n_batches)

        if shuffle:
            random.shuffle(self.random_data_batches)

        for data_batch in self.random_data_batches:
            six = data_batch["six"]
            eix = data_batch["eix"]
            label_rng = data_batch["label_rng"]

            batch_labs = torch.Tensor([])  # empty by default
            if label_rng:
                batch_labs = torch.from_numpy(
                    self.labs[label_rng[0] : label_rng[1]]
                ).to(self.device)

            batch_npz = self.data_npz[six:eix, :].T.tocoo()

            rng = [
                torch.LongTensor([six]).to(device=self.device),
                torch.LongTensor([eix]).to(device=self.device),
            ]

            rixs, cixs = batch_npz.nonzero()
            ixs = (
                torch.concat(
                    (
                        torch.from_numpy(rixs).view(1, -1),
                        torch.from_numpy(cixs).view(1, -1),
                    ),
                    dim=0,
                )
                .long()
                .to(self.device)
            )
            counts = torch.Tensor(batch_npz.data.astype(np.float32)).to(
                device=self.device
            )

            yield {"ixs": ixs, "counts": counts, "rng": rng, "Y": batch_labs}

    def get_data_npz(self):
        """Return stats in scipy.sparse.csc format"""
        return self.data_npz.tocsc()

    def get_labels(self):
        """Return labels in numpy ndarray format"""
        return self.labs.numpy()

    def get_labels_tensor(self):
        """Return labels in torch.Tensor format"""
        return self.labs

    def get_data_tensor(self):
        """Return data in torch.Tensor"""
        return torch.from_numpy(self.data_npz.A).float()

    def get_n_labels(self):
        """Return number of unique labels"""
        return self.n_labels


class MBaySMMDataset:
    """Dataset for training Multilingual BaySMM model"""

    def __init__(self, data_json_file, lang_vocab_json_file):
        """Initialize the dataset"""

        # self.device = torch.device("cuda" if cuda else "cpu")

        self.info = {}
        self.data = {}
        self.lang2vocab = {}
        self.languages = set()
        self.par2docs = {}
        self.par2langs = {}

        self._validate_data_and_load(data_json_file, lang_vocab_json_file)

    @property
    def n_docs(self):
        return sum([(d * self.par2langs[p_num]) for p_num, d in self.par2docs.items()])

    def get_vocab_size(self, vocab_file):
        """Get vocab size by reading the vocab_file that can be a json dict or plain text."""

        vocab_size = -1
        ext = os.path.basename(vocab_file).split(".")[-1]
        if ext == "json":
            vocab = {}
            with open(vocab_file, "r", encoding="utf-8") as fpr:
                vocab = json.load(fpr)
                vocab_size = len(vocab)
        else:
            vocab_size = len(read_simple_flist(vocab_file))

        return vocab_size

    def _validate_data_and_load(self, data_json_file, lang_vocab_json_file):
        """Validate the data - checks the format of json file. Then loads it into dict"""

        with open(data_json_file, "r", encoding="utf-8") as fpr:
            data_json = json.load(fpr)
        self.info = data_json

        with open(lang_vocab_json_file, "r", encoding="utf-8") as fpr:
            lang_vocab = json.load(fpr)

        for p_num in data_json:
            par_set = data_json[p_num]

            par_data = {}
            if p_num in self.data:
                par_data = self.data[p_num]

            for lid in par_set["npz"]:
                npz = scipy.sparse.load_npz(par_set["npz"][lid]).tocsr()
                try:
                    vocab_size = self.get_vocab_size(lang_vocab[lid])
                except KeyError:
                    print(
                        f"- Error: {lid} found in {lang_vocab_json_file}",
                        file=sys.stderr,
                    )
                    sys.exit()

                assert (
                    npz.shape[1] == vocab_size
                ), "dim 1 in {:s} ({:d}) != ({:d}) vocab_size from {:s}".format(
                    par_set["npz"][lid], npz.shape[1], vocab_size, lang_vocab[lid]
                )

                par_data[lid] = npz
                self.languages.add(lid)

                if lid not in self.lang2vocab:
                    self.lang2vocab[lid] = vocab_size

                assert (
                    self.lang2vocab[lid] == vocab_size
                ), "vocab_size mismatch for language: {:s}  {:d} != {:d}".format(
                    lid, self.lang2vocab[lid], vocab_size
                )

                if p_num not in self.par2docs:
                    self.par2docs[p_num] = npz.shape[0]

                assert (
                    self.par2docs[p_num] == npz.shape[0]
                ), "Number of parallel docs mismatch in parallel set number {:2s}, {:d} != {:d}".format(
                    p_num, self.par2docs[p_num], npz.shape[0]
                )

                try:
                    self.par2langs[p_num] += 1
                except KeyError:
                    self.par2langs[p_num] = 1

            self.data[p_num] = par_data

    def __str__(self):
        print_str = "========== Dataset ===========\n"
        for p_num in self.data:
            print_str += "{:2s}\n".format(p_num)
            for lid in self.data[p_num]:
                print_str += " !_._ {:2s} [{:7d} x {:6d} ]\n".format(
                    lid, *self.data[p_num][lid].shape
                )
        print_str += "=============================="
        return print_str.strip()

    def yield_batches(self, bsize: int, shuffle: bool):
        """A generator function that yields batches in the form of a nested dictionary.

        Args:
        ----
            bsize (int): Number of docs per language per batch

        Returns:
        --------
            (torch.LongTensor, dict): Tuple of (p_num, doc_ixs, batch_dict). \
              Where `batch_dict` is a nested dictionary of the format \
              `{lid: {counts: torch.FloatTensor, ixs: torch.LongTensor}`
        """

        p_nums = [p for p in self.data]
        if shuffle:
            random.shuffle(p_nums)

        for p_num in p_nums:
            par_data = self.data[p_num]

            row_ixs = np.arange(0, self.par2docs[p_num], 1, dtype=np.int64)

            # print("p_num:", p_num, "n_docs:", row_ixs.size)

            if shuffle:
                np.random.shuffle(row_ixs)

            six = 0
            eix = bsize if bsize < row_ixs.size else row_ixs.size

            while six < eix and eix <= row_ixs.size:
                # print("six:", six, "eix:", eix)

                batch_dict = {}
                batch_row_ixs = row_ixs[six:eix]

                doc_ixs = torch.from_numpy(batch_row_ixs).to(dtype=torch.long)
                for lid in par_data:
                    batch_npz = par_data[lid][batch_row_ixs, :]

                    rixs, cixs = batch_npz.nonzero()
                    ixs = torch.concat(
                        (
                            torch.from_numpy(rixs).view(1, -1),
                            torch.from_numpy(cixs).view(1, -1),
                        ),
                        dim=0,
                    ).long()
                    batch_dict[lid] = {
                        "counts": torch.from_numpy(
                            batch_npz.data.astype(np.float32)
                        ).to(dtype=torch.float),
                        "ixs": ixs,
                    }

                yield (p_num, doc_ixs, batch_dict)

                six = eix
                eix += bsize
                if eix > row_ixs.size:
                    eix = row_ixs.size

            # break


def read_simple_flist(fname):
    """Load a file into list. Should be called from smaller files only."""

    lst = []
    with codecs.open(fname, "r") as fpr:
        lst = [line.strip() for line in fpr if line.strip()]
    return lst


def test():
    """Test method for dataset and yielding batches"""

    args = parse_arguments()

    # npz_flist = read_simple_flist(args.train_list)
    # vocab_flist = read_simple_flist(args.vocab_list)
    # label_flist = read_simple_flist(args.label_list)

    # dset = MDSMMDataset(npz_flist, label_flist, vocab_flist,
    #                     dset_type='hybrid')

    # for data_dict in dset.yield_batches(args.bsize):

    #     print('idx:', data_dict['idx'],
    #           'doc ixs range:', data_dict['rng'][0].numpy(), data_dict['rng'][1].numpy(),
    #           'Y:', data_dict['Y'].shape)

    # -- new --

    dset = MBaySMMDataset(args.data_json_file, args.lang_vocab_json_file)
    print(dset)
    print("-----------------")
    for bno, (p_num, doc_ixs, batch_dict) in enumerate(
        dset.yield_batches(args.bsize, args.shuffle)
    ):
        print("batch:", bno, "p_num:", p_num, "doc_ixs:", doc_ixs.shape)
        for lid, par_dict in batch_dict.items():
            print("!_._ ", lid, end=" : ")
            for par, v in par_dict.items():
                print(par, ":", v.shape, end=" ")

            print()
        if bno > 5:
            break


def parse_arguments():
    """Parser command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # parser.add_argument("train_list", help="path to train list")
    # parser.add_argument("vocab_list", help="path to vocab list")
    # parser.add_argument("label_list", help="path to labels list")
    parser.add_argument("-bsize", type=int, default=1000, help="batch size")
    parser.add_argument("--shuffle", action="store_true")
    # -- new --
    parser.add_argument("data_json_file")
    parser.add_argument("lang_vocab_json_file")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    test()
