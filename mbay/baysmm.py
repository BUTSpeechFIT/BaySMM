#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian subspace multinomial model for learning
document i-vector (embedding) posterior distributions
"""

import os
import sys
import logging
from time import time
import json
import numpy as np
import torch
from torch import nn
import h5py


class BaySMM(nn.Module):
    """Bayesian Subspace Multinomial Model"""

    def __init__(self, ubm, config, cuda=False):
        """Initialize model

        Args:
        -----
            ubm (torch.Tensor) : Universal background model.
                See `utils.estimate_ubm` to get it.
            config (dict)      : Dictionary with configuration
                information. See `utils.create_baysmm_config`
                method on how to create and get it.
        """

        super(BaySMM, self).__init__()

        self.config = config

        self.device = torch.device("cuda" if cuda else "cpu")

        self.dtype = torch.float

        # if config['dtype'] == 'double' and not cuda:
        #    self.dtype = torch.double

        self.m = None
        self.T = None
        self.Q = None
        self.eps = None

        self.__init_params(ubm)

    def __init_params(self, ubm):
        """Initialize model parameters."""

        self.m = nn.Parameter(ubm, requires_grad=True)

        V = self.config["vocab_size"]
        N = self.config["n_docs"]
        K = self.config["hyper"]["K"]
        R = self.config["hyper"]["R"]

        torch.manual_seed(0)  # for consistent results on CPU and GPU

        # initialize bases or subspace or total variability matrix
        self.T = nn.Parameter(torch.randn(V, K) * 0.001, requires_grad=True)

        # initialize i-vector posterior distributions = variational params
        self.init_var_params(N)

        self.eps = nn.Parameter(torch.randn(R, N, K), requires_grad=False)

        self.lam_w = nn.Parameter(
            torch.Tensor([self.config["hyper"]["lw"]]), requires_grad=False
        )
        self.lam_t = nn.Parameter(
            torch.Tensor([self.config["hyper"]["lt"]]), requires_grad=False
        )
        self.K = nn.Parameter(torch.Tensor([K]).int(), requires_grad=False)
        self.R = nn.Parameter(torch.Tensor([R]).int(), requires_grad=False)

    def init_var_params(self, N):
        """Initialize variational parameters.

        Args:
        -----
            N (int) : Number of documents
        """

        # means and log std.devs for each document, shape = (n_docs x 2*iv_dim)
        # logging.info("Initializing variational posteior to N(0, %.2f)",
        #             1./self.config['var_p'])
        q_n = np.repeat(
            [0.0, -0.5 * np.log(self.config["hyper"]["var_p"])],
            self.config["hyper"]["K"],
        )
        Q = np.tile(q_n, reps=(1, N)).reshape(N, -1).astype(np.float32)
        self.Q = nn.Parameter(torch.from_numpy(Q), requires_grad=True)

        self.eps = nn.Parameter(
            torch.randn(self.config["hyper"]["R"], N, self.config["hyper"]["K"]),
            requires_grad=False,
        )

        # self.nelbo_d = torch.zeros(N).to(self.device)

    def to_device(self, device=torch.device("cpu")):
        """Transfer all model parameters and variables to given device"""

        self.device = device
        self.to(self.device)

    def sample(self):
        """Sample from Normal distribution for the re-parametrization"""
        self.eps.data.normal_()

    def t_penalty(self):
        """Compute penalty term (regularization) for the bases"""

        if self.config["hyper"]["rt"] == "l2":
            t_pen = self.lam_t * torch.sum(torch.pow(self.T, 2))
        else:
            t_pen = self.lam_t * torch.sum(torch.abs(self.T))
        return t_pen

    def compute_kld(self, rng):
        """Compute KL divergence of q_n from p_0 (prior), KL(q_n || p_0),
        for all `n`

        Returns:
            torch.Tensor: KL divergence between `q` and prior
        """

        # This is only when the prior is isotropic with precision lam_w
        # and posterior is diagonal with parametrization -> np.log(std.dev)

        kld_n = (
            (-2.0 * torch.sum(self.Q[rng[0] : rng[1], self.K :], dim=1))
            + (
                self.lam_w
                * torch.sum(torch.exp(2 * self.Q[rng[0] : rng[1], self.K :]), dim=1)
            )
            + (
                torch.sum(
                    self.Q[rng[0] : rng[1], : self.K]
                    * self.Q[rng[0] : rng[1], : self.K],
                    dim=1,
                )
                * self.lam_w
            )
            + (-1.0 * self.K.float() * torch.log(self.lam_w))
            - self.K.float()
        ) / 2.0

        return kld_n.sum()

    def compute_exp_llh(self, data_dict):
        """Compute expected log-likelihood.

        Args:
        -----
            data_dict (dict) : dictionary with indices, counts,
                range and optional labels `Y`
        Returns:
        --------
            torch.Tensor: Expected log-likelihood
        """

        ixs = data_dict["ixs"]
        counts = data_dict["counts"]
        rng = data_dict["rng"]

        aux = (
            torch.exp(self.Q[rng[0] : rng[1], self.K :])
            * self.eps[:, rng[0] : rng[1], :]
        ) + self.Q[rng[0] : rng[1], : self.K]

        lse_term = (aux @ torch.t(self.T)) + torch.t(self.m)
        lse_tmp = torch.t(torch.sum(lse_term, dim=0)) / self.R.float()

        # lse = torch.log(torch.sum(torch.exp(lse_tmp), dim=0))
        lse = torch.logsumexp(lse_tmp, dim=0)
        mtn = (self.T @ torch.t(self.Q[rng[0] : rng[1], : self.K])) + self.m

        exp_llh = ((mtn - lse)[ixs[0, :], ixs[1, :]] * counts).sum()

        # mtx = torch.sparse.FloatTensor(ixs, counts).to_dense()
        # exp_llh = torch.sum((mtx * (mtn - lse)), dim=0)
        # return exp_llh.sum()
        return exp_llh

    def compute_loss(self, data_dict, use_params="all"):
        """Compute loss ( Exp. LLH + KLD(q || prior) ) for the entire data.
            Note that loss already includes KLD.

        Args:
        -----
            data_dict (dict) : dictionary with indices, counts,
                range and labels `Y` doc indices
            use_params (str) : Which parameters to use for
                computing the loss ? (Q or T or all or
                or T_NP (T without penalty))

        Returns:
        --------
            torch.Tensor : Loss (negative evidence lower bound)
            torch.Tensor : KLD (KL divergence)
        """

        loss = torch.Tensor([0]).to(device=self.device, dtype=self.dtype)
        kld = torch.Tensor([0]).to(device=self.device, dtype=self.dtype)

        exp_llh = self.compute_exp_llh(data_dict)

        loss[0] = -1.0 * exp_llh

        if use_params in ("Q", "all"):
            kld = self.compute_kld(data_dict["rng"])
            loss += kld

        if use_params in ("T", "all"):
            t_pen = self.t_penalty()
            loss += t_pen

        if use_params == "T_NP":
            pass

        return loss, kld

    def compute_total_loss_batch_wise(self, dset, n_batches=1, use_params="all"):
        """Compute total loss by accumulating batch wise losses.
            Note that loss already includes KLD.

        Args:
        ----
            dset (SMMDataset) : SMM Dataset object
            use_params (str)  : Q or T or all

        Returns:
        --------
            torch.Tensor : Total loss [negative ELBO, KLD]
        """

        total_loss = torch.Tensor([0, 0]).to(device=self.device, dtype=self.dtype)

        if use_params in ("all", "Q"):
            temp_use = "Q"
        elif use_params == "T":
            # compute loss using only T, but without the t_penalty
            temp_use = "T_NP"
        else:
            temp_use = use_params

        for data_dict in dset.yield_batches(n_batches):

            loss, kld = self.compute_loss(data_dict, use_params=temp_use)
            total_loss[0] += loss.item()
            total_loss[1] += kld.item()

        if use_params in ("all", "T"):
            total_loss[0] += self.t_penalty().item()

        return total_loss

    def update_ivectors(self, opt_w, data_dict):
        """Update parameters of i-vector posterior distributions.
            Note that loss already includes KLD.

        Args:
        ----
            opt_q (torch.optim) : Current optimizer object
            data_dict (dict)    : Data dictionary with counts and indices

        Returns:
        --------
            torch.Tensor (loss, kld) : [Loss before updating i-vectors, KL divergence]
        """

        loss, kld = self.compute_loss(data_dict, use_params="Q")

        opt_w.zero_grad()
        loss.backward()
        opt_w.step()

        return torch.Tensor([loss.data.clone(), kld.data.clone()]).to(
            device=self.device, dtype=self.dtype
        )

    def update_ivectors_batch_wise(self, opt_q, dset, n_batches=1):
        """Update parameters of all i-vector posterior
        distributions batch-wise.
            Note that loss already includes KLD.

        Args:
        ----
            opt_q (torch.optim) : Current optimizer object
            dset (SMMDataset)   : SMMDataset object

        Returns:
        --------
            torch.Tensor (loss) : [Loss before updating i-vectors, KLD]
        """

        loss = torch.Tensor([0, 0]).to(dtype=self.dtype, device=self.device)
        for data_dict in dset.yield_batches(n_batches):
            loss += self.update_ivectors(opt_q, data_dict)
            # torch.cuda.empty_cache()
        return loss

    def orthant_projection(self, opt_t):
        """Apply orthant projection while updating T matrix.
        Only in case of L1 regularization.

        Args:
        -----
            opt_t (torch.optim): Current optimizer object
        """

        diff_pts = self.T.data.nonzero()  # differtiable points

        if diff_pts.nelement() == 0:

            logging.warning(
                "All the co-ordinates in matrix `T` are zeros. \
                            There are no differentialbe points. This can happen \
                            if you have used very strong \
                            regularization `lam_t = %f` %s",
                self.lam_t.item(),
                "Try using a lower value.",
            )

        else:

            i = diff_pts[:, 0]
            j = diff_pts[:, 1]

            self.T.grad[i, j] += self.lam_t * torch.sign(self.T.data[i, j])

        # non-differentiable points
        non_diff_pts = (self.T.data == 0).nonzero()

        if non_diff_pts.nelement() > 0:

            i = non_diff_pts[:, 0]
            j = non_diff_pts[:, 1]

            self.T.grad[i, j] = torch.where(
                torch.abs(self.T.grad[i, j]) <= self.lam_t,
                torch.zeros_like(self.T.grad[i, j]),
                self.T.grad[i, j],
            )

            self.T.grad.data[i, j] = torch.where(
                self.T.grad[i, j] < -self.lam_t,
                self.T.grad[i, j] + self.lam_t,
                self.T.grad[i, j],
            )

            self.T.grad.data[i, j] = torch.where(
                self.T.grad[i, j] > self.lam_t,
                self.T.grad[i, j] - self.lam_t,
                self.T.grad[i, j],
            )

        t_old_sign = torch.sign(self.T.data).to(dtype=torch.int8, device=self.device)

        opt_t.step()  # make step, i.e, update T

        t_new_sign = torch.sign(self.T.data).to(dtype=torch.int8, device=self.device)

        self.T.data = torch.where(
            (t_new_sign * t_old_sign < 0), torch.zeros_like(self.T), self.T.data
        )
        del t_old_sign, t_new_sign

    def update_bases(self, opt_t, data_dict):
        """Update bases matrix"""

        opt_t.zero_grad()  # clear previous gradients

        if self.config["hyper"]["rt"] == "l1":
            # In case of L1 regularization, compute the loss without T_penalty.
            # T penalty will be taken care in orthant wise learning
            loss, _ = self.compute_loss(data_dict, use_params="T_NP")
        else:
            loss, _ = self.compute_loss(data_dict, use_params="T")

        loss.backward()  # get gradients

        if self.config["hyper"]["rt"] == "l1":
            self.orthant_projection(opt_t)
        else:
            opt_t.step()

    def update_bases_batch_wise(self, opt_t, dset, n_batches=1):
        """Update the bases by accumulating the gradients batch wise.

        Args:
        -----
            opt_t (torch.optim): Optimizer object
            data_loader (torch.utils.dataset.DataLoader): Data loader object

        Returns:
        --------
            loss (torch.tensor): Loss before updating the bases
        """

        loss = torch.Tensor([0]).to(dtype=self.dtype, device=self.device)

        opt_t.zero_grad()

        total_batches, _ = dset.get_total_batches(n_batches)
        bix = 0

        for data_dict in dset.yield_batches(n_batches):

            bix += 1

            # Compute loss without T penalty term (T_NP)
            batch_loss, _ = self.compute_loss(data_dict, use_params="T_NP")

            if bix < total_batches:
                # accumulate gradients until penultimate batch
                batch_loss.backward()
                loss += batch_loss.data.clone()

            else:
                # for the last batch, add T penalty
                if self.config["hyper"]["rt"] == "l2":
                    t_pen = self.t_penalty()
                    batch_loss += t_pen
                    loss += batch_loss.data.clone()
                    batch_loss.backward()  # obtain gradients
                    opt_t.step()  # update T

                else:
                    loss += batch_loss.data.clone() + self.t_penalty().data
                    batch_loss.backward()  # obtain gradients
                    # T penalty is taken care in orthant projection
                    self.orthant_projection(opt_t)

        return loss

    def train_me(self, dset, optims, n_batches=1):
        """Train the model, given the data and optimizers.

        Args:
        -----
            dset (SMMDataset) : Dataset object
            optims (dict)     : Dictionary with optimizers
            n_batches (int)   : Number of batches

        Returns:
        --------
            numpy.ndarray : Loss over iterations
        """

        # Total loss over iterations
        loss_iters = torch.zeros(2, 2).to(dtype=self.dtype, device=self.device)

        stime = time()
        for i in range(self.config["trn_done"], self.config["trn_iters"]):

            stime = time()

            # We return prev_loss i.e., loss before updating i-vector
            # posterior params - just for computational convenience
            prev_loss = torch.zeros([2, 2]).to(dtype=self.dtype, device=self.device)

            if optims["Q"]:
                prev_loss[0, :] = self.update_ivectors_batch_wise(
                    optims["Q"], dset, n_batches
                )

            prev_loss[0, 0] += self.t_penalty().item()
            # loss_iters = torch.cat((loss_iters, prev_loss), dim=1)
            # torch.cuda.empty_cache()

            prev_loss[1, 0] = self.update_bases_batch_wise(optims["T"], dset, n_batches)
            prev_loss[1, 1] = self.compute_kld([0, self.Q.size()[0]]).data
            prev_loss[1, 0] += prev_loss[1, 1]
            # torch.cuda.empty_cache()

            # if np.isnan(prev_loss[1, 0].detach().cpu().numpy()
            logging.info(
                "Iter: %4d/%4d %s: %.1f %s: %.1f %s: %.2f",
                i + 1,
                self.config["trn_iters"],
                "ELBO",
                -prev_loss[1, 0].detach().cpu().numpy(),
                "KLD",
                prev_loss[1, 1].detach().cpu().numpy(),
                "Time per iter",
                (time() - stime),
            )

            loss_iters = torch.cat((loss_iters, prev_loss.data.clone()))

            if (i + 1) % self.config["save"] == 0:
                self.save_params(i + 1)

            # sample from std. Normal for re-parametrization trick
            self.sample()

        logging.info("Training done.")
        self.config["trn_done"] = self.config["trn_iters"]

        # -- After training, compute total loss

        # Turn off gradient computation
        self.T.requires_grad_(False)
        self.Q.requires_grad_(False)
        self.m.requires_grad_(False)

        loss, kld = self.compute_total_loss_batch_wise(
            dset, n_batches, use_params="all"
        )
        loss_iters = loss_iters.cpu().numpy()
        loss_iters = np.concatenate(
            (
                loss_iters[2:, :],
                np.asarray([loss.cpu().item(), kld.cpu().item()]).reshape(1, -1),
            )
        )

        torch.cuda.empty_cache()

        return loss_iters

    def extract_ivector_posteriors(self, dset, opt_q, sbase, n_batches=1):
        """Extract posterior distributions of i-vectors for the
        given documents, using existing model parameters.

        Args:
        -----
            dset (SMMDataset) : SMMDataset object
            opt_q (torch.optim) : Optimizer for posterior
                                  (variational) distributions
            sbase (str) : Basename for saving the extracted
                          posterior distributions
            n_batches (int) : Number of batches

        Returns:
        --------
            numpy.ndarray: Loss over iterations
        """

        # loss_iters = torch.Tensor([]).to(dtype=self.dtype, device=self.device)

        loss_iters = torch.zeros((self.config["xtr"] - self.config["xtr_done"], 2)).to(
            dtype=self.dtype, device=self.device
        )

        self.Q.requires_grad_(True)
        self.sample()

        for b_no, data_dict in enumerate(dset.yield_batches(n_batches, shuffle=False)):

            rng = data_dict["rng"]

            for i in range(self.config["xtr_done"], self.config["xtr"]):

                stime = time()

                if opt_q == "nat":
                    loss = self.update_ivectors_ng(data_dict)
                else:
                    loss = self.update_ivectors(opt_q, data_dict)
                loss_iters[i - self.config["xtr_done"], :] += loss.clone().detach()

                if (i + 1) % self.config["nth"] == 0 or (i + 1 == self.config["xtr"]):
                    ivec_file = os.path.join(
                        self.config["tmp_dir"], f"ivecs_{sbase}_{i+1}_b_{b_no}.npy"
                    )
                    np.save(
                        ivec_file, self.Q.detach().cpu().numpy()[rng[0] : rng[1], :].T
                    )

                # logging.info(
                #     "Batch: %4d Iter %4d/%4d %s: %.1f %s: %.1f %s: %.2f s",
                #     b_no + 1,
                #     i + 1,
                #     self.config["xtr"],
                #     "ELBO",
                #     -loss[0].detach().cpu().numpy(),
                #     "KLD",
                #     loss[1].detach().cpu().numpy(),
                #     "Time per batch",
                #     (time() - stime),
                # )
                self.sample()

        self.Q.requires_grad_(False)

        loss, kld = self.compute_total_loss_batch_wise(dset, n_batches, use_params="Q")

        loss_iters = loss_iters.cpu().numpy()
        loss_iters = np.concatenate(
            (
                loss_iters,
                np.asarray([loss.cpu().item(), kld.cpu().item()]).reshape(1, -1),
            )
        )

        torch.cuda.empty_cache()
        return loss_iters

    def save_params(self, iter_num):
        """Save model parameters to h5 file"""

        sfx = str(iter_num) + ".h5"
        self.config["trn_done"] = iter_num
        self.config["latest_trn_model"] = os.path.join(
            self.config["exp_dir"], f"model_T{sfx}"
        )

        h5f = h5py.File(self.config["latest_trn_model"], "w")
        params = h5f.create_group("params")
        ivecs = h5f.create_group("ivecs")

        ivecs.create_dataset("Q", data=self.Q.data.cpu().numpy())

        params.create_dataset("T", data=self.T.data.cpu().numpy())
        params.create_dataset("m", data=self.m.data.cpu().numpy())

        h5f.close()

        json.dump(
            self.config, open(self.config["cfg_file"], "w"), indent=2, sort_keys=True
        )

        logging.info("Model parameters saved: %s", self.config["latest_trn_model"])


class MBaySMM(nn.Module):
    """Multilingual Bayesian Subspace Multinomial Model"""

    def __init__(self, lang2vocab, lang2ubm, hyper, cuda: bool, seed: int = 0):
        """Initialize the model.

        Args:
        -----
            lang2vocab (dict): Language to vocab size mapping
            lang2ubm (dict): Language to prior log-unigram distribution
            hyper (dict): Hyper parameters
            cuda (bool): Use CUDA ?
            seed (int): Random seed (default: 0)
        """

        super(MBaySMM, self).__init__()

        self.device = torch.device("cuda" if cuda else "cpu")

        self.lang2vocab = lang2vocab
        self.lang2ubm = lang2ubm
        self.hyper = hyper

        self.lam_t = nn.Parameter(torch.Tensor([self.hyper["lt"]]), requires_grad=False)

        self.__init_params()

    def __init_params(self):
        """Initialize model parameters"""

        self.T = nn.ParameterDict()
        self.m = nn.ParameterDict()

        for lid, vocab_size in self.lang2vocab.items():
            self.T[lid] = nn.Parameter(
                torch.zeros(vocab_size, self.hyper["K"]), requires_grad=True
            )
            nn.init.xavier_uniform_(self.T[lid].data)

            self.m[lid] = nn.Parameter(
                torch.from_numpy(self.lang2ubm[lid]).view(-1, 1), requires_grad=True
            )

    def add_params(self, lang2vocab, lang2ubm):
        """Add parameters for new languages.

        Args:
        -----
            lang2vocab (dict): Language to vocab size mapping
            lang2ubm (dict): Language to prior log-unigram distribution
        """

        for lid, vocab_size in lang2vocab.items():
            if lid in self.lang2vocab:
                print(lid, "already in lang2vocab", self.lang2vocab, file=sys.stderr)
                sys.exit()

            self.lang2vocab[lid] = vocab_size

            self.T[lid] = nn.Parameter(
                torch.zeros(vocab_size, self.hyper["K"]), requires_grad=True
            )
            nn.init.xavier_uniform_(self.T[lid].data)
            self.m[lid] = nn.Parameter(
                torch.from_numpy(lang2ubm[lid]).view(-1, 1), requires_grad=True
            )

    def compute_exp_llh(
        self, doc_ixs: torch.LongTensor, data_dict: dict, gau_emb: nn.Module
    ):
        """Compute expected log-likelihood.

        Args:
            doc_ixs (torch.LongTensor): Document or sentence indices
            data_dict (dict): Nested dictionary of the format \
        `{lid_1: {"counts": torch.FloatTensor, "ixs": torch.LongTensor}, lid_2: {...}, ... }`
            gau_emb (nn.Module): Gaussian document (sentence) embeddings

        Returns:
            torch.FloatTensor: Expected log-likelihood of the batch
        """

        nus, log_sigmas = gau_emb(doc_ixs)
        eps = gau_emb.sample(doc_ixs.shape[0])
        aux = (torch.exp(log_sigmas) * eps) + nus

        exp_llh = torch.tensor(0.0).to(device=self.device)

        # n_tokens = torch.tensor(0.0).to(device=self.device)

        # iterate over all the L-way parallel sentences, one language at a time.
        for idx, l_data in data_dict.items():

            ixs = l_data["ixs"]
            counts = l_data["counts"]

            lse_term = (aux @ torch.t(self.T[idx])) + torch.t(self.m[idx])
            lse_tmp = torch.t(torch.sum(lse_term, dim=0)) / gau_emb.R_float

            # lse = torch.log(torch.sum(torch.exp(lse_tmp), dim=0))
            lse = torch.logsumexp(lse_tmp, dim=0)

            mtn = (self.T[idx] @ torch.t(nus)) + self.m[idx]

            exp_llh += ((mtn - lse)[ixs[1, :], ixs[0, :]] * counts).sum()

            # n_tokens += torch.sum(counts)

            # mtx = torch.sparse.FloatTensor(ixs, counts).to_dense()
            # exp_llh = (torch.sum((mtx * (mtn - lse)), dim=0)).sum()

        return exp_llh #, n_tokens

    def t_penalty(self, idx):
        """Compute penalty term (regularization) for the bases"""

        if self.hyper["rt"] == "l2":
            t_pen = self.lam_t * torch.sum(torch.pow(self.T[idx], 2))
        elif self.hyper["rt"] == "l1":
            t_pen = self.lam_t * torch.sum(torch.abs(self.T[idx]))
        return t_pen.squeeze()


class GaussianEmbeddings(nn.Module):
    """Gaussian Embeddings"""

    def __init__(
        self,
        n_samples: int,
        K: int,
        prior_mean: float = 0.0,
        prior_precision: float = 10.0,
        cov_type: str = "diag",
        R: int = 1,
        cuda: bool = False,
    ):

        super(GaussianEmbeddings, self).__init__()

        self.device = torch.device("cuda" if cuda else "cpu")

        self.dtype = torch.float

        self.n_samples = n_samples
        self.K = K
        self.cov_type = cov_type
        self.prior_mean = torch.Tensor([prior_mean])
        self.prior_precision = nn.Parameter(
            torch.FloatTensor([prior_precision]), requires_grad=False
        )
        self.R = R

        self.K_float = nn.Parameter(torch.FloatTensor([self.K]), requires_grad=False)
        self.R_float = nn.Parameter(torch.FloatTensor([self.R]), requires_grad=False)

        #     self.__init_params(n_samples, K)

        # def __init_params(self, n_samples, K):

        # mean of the variational posterior q(z)
        self.mu = nn.Parameter(
            torch.ones(n_samples, K) * self.prior_mean, requires_grad=True
        )

        # log_variance of the variational posterior q(z)
        if self.cov_type == "diag":
            self.log_sigma = nn.Parameter(
                torch.ones(n_samples, K) * torch.log(self.prior_precision) * -0.5,
                requires_grad=True,
            )
        elif self.cov_type == "iso":
            self.log_sigma = nn.Parameter(
                torch.ones(n_samples, 1) * torch.log(self.prior_precision) * -0.5,
                requires_grad=True,
            )
        else:
            # print("cov_type is None", file=sys.stderr)
            self.cov_type = None
            self.log_sigma = None

    def __len__(self):
        return self.n_samples

    def set_R(self, R):
        """Set the value of R (number of MC samples)"""
        self.R = R
        self.R_float.data = torch.tensor(R).to(device=self.device)

    def forward(self, indices):
        if self.cov_type:
            return [
                self.mu[indices, :].to(device=self.device),
                self.log_sigma[indices, :].to(device=self.device),
            ]
        else:
            return self.mu[indices, :].to(device=self.device)

    def sample(self, N):
        """Sample for re-parametrization trick"""

        eps = torch.randn(self.R, N, self.K).to(device=self.device, dtype=self.dtype)
        return eps

    def compute_kld(self, ixs):
        """Compute KL divergence of q_n from p_0 (prior), KL(q_n || p_0),
        for all `n`

        Returns:
            torch.Tensor: KL divergence from variational (q) to prior (p_0)
        """

        # This is only when the prior is isotropic with precision lam_w
        # and posterior is diagonal with parametrization -> np.log(std.dev)

        kld_n = (
            (-2.0 * torch.sum(self.log_sigma[ixs], dim=1))
            + (
                self.prior_precision
                * torch.sum(torch.exp(2 * self.log_sigma[ixs]), dim=1)
            )
            + (
                torch.sum(
                    self.mu[ixs] * self.mu[ixs],
                    dim=1,
                )
                * self.prior_precision
            )
            + (-1.0 * self.K_float * torch.log(self.prior_precision))
            - self.K_float
        ) / 2.0

        return kld_n.sum()



def main():
    """main method"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_gau_emb", action="store_true")
    args = parser.parse_args()

    if args.test_gau_emb:
        gau_emb = GaussianEmbeddings(10, 5, prior_mean=0, prior_precision=10)
        ixs = torch.arange(len(gau_emb))
        print("KLD( N(0, 0.1) || N(0, 1) ):", gau_emb.compute_kld(ixs).detach())

        gau_emb = GaussianEmbeddings(10, 5, prior_mean=0, prior_precision=1)
        ixs = torch.arange(len(gau_emb))
        print("KLD( N(0, 1.0) || N(0, 1) ):", gau_emb.compute_kld(ixs).detach())


if __name__ == "__main__":
    main()
