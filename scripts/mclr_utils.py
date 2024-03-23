#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Utilities for training and predicting MCLR models
"""

import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import log_loss, accuracy_score

logger = logging.getLogger(__name__)


def save_model(model, out_file):
    """ Save model """

    logger.info("Saving model %s", out_file)
    torch.save(model.state_dict(), out_file)


def load_model(model, model_file):
    """ Load model """

    model.load_state_dict(torch.load(model_file))
    return model


def train(model, optim, X, Y, trn_iters):
    """ Train the model """

    X = X.to(device=model.device)
    Y = Y.to(device=model.device)

    # train_losses = []
    # best_loss = torch.Tensor([9999999]).to(model.device)

    for i in range(trn_iters):

        optim.zero_grad()
        logits = model(X)
        xen = model.loss(logits, Y)
        xen.backward()
        optim.step()

        # train_losses.append(xen.detach().item())

        if best_loss > xen.detach().item():
            best_loss = xen.detach().item()

        else:
            break

    model.compute_grads(False)
    return model


def train_and_validate(model, optim, x_train, y_train, x_dev, y_dev, out_sfx,
                       trn_iters, val_iters=10, dev_metric="acc"):
    """ Train the model and validate on dev set after every `val_iters` """

    logger = logging.getLogger()

    # [trn_acc, trn_xen, dev_acc, dev_xen]
    scores = np.zeros(shape=(trn_iters, 4), dtype=np.float32)
    vix = 2 if dev_metric == "acc" else 3

    best_dev_score = 0.001 if dev_metric == "acc" else 999.
    best_model_file = ""

    train_dset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dset, pin_memory=True, batch_size=256, shuffle=True)

    dev_dset = TensorDataset(x_dev, y_dev)
    dev_loader = DataLoader(dev_dset, pin_memory=True, batch_size=256, shuffle=False)

    for i in range(trn_iters):

        for inputs, targets in train_loader:

            inputs = inputs.to(device=model.device)
            targets = targets.to(device=model.device)

            optim.zero_grad()
            logits = model.forward(inputs)
            xen = model.loss(logits, targets)
            xen.backward()
            optim.step()

        if (i+1) % val_iters == 0:

            scores[i, :2] = evaluate(model, train_loader)
            scores[i, 2:] = evaluate(model, dev_loader)

            if dev_metric == "acc":
                if scores[i, vix] > best_dev_score:
                    best_dev_score = scores[i, 2]
                    best_model_file = out_sfx + f"_{i+1}.pt"
                    torch.save(model, best_model_file)
            else:
                if scores[i, vix] < best_dev_score:
                    best_dev_score = scores[i, 3]
                    best_model_file = out_sfx + f"_{i+1}.pt"
                    torch.save(model, best_model_file)

            # turn on gradient computations
            model.compute_grads(True)

            logger.info("Iter {:4d}/{:4d} Loss: {:.2f} "
                        "Train acc: {:.2f} xen: {:.4f} | Dev acc: {:.2f} xen {:.4f}".format(
                            i+1, trn_iters, xen.detach().cpu().numpy().item(),
                            *scores[i, :]))

    return best_model_file, scores


def predict(model, X):
    """ Predict post. prob of classes given the features """

    with torch.no_grad():
        probs = model.predict_proba(X)
    return torch.argmax(probs, dim=1).cpu().numpy()


def evaluate(model, data_loader):

    all_true_labels = []
    all_pred_labels = []
    all_pred_probas = []
    with torch.no_grad():

        for inputs, targets in data_loader:

            inputs = inputs.to(device=model.device)
            targets = targets.to(device=model.device)

            pred_proba = model.predict_proba(inputs)
            pred_labels = torch.argmax(pred_proba, dim=1)

            all_true_labels.append(targets.cpu().numpy())
            all_pred_labels.append(pred_labels.cpu().numpy())
            all_pred_probas.append(pred_proba.cpu().numpy())

    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    all_pred_probas = np.concatenate(all_pred_probas)

    return [
        accuracy_score(all_true_labels, all_pred_labels),
        log_loss(all_true_labels, all_pred_probas)
    ]
