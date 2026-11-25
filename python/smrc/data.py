"""Data utilities for the experiments."""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def load_mat(path):
    """Load a dataset stored as in the MATLAB code.

    The ``.mat`` file must contain a matrix ``x`` of size ``d x n`` (one
    instance per column) and a vector ``y`` with ``n`` labels.

    Returns
    -------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    """
    d = loadmat(path)
    X = np.asarray(d["x"], dtype=float).T
    y = np.asarray(d["y"]).ravel()
    if X.shape[0] != y.shape[0]:  # instances stored as rows
        X = X.T
    return X, y


def partition(y, n_lab, n_unl, n_test, random_state=None):
    """Random stratified partition into labeled, unlabeled, and test indices.

    Mirrors the partitions in the paper's experiments: ``n_lab`` labeled
    samples, ``n_unl`` unlabeled samples, and ``n_test`` test samples are
    drawn without replacement, each set stratified by class.
    """
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(y))
    lab, rest = train_test_split(idx, train_size=n_lab, stratify=y, random_state=rng)
    unl, rest = train_test_split(rest, train_size=n_unl, stratify=y[rest], random_state=rng)
    if n_test >= len(rest):
        test = rest
    else:
        test, _ = train_test_split(rest, train_size=n_test, stratify=y[rest], random_state=rng)
    return lab, unl, test
