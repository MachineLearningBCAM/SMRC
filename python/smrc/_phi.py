"""Compact representation of the feature mappings used by SMRCs.

The paper uses feature mappings of the form (Section 3.1)

    Phi_uns(x, y) = e_y (x) Psi_uns(x)
    Phi_sup(x, y) = e_y (x) Psi_sup(x)   if x is confidently classified
                    1   (x) Psi_sup(x)   otherwise

where (x) is the Kronecker product.  Building these vectors explicitly would
need ``n * |Y| * m`` numbers, so instead we store, for each block, the
instance features ``Psi_b(x)`` (an ``n x D_b`` matrix) together with an
optional boolean mask telling which instances use the one-hot label encoding.
All the quantities needed by SMRCs (inner products ``Phi(x, y)^T mu`` and
weighted sums of ``Phi(x, y)``) are computed directly from this
representation.

The ordering of the components of ``Phi`` (and hence of ``mu``, ``tau`` and
``lambda``) is the same as in the MATLAB code: blocks are concatenated, and
within each block the ``D_b`` features of label ``k`` occupy positions
``k * D_b, ..., (k + 1) * D_b - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FeatureMap:
    """Feature mapping ``Phi(x, y)`` evaluated at ``n`` instances.

    Parameters
    ----------
    n_classes : int
        Number of classes ``|Y|``.
    blocks : list of ndarray of shape (n, D_b)
        Instance features ``Psi_b(x_i)`` of each block.
    confident : list of (ndarray of shape (n,) of bool or None)
        For each block, ``None`` if the block always uses the one-hot label
        encoding ``e_y (x) Psi_b(x)``; otherwise a boolean mask whose entry
        ``i`` is ``True`` if ``x_i`` uses ``e_y (x) Psi_b(x)`` and ``False``
        if it uses ``1 (x) Psi_b(x)``.
    """

    n_classes: int
    blocks: List[np.ndarray] = field(default_factory=list)
    confident: List[Optional[np.ndarray]] = field(default_factory=list)

    def __post_init__(self):
        self.blocks = [np.asarray(b, dtype=float) for b in self.blocks]
        if len(self.confident) != len(self.blocks):
            raise ValueError("`confident` must have one entry per block")
        n = {b.shape[0] for b in self.blocks}
        if len(n) > 1:
            raise ValueError("all blocks must have the same number of rows")

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.blocks[0].shape[0]

    @property
    def block_sizes(self) -> List[int]:
        """Length ``|Y| * D_b`` of each block of ``Phi``."""
        return [self.n_classes * b.shape[1] for b in self.blocks]

    @property
    def n_features(self) -> int:
        """Dimensionality ``m`` of ``Phi``."""
        return int(sum(self.block_sizes))

    def _split(self, mu: np.ndarray):
        """Split a vector of length ``m`` into ``(|Y|, D_b)`` matrices."""
        out, start = [], 0
        for b in self.blocks:
            size = self.n_classes * b.shape[1]
            out.append(mu[start:start + size].reshape(self.n_classes, b.shape[1]))
            start += size
        return out

    def subset(self, idx) -> "FeatureMap":
        """Feature mapping restricted to the instances in ``idx``."""
        return FeatureMap(
            self.n_classes,
            [b[idx] for b in self.blocks],
            [None if c is None else c[idx] for c in self.confident],
        )

    def concatenate(self, other: "FeatureMap") -> "FeatureMap":
        """Stack the blocks of two feature mappings (``[Phi_1; Phi_2]``)."""
        if other.n_classes != self.n_classes:
            raise ValueError("number of classes does not match")
        return FeatureMap(self.n_classes, self.blocks + other.blocks,
                          self.confident + other.confident)

    def squared(self) -> "FeatureMap":
        """Feature mapping with component-wise squared values.

        Since the label encodings only take values in {0, 1}, the squared
        feature mapping is obtained by squaring the instance features.
        """
        return FeatureMap(self.n_classes, [b ** 2 for b in self.blocks],
                          list(self.confident))

    # ------------------------------------------------------------------
    def scores(self, mu: np.ndarray) -> np.ndarray:
        """Return the ``(n, |Y|)`` matrix with entries ``Phi(x_i, y)^T mu``."""
        v = np.zeros((self.n_samples, self.n_classes))
        for psi, conf, M in zip(self.blocks, self.confident, self._split(mu)):
            s = psi @ M.T
            if conf is not None:
                not_conf = ~conf
                # 1 (x) Psi(x): the same value for every label
                s[not_conf] = s[not_conf].sum(axis=1, keepdims=True)
            v += s
        return v

    def weighted_sum(self, weights: np.ndarray) -> np.ndarray:
        """Return ``sum_i sum_y weights[i, y] * Phi(x_i, y)`` (length ``m``)."""
        out = []
        for psi, conf in zip(self.blocks, self.confident):
            w = weights
            if conf is not None:
                w = weights.copy()
                not_conf = ~conf
                w[not_conf] = weights[not_conf].sum(axis=1, keepdims=True)
            out.append((w.T @ psi).ravel())
        return np.concatenate(out)

    def at_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return the dense ``(n, m)`` matrix with rows ``Phi(x_i, labels[i])``."""
        labels = np.asarray(labels)
        n, Y = self.n_samples, self.n_classes
        out = []
        for psi, conf in zip(self.blocks, self.confident):
            A = one_hot(labels, Y)  # label encoding of each row
            if conf is not None:
                A[~conf] = 1.0
            out.append((A[:, :, None] * psi[:, None, :]).reshape(n, -1))
        return np.hstack(out)


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encoding of integer labels in ``{0, ..., n_classes - 1}``."""
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def mean_and_confidence(phi: FeatureMap, labels: np.ndarray):
    """Mean and confidence vectors of ``Phi(x_i, labels[i])``, i = 1..n.

    Returns ``tau`` (sample mean) and ``lambda`` (sample standard deviation
    divided by ``sqrt(n)``), as in equations (3.4)-(3.5) and (3.7)-(3.8) of
    the paper.
    """
    n = phi.n_samples
    w = one_hot(labels, phi.n_classes)
    mean = phi.weighted_sum(w) / n
    second = phi.squared().weighted_sum(w) / n
    var = np.maximum(second - mean ** 2, 0.0) * n / max(n - 1, 1)
    return mean, np.sqrt(var) / np.sqrt(n)
