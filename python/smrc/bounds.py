"""Computable performance guarantees (Section 4.2 of the paper).

* :func:`betting_confidence_interval` obtains confidence intervals for the
  mean of bounded random variables using the betting approach of
  Waudby-Smith and Ramdas (2024), as done for Figure 1 of the paper (this is
  a port of ``conf_int.m``).
* :func:`confidence_vectors` applies it to every component of a feature
  mapping to obtain vectors ``tau_delta`` and ``lambda_delta`` with
  ``P{|E{Phi} - tau_delta| < lambda_delta} >= 1 - delta``.
* :func:`error_bounds` computes the upper and lower bounds of Theorem 6.
"""

from __future__ import annotations

import numpy as np

from ._optim import fit_bound
from ._phi import FeatureMap, one_hot


def betting_confidence_interval(x, alpha=0.05, grid_size=100):
    """Confidence interval for the mean of samples ``x`` in ``[0, 1]``.

    Hedged-capital betting confidence interval (Waudby-Smith & Ramdas,
    "Estimating means of bounded random variables by betting", JRSS-B 2024).
    The interval is computed over a grid of ``grid_size`` candidate means.

    Returns
    -------
    center, radius : float
        The interval is ``[center - radius, center + radius]``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (i + 1)
    var = (0.25 + np.cumsum((x - mu) ** 2)) / (i + 1)
    var_prev = np.concatenate([[0.25], var[:-1]])
    lam_dot = np.sqrt(2 * np.log(2 / alpha) / (n * var_prev))

    grid = np.linspace(0, 1, grid_size)[:, None]
    with np.errstate(divide="ignore"):
        lam1 = np.minimum(lam_dot[None, :], 0.5 / grid)
        lam2 = np.minimum(lam_dot[None, :], 0.5 / (1 - grid))
    diff = x[None, :] - grid
    # Capital processes (all factors are >= 1/2 so logs are well defined).
    log_k1 = np.cumsum(np.log1p(lam1 * diff), axis=1)
    log_k2 = np.cumsum(np.log1p(-lam2 * diff), axis=1)
    log_capital = np.log(0.5) + np.maximum(log_k1, log_k2)

    lo, hi = 0, grid_size - 1
    for t in range(n):
        idx = np.flatnonzero(log_capital[:, t] < np.log(1 / alpha))
        if idx.size:
            hi = min(hi, idx.max())
            lo = max(lo, idx.min())
    g = grid.ravel()
    center = (g[hi] + g[lo]) / 2
    radius = max(g[hi] - center, g[lo] - center)
    return float(center), float(radius)


def confidence_vectors(F, delta=0.05, grid_size=100):
    """Component-wise confidence intervals with a union bound.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Samples ``Phi(x_i, y_i)``.
    delta : float
        Overall confidence level (each component uses ``delta / m``).

    Returns
    -------
    tau, lam : ndarray of shape (m,)
    """
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    tau, lam = np.empty(m), np.empty(m)
    lo, hi = F.min(axis=0), F.max(axis=0)
    for j in range(m):
        if hi[j] == lo[j]:
            tau[j], lam[j] = lo[j], 0.0
            continue
        c, r = betting_confidence_interval((F[:, j] - lo[j]) / (hi[j] - lo[j]),
                                           delta / m, grid_size)
        tau[j] = (hi[j] - lo[j]) * c + lo[j]
        lam[j] = (hi[j] - lo[j]) * r
    return tau, lam


def error_bounds(phi_lab: FeatureMap, y_lab, phi_unl: FeatureMap, h_unl,
                 delta=0.05, ensure_feasible=True, y_unl_pred=None,
                 max_iter=100000, batch_size=100, step=None, random_state=None):
    """Upper and lower bounds of Theorem 6 for the error of a rule ``h``.

    Parameters
    ----------
    phi_lab : FeatureMap
        Feature mapping evaluated at labeled instances. It must not depend on
        these labeled samples (e.g., ``Phi_uns``, learned from unlabeled
        instances), so that ``Phi(x_i, y_i)`` are i.i.d.
    y_lab : ndarray of int
        Labels (encoded as ``0, ..., |Y|-1``).
    phi_unl : FeatureMap
        The same feature mapping evaluated at unlabeled instances.
    h_unl : ndarray of shape (n_unl, |Y|)
        Classification rule evaluated at the unlabeled instances.
    delta : float
        The bounds hold with probability at least ``1 - delta``.
    ensure_feasible : bool
        As in the paper's experiments, enlarge ``lambda_delta`` so that the
        problems are bounded (uses ``y_unl_pred``, labels assigned to the
        unlabeled instances, by default ``argmax h``).

    Returns
    -------
    lower, upper : float
    """
    n_classes = phi_lab.n_classes
    F = phi_lab.at_labels(np.asarray(y_lab))  # rows Phi(x_i, y_i)
    tau, lam = confidence_vectors(F, delta)
    if ensure_feasible:
        pred = h_unl.argmax(axis=1) if y_unl_pred is None else y_unl_pred
        tau_rec = phi_unl.weighted_sum(one_hot(pred, n_classes)) / phi_unl.n_samples
        lam = np.maximum(lam, np.abs(tau - tau_rec))
    kw = dict(max_iter=max_iter, batch_size=batch_size, step=step,
              random_state=random_state)
    upper = fit_bound(phi_unl, h_unl, tau, lam, kind="upper", **kw)
    lower = fit_bound(phi_unl, h_unl, tau, lam, kind="lower", **kw)
    return lower, upper
