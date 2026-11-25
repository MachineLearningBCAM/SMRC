"""Stochastic subgradient methods used by SMRCs.

* :func:`varphi` implements Algorithm 1 of the paper (vectorized over a
  batch of instances): it returns ``varphi(mu, x)``, a subgradient of it, and
  the disagreement ``psi(mu, x)`` of Theorem 4.
* :func:`fit_smrc` solves the learning problem (3.2) with an averaged
  stochastic subgradient method.
* :func:`fit_bound` solves the convex problems (4.6)-(4.7) of Theorem 6 that
  give high-confidence upper and lower bounds for the error probability.
"""

from __future__ import annotations

import numpy as np

from ._phi import FeatureMap


def varphi(v: np.ndarray):
    """Algorithm 1: values of ``varphi`` and its subgradient weights.

    Parameters
    ----------
    v : ndarray of shape (n, |Y|)
        Values ``Phi(x_i, y)^T mu``.

    Returns
    -------
    phi : ndarray of shape (n,)
        ``varphi(mu, x_i) = max_C (sum_{y in C} v[i, y] - 1) / |C|``.
    weights : ndarray of shape (n, |Y|)
        ``weights[i, y] = 1/|C_i|`` if ``y`` belongs to the maximizing set
        ``C_i`` and 0 otherwise, so that ``sum_y weights[i, y] Phi(x_i, y)``
        is a subgradient of ``varphi(., x_i)``.
    psi : ndarray of shape (n,)
        ``psi(mu, x_i) = varphi(mu, x_i) - (max_y v[i, y] - 1)``.
    """
    n, n_classes = v.shape
    order = np.argsort(-v, axis=1, kind="stable")
    v_sorted = np.take_along_axis(v, order, axis=1)
    sizes = np.arange(1, n_classes + 1)
    candidates = (np.cumsum(v_sorted, axis=1) - 1.0) / sizes
    # Greedy rule of Algorithm 1: keep adding labels while the value does
    # not decrease, stop at the first decrease.
    keep = candidates[:, 1:] >= candidates[:, :-1]
    k_star = 1 + np.cumprod(keep, axis=1).sum(axis=1)
    rows = np.arange(n)
    phi = candidates[rows, k_star - 1]
    weights_sorted = (sizes[None, :] <= k_star[:, None]) / k_star[:, None]
    weights = np.empty_like(weights_sorted)
    np.put_along_axis(weights, order, weights_sorted, axis=1)
    psi = phi - (v_sorted[:, 0] - 1.0)
    return phi, weights, psi


def classifier_probabilities(v: np.ndarray) -> np.ndarray:
    """SMRC rule (3.3): ``h(y|x) = (Phi(x, y)^T mu - varphi(mu, x))_+``."""
    phi, _, _ = varphi(v)
    return np.maximum(v - phi[:, None], 0.0)


def _batches(n, batch_size, rng):
    """Infinite generator of mini-batches (one random permutation per epoch)."""
    batch_size = min(batch_size, n)
    while True:
        perm = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield perm[start:start + batch_size]


def _default_step(m, batch_size):
    # The MATLAB implementation uses single-sample subgradients with step
    # 0.1 / m. Averaging B subgradients and taking a step B times larger gives
    # practically the same iterates with B times fewer steps.
    return 0.1 * batch_size / m


def fit_smrc(phi: FeatureMap, tau, lam, max_iter=100000, batch_size=100,
             step=None, random_state=None):
    """Solve problem (3.2) by averaged stochastic subgradient descent.

    Minimizes ``1 - tau^T mu + lam^T |mu| + mean_i varphi(mu, x_i)`` over
    ``mu``, where the expectation with respect to the instances' marginal is
    approximated by the instances in ``phi``.

    Parameters
    ----------
    phi : FeatureMap
        Feature mapping evaluated at the instances used for subgradients.
    tau, lam : ndarray of shape (m,)
        Mean and confidence vectors.
    max_iter : int
        Number of subgradient steps.
    batch_size : int
        Number of instances per stochastic subgradient (``B`` in the MATLAB
        code).
    step : float, optional
        Step size. Defaults to ``0.1 * batch_size / m``, which corresponds to
        the MATLAB implementation (``B = 1``, step ``0.1 / m``, and ``10^7``
        iterations) when ``max_iter * batch_size = 10^7``.
    random_state : int, Generator or None

    Returns
    -------
    mu : ndarray of shape (m,)
        Averaged iterate (SMRC parameters).
    value : float
        Objective value at ``mu`` (minimax risk ``R(V)``).
    """
    rng = np.random.default_rng(random_state)
    m = phi.n_features
    step = _default_step(m, min(batch_size, phi.n_samples)) if step is None else step
    mu = np.zeros(m)
    mu_avg = np.zeros(m)
    batches = _batches(phi.n_samples, batch_size, rng)
    for t in range(1, max_iter + 1):
        idx = next(batches)
        sub = phi.subset(idx)
        _, w, _ = varphi(sub.scores(mu))
        grad = sub.weighted_sum(w) / len(idx)
        mu -= step * (-tau + lam * np.sign(mu) + grad)
        mu_avg += (mu - mu_avg) / (t + 1)
    return mu_avg, smrc_objective(phi, tau, lam, mu_avg)[0]


def smrc_objective(phi: FeatureMap, tau, lam, mu):
    """Objective of (3.2) and mean disagreement ``E{psi(mu, x)}``."""
    vphi, _, psi = varphi(phi.scores(mu))
    value = 1.0 - tau @ mu + lam @ np.abs(mu) + vphi.mean()
    return float(value), float(psi.mean())


def fit_bound(phi: FeatureMap, h: np.ndarray, tau, lam, kind="upper",
              max_iter=100000, batch_size=100, step=None, random_state=None):
    """High-confidence bound of Theorem 6 for the error of a rule ``h``.

    ``kind="upper"`` solves (4.6):
        min_mu 1 - tau^T mu + lam^T |mu| + E{max_y Phi(x, y)^T mu - h(y|x)}
    ``kind="lower"`` solves (4.7):
        max_mu 1 - tau^T mu - lam^T |mu| + E{min_y Phi(x, y)^T mu - h(y|x)}

    Parameters
    ----------
    phi : FeatureMap
        Feature mapping evaluated at unlabeled instances.
    h : ndarray of shape (n, |Y|)
        Classification rule evaluated at the same instances
        (``h[i, y] = h(y|x_i)``).
    tau, lam : ndarray of shape (m,)
        Mean and confidence vectors satisfying the condition in Theorem 6.
    """
    if kind not in ("upper", "lower"):
        raise ValueError("kind must be 'upper' or 'lower'")
    sgn = 1.0 if kind == "upper" else -1.0
    rng = np.random.default_rng(random_state)
    m = phi.n_features
    step = _default_step(m, min(batch_size, phi.n_samples)) if step is None else step
    mu = np.zeros(m)
    mu_avg = np.zeros(m)
    batches = _batches(phi.n_samples, batch_size, rng)
    for t in range(1, max_iter + 1):
        idx = next(batches)
        sub = phi.subset(idx)
        v = sub.scores(mu) - h[idx]
        y = v.argmax(axis=1) if kind == "upper" else v.argmin(axis=1)
        w = np.zeros_like(v)
        w[np.arange(len(idx)), y] = 1.0
        grad = sub.weighted_sum(w) / len(idx)
        # Minimize sgn * (objective without the constant): for the lower
        # bound this is  tau^T mu + lam^T |mu| - E{min_y ...}.
        mu -= step * (-sgn * tau + lam * np.sign(mu) + sgn * grad)
        mu_avg += (mu - mu_avg) / (t + 1)
    v = phi.scores(mu_avg) - h
    if kind == "upper":
        return float(1.0 - tau @ mu_avg + lam @ np.abs(mu_avg) + v.max(axis=1).mean())
    return float(1.0 - tau @ mu_avg - lam @ np.abs(mu_avg) + v.min(axis=1).mean())
