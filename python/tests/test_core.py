import itertools

import numpy as np
import pytest

from smrc import FeatureMap
from smrc._optim import classifier_probabilities, fit_smrc, varphi
from smrc._phi import mean_and_confidence


def brute_force_varphi(v):
    """max over non-empty subsets C of (sum_{y in C} v_y - 1) / |C|."""
    best, best_set = -np.inf, None
    Y = len(v)
    for k in range(1, Y + 1):
        for C in itertools.combinations(range(Y), k):
            val = (v[list(C)].sum() - 1) / k
            if val > best + 1e-12:
                best, best_set = val, C
    return best, best_set


@pytest.mark.parametrize("Y", [2, 3, 5, 10])
def test_varphi_matches_brute_force(Y):
    rng = np.random.default_rng(Y)
    v = rng.normal(scale=rng.uniform(0.1, 2), size=(200, Y))
    phi, w, psi = varphi(v)
    for i in range(len(v)):
        best, _ = brute_force_varphi(v[i])
        assert phi[i] == pytest.approx(best, abs=1e-10)
        # Subgradient weights are uniform over a maximizing set.
        C = np.flatnonzero(w[i] > 0)
        assert np.allclose(w[i][C], 1 / len(C))
        assert (v[i][C].sum() - 1) / len(C) == pytest.approx(best, abs=1e-10)
        assert psi[i] == pytest.approx(best - (v[i].max() - 1), abs=1e-10)
        assert psi[i] >= -1e-12


def test_varphi_subgradient_inequality():
    """varphi(mu', x) >= varphi(mu, x) + g^T (mu' - mu) for the subgradient g."""
    rng = np.random.default_rng(0)
    Y, D = 4, 6
    phi_map = FeatureMap(Y, [rng.normal(size=(30, D))], [rng.random(30) > 0.3])
    for _ in range(20):
        mu, mu2 = rng.normal(size=Y * D), rng.normal(size=Y * D)
        f1, w, _ = varphi(phi_map.scores(mu))
        f2, _, _ = varphi(phi_map.scores(mu2))
        for i in range(phi_map.n_samples):
            g = phi_map.subset([i]).weighted_sum(w[[i]])
            assert f2[i] >= f1[i] + g @ (mu2 - mu) - 1e-10


def dense_phi_matlab(psi, conf, Y):
    """Feature vectors built as in compute_phi_*.m with kron."""
    I = np.eye(Y)
    out = []
    for i in range(len(psi)):
        rows = []
        for k in range(Y):
            e = I[:, k] if (conf is None or conf[i]) else np.ones(Y)
            rows.append(np.kron(e, psi[i]))
        out.append(np.array(rows))
    return out


def test_feature_map_matches_kron():
    rng = np.random.default_rng(1)
    Y, n = 3, 7
    psi_u, psi_s = rng.normal(size=(n, 4)), rng.normal(size=(n, 5))
    conf = rng.random(n) > 0.5
    fmap = FeatureMap(Y, [psi_u, psi_s], [None, conf])
    dense = [np.hstack([a, b]) for a, b in zip(dense_phi_matlab(psi_u, None, Y),
                                               dense_phi_matlab(psi_s, conf, Y))]
    mu = rng.normal(size=fmap.n_features)
    v = fmap.scores(mu)
    W = rng.random((n, Y))
    expected_sum = sum(W[i] @ dense[i] for i in range(n))
    labels = rng.integers(0, Y, n)
    for i in range(n):
        assert np.allclose(v[i], dense[i] @ mu)
        assert np.allclose(fmap.at_labels(labels)[i], dense[i][labels[i]])
    assert np.allclose(fmap.weighted_sum(W), expected_sum)


def test_mean_and_confidence():
    rng = np.random.default_rng(2)
    Y, n = 3, 50
    fmap = FeatureMap(Y, [rng.normal(size=(n, 4))], [rng.random(n) > 0.5])
    labels = rng.integers(0, Y, n)
    F = fmap.at_labels(labels)
    tau, lam = mean_and_confidence(fmap, labels)
    assert np.allclose(tau, F.mean(0))
    assert np.allclose(lam, F.std(0, ddof=1) / np.sqrt(n))


def test_probabilities_are_distributions():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(100, 6))
    h = classifier_probabilities(v)
    assert np.all(h >= 0)
    assert np.allclose(h.sum(1), 1)
    assert np.all(h.argmax(1) == v.argmax(1))


def test_fit_smrc_separable_problem():
    """With informative features and exact expectations, R(V) is small."""
    rng = np.random.default_rng(4)
    Y, n = 3, 600
    y = rng.integers(0, Y, n)
    psi = np.hstack([np.ones((n, 1)), np.eye(Y)[y] + 0.05 * rng.normal(size=(n, Y))])
    fmap = FeatureMap(Y, [psi], [None])
    tau, lam = mean_and_confidence(fmap, y)
    mu, value = fit_smrc(fmap, tau, lam, max_iter=5000, batch_size=50, step=0.5,
                         random_state=0)
    assert value < 0.1
    assert np.mean(fmap.scores(mu).argmax(1) == y) > 0.95
