import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import smrc
from smrc.bounds import betting_confidence_interval, confidence_vectors


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=8,
                               n_classes=3, n_clusters_per_class=1, class_sep=1.5,
                               random_state=0)
    y = np.array(["a", "b", "c"])[y]  # non-integer labels
    lab, unl, test = smrc.partition(y, 100, 1000, 400, random_state=0)
    return X, y, lab, unl, test


def make_smrc(**kw):
    params = dict(
        unsupervised_features=smrc.PCAKMeansFeatures(5, 5, random_state=0),
        supervised_features=smrc.NeuralNetworkFeatures(10, random_state=0),
        supervised_classifier=smrc.ConfidentClassifier(
            LogisticRegression(max_iter=1000), random_state=0),
        max_iter=3000, batch_size=50, random_state=0)
    params.update(kw)
    return smrc.SMRC(**params)


def test_fit_predict(data):
    X, y, lab, unl, test = data
    clf = make_smrc().fit(X[lab], y[lab], X[unl])
    pred = clf.predict(X[test])
    assert set(pred) <= set(clf.classes_)
    assert np.mean(pred != y[test]) < 0.3
    p = clf.predict_proba(X[test])
    assert p.shape == (len(test), 3)
    assert np.allclose(p.sum(1), 1) and np.all(p >= 0)
    assert np.all(clf.classes_[p.argmax(1)] == pred)
    assert set(clf.predict_randomized(X[test], random_state=0)) <= set(clf.classes_)
    assert 0 <= clf.minimax_risk_ <= 1
    m = 3 * (1 + 5 + 5) + 3 * 10
    assert clf.mu_.shape == clf.tau_.shape == clf.lambda_.shape == (m,)


@pytest.mark.parametrize("kw", [dict(split_unlabeled=True),
                                dict(supervised_classifier=smrc.ConfidentClassifier(
                                    LogisticRegression(max_iter=1000), threshold="cv",
                                    cv=5, random_state=0)),
                                dict(supervised_features="none"),
                                dict(unsupervised_features="none")])
def test_variants(data, kw):
    X, y, lab, unl, test = data
    clf = make_smrc(**kw).fit(X[lab], y[lab], X[unl])
    assert np.mean(clf.predict(X[test]) != y[test]) < 0.4


def test_error_bounds(data):
    X, y, lab, unl, test = data
    clf = make_smrc().fit(X[lab], y[lab], X[unl])
    lower, upper = clf.error_bounds(delta=0.05, max_iter=2000)
    assert lower <= upper
    assert -0.5 < lower < 1.5 and -0.5 < upper < 1.5


def test_betting_interval_coverage():
    rng = np.random.default_rng(0)
    misses = 0
    for _ in range(100):
        x = rng.beta(2, 5, size=200)
        c, r = betting_confidence_interval(x, alpha=0.1)
        misses += abs(c - 2 / 7) > r + 1e-2  # grid resolution 1/99
    assert misses <= 20
    # Width shrinks with more samples.
    _, r1 = betting_confidence_interval(rng.random(100), 0.05)
    _, r2 = betting_confidence_interval(rng.random(2000), 0.05)
    assert r2 < r1


def test_confidence_vectors_constant_columns():
    F = np.column_stack([np.ones(50), np.zeros(50), np.linspace(0, 1, 50)])
    tau, lam = confidence_vectors(F, delta=0.05)
    assert tau[0] == 1 and lam[0] == 0 and tau[1] == 0 and lam[1] == 0
    assert lam[2] > 0


def test_confident_classifier_thresholds(data):
    X, y, lab, unl, test = data
    est = LogisticRegression(max_iter=1000)
    _, conf = smrc.ConfidentClassifier(est).fit(X[lab], y[lab]).predict_confident(X[unl])
    assert conf.all()
    cc = smrc.ConfidentClassifier(est, threshold="cv", cv=5, random_state=0).fit(X[lab], y[lab])
    _, conf = cc.predict_confident(X[unl])
    assert 0 < conf.mean() < 1 and 0 <= cc.cv_error_ <= 1
    _, conf = smrc.ConfidentClassifier(est, threshold=2.0).fit(X[lab], y[lab]).predict_confident(X[unl])
    assert not conf.any()
