"""Semi-supervised minimax risk classifiers (SMRCs)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._optim import classifier_probabilities, fit_smrc, smrc_objective
from ._phi import FeatureMap, mean_and_confidence, one_hot
from .bounds import error_bounds
from .features import ConfidentClassifier, NeuralNetworkFeatures, PCAKMeansFeatures


class SMRC(ClassifierMixin, BaseEstimator):
    """Semi-supervised minimax risk classifier.

    Implements Algorithm 2 of the paper:

    1. ``Psi_sup`` and ``h_sup`` are learned from the labeled samples, and
       ``Psi_uns`` is learned from unlabeled instances.
    2. ``tau_uns``, ``lambda_uns`` are estimated with the labeled samples and
       ``tau_sup``, ``lambda_sup`` with the unlabeled instances labeled by
       ``h_sup``.
    3. ``lambda`` is enlarged if needed so that the uncertainty set is not
       empty.
    4. Problem (3.2) is solved with stochastic subgradients computed at
       unlabeled instances, giving the parameters ``mu*`` and the minimax risk
       ``R(V)``.

    Parameters
    ----------
    unsupervised_features : transformer or None, default=None
        Unsupervised feature learner ``A_uns^feat`` (``fit(X)`` /
        ``transform(X)``). Defaults to :class:`PCAKMeansFeatures` with 50
        principal components and 50 clusters. Use ``"none"`` to skip
        ``Phi_uns``.
    supervised_features : transformer or None, default=None
        Supervised feature learner ``A_sup^feat`` (``fit(X, y)`` /
        ``transform(X)``). Defaults to :class:`NeuralNetworkFeatures` with a
        hidden layer of size 200. Use ``"none"`` to skip ``Phi_sup``.
    supervised_classifier : ConfidentClassifier or None, default=None
        Supervised classifier ``A_sup^clas`` together with the confidence
        threshold. Defaults to :class:`ConfidentClassifier` with 200 boosted
        trees.
    split_unlabeled : bool, default=False
        If ``False`` (as in Algorithm 2), all unlabeled instances are used both
        to learn ``Psi_uns`` and to compute stochastic subgradients. If
        ``True``, the unlabeled instances are randomly split in two halves:
        the first one is used to learn ``Psi_uns`` and the second one to
        compute stochastic subgradients.
    lambda_scale : float, default=1.0
        Factor multiplying the confidence vector before ensuring that the
        uncertainty set is not empty.
    max_iter : int, default=100000
        Number of stochastic subgradient steps.
    batch_size : int, default=100
        Number of instances per stochastic subgradient.
    step : float or None, default=None
        Step size. If ``None``, ``0.1 * batch_size / m`` where ``m`` is the
        number of features; with the default ``max_iter`` and ``batch_size``
        this matches the optimization in the MATLAB code (``10^7``
        single-sample steps of size ``0.1 / m``).
    random_state : int or None, default=None

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    mu_ : ndarray of shape (m,)
        SMRC parameters ``mu*``.
    tau_, lambda_ : ndarray of shape (m,)
        Mean and confidence vectors.
    minimax_risk_ : float
        Minimax risk ``R(V)`` (worst-case error probability), which can be
        used to assess the classifier's performance (Theorem 5).
    disagreement_ : float
        Estimate of ``E{psi(mu*, x)}`` that bounds the difference between the
        error of the deterministic and the randomized SMRCs (Theorem 4).
    unsupervised_features_, supervised_features_, supervised_classifier_ :
        Fitted algorithms.
    """

    def __init__(self, unsupervised_features=None, supervised_features=None,
                 supervised_classifier=None, split_unlabeled=False,
                 lambda_scale=1.0, max_iter=100000, batch_size=100, step=None,
                 random_state=None):
        self.unsupervised_features = unsupervised_features
        self.supervised_features = supervised_features
        self.supervised_classifier = supervised_classifier
        self.split_unlabeled = split_unlabeled
        self.lambda_scale = lambda_scale
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.step = step
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _algorithms(self):
        uns = self.unsupervised_features
        sup = self.supervised_features
        clf = self.supervised_classifier
        rs = self.random_state
        uns = PCAKMeansFeatures(random_state=rs) if uns is None else uns
        sup = NeuralNetworkFeatures(random_state=rs) if sup is None else sup
        clf = ConfidentClassifier(random_state=rs) if clf is None else clf
        uns = None if isinstance(uns, str) and uns == "none" else clone(uns)
        sup = None if isinstance(sup, str) and sup == "none" else clone(sup)
        return uns, sup, clone(clf)

    def _phi_uns(self, X):
        Y = len(self.classes_)
        if self.unsupervised_features_ is None:
            return FeatureMap(Y)
        return FeatureMap(Y, [self.unsupervised_features_.transform(X)], [None])

    def _phi_sup(self, X):
        """``Phi_sup`` at ``X`` and the labels predicted by ``h_sup``."""
        Y = len(self.classes_)
        pred, conf = self.supervised_classifier_.predict_confident(X)
        if self.supervised_features_ is None:
            return FeatureMap(Y), pred
        return FeatureMap(Y, [self.supervised_features_.transform(X)], [conf]), pred

    def feature_map(self, X):
        """Feature mapping ``Phi = [Phi_uns; Phi_sup]`` evaluated at ``X``."""
        check_is_fitted(self, "mu_")
        return self._phi_uns(X).concatenate(self._phi_sup(X)[0])

    # ------------------------------------------------------------------
    def fit(self, X_lab, y_lab, X_unl):
        """Learn an SMRC from labeled and unlabeled samples.

        Parameters
        ----------
        X_lab : array-like of shape (l, d)
            Labeled instances.
        y_lab : array-like of shape (l,)
            Labels.
        X_unl : array-like of shape (u, d)
            Unlabeled instances.
        """
        X_lab = np.asarray(X_lab, dtype=float)
        X_unl = np.asarray(X_unl, dtype=float)
        self.classes_, y = np.unique(np.asarray(y_lab), return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        uns, sup, clf = self._algorithms()
        if uns is None and sup is None:
            raise ValueError("at least one feature learner is needed")

        # Unlabeled instances used to learn Psi_uns and to get subgradients.
        if self.split_unlabeled:
            perm = rng.permutation(len(X_unl))
            half = len(X_unl) // 2
            X_feat, X_grad = X_unl[perm[:half]], X_unl[perm[half:]]
        else:
            X_feat = X_grad = X_unl

        # Lines 1-3 of Algorithm 2: supervised and unsupervised algorithms.
        self.supervised_classifier_ = clf.fit(X_lab, y)
        self.supervised_features_ = None if sup is None else sup.fit(X_lab, y)
        self.unsupervised_features_ = None if uns is None else uns.fit(X_feat)

        # Lines 5-7: mean and confidence vectors.
        Y = len(self.classes_)
        tau, lam = [], []
        if uns is not None:
            t, l_ = mean_and_confidence(self._phi_uns(X_lab), y)
            tau.append(t)
            lam.append(l_)
        if sup is not None:
            phi_s, pred = self._phi_sup(X_unl)
            t, l_ = mean_and_confidence(phi_s, pred)
            tau.append(t)
            lam.append(l_)
        tau = np.concatenate(tau)
        lam = self.lambda_scale * np.concatenate(lam)

        # Lines 8-9: feature mapping at the subgradient instances and
        # non-empty uncertainty set (Section 5.2).
        phi_s_grad, pred_grad = self._phi_sup(X_grad)
        phi_grad = self._phi_uns(X_grad).concatenate(phi_s_grad)
        tau_rec = phi_grad.weighted_sum(one_hot(pred_grad, Y)) / len(X_grad)
        lam = np.maximum(lam, np.abs(tau - tau_rec))

        # Line 10: stochastic subgradient method.
        mu, value = fit_smrc(phi_grad, tau, lam, max_iter=self.max_iter,
                             batch_size=self.batch_size, step=self.step,
                             random_state=rng)
        self.tau_, self.lambda_, self.mu_ = tau, lam, mu
        self.minimax_risk_, self.disagreement_ = smrc_objective(phi_grad, tau, lam, mu)
        # Kept to compute performance bounds (see `error_bounds`).
        self._X_lab, self._y_lab, self._X_grad = X_lab, y, X_grad
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X):
        """Values ``Phi(x, y)^T mu*`` for each class (shape ``(n, |Y|)``)."""
        return self.feature_map(np.asarray(X, dtype=float)).scores(self.mu_)

    def predict_proba(self, X):
        """Probabilities ``h(y|x)`` of the (randomized) SMRC rule (3.3)."""
        return classifier_probabilities(self.decision_function(X))

    def predict(self, X):
        """Labels assigned by the deterministic SMRC (4.4)."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def predict_randomized(self, X, random_state=None):
        """Labels sampled from the randomized SMRC ``h(y|x)``."""
        rng = np.random.default_rng(random_state)
        p = self.predict_proba(X)
        p = p / p.sum(axis=1, keepdims=True)
        u = rng.random((p.shape[0], 1))
        idx = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), p.shape[1] - 1)
        return self.classes_[idx]

    def error_bounds(self, delta=0.05, deterministic=True, max_iter=None,
                     random_state=None):
        """High-confidence upper and lower bounds for the error (Theorem 6).

        The bounds use the feature mapping ``Phi_uns`` (whose expectation is
        estimated with the labeled samples) and confidence intervals obtained
        by betting (:func:`smrc.bounds.confidence_vectors`). The
        expectation over instances is approximated with the training
        instances (labeled and unlabeled), as in the paper's experiments.

        Parameters
        ----------
        delta : float, default=0.05
            The bounds hold with probability at least ``1 - delta``.
        deterministic : bool, default=True
            Bound the error of the deterministic SMRC (otherwise the
            randomized SMRC).

        Returns
        -------
        lower, upper : float
        """
        check_is_fitted(self, "mu_")
        if self.unsupervised_features_ is None:
            raise ValueError("error bounds require unsupervised features")
        X_all = np.vstack([self._X_lab, self._X_grad])
        v = self.decision_function(X_all)
        if deterministic:
            h = one_hot(v.argmax(axis=1), len(self.classes_))
        else:
            h = classifier_probabilities(v)
        # Labels used to ensure the problems are bounded: true labels for
        # the labeled instances and h_sup labels for the unlabeled ones.
        pred_unl, _ = self.supervised_classifier_.predict_confident(self._X_grad)
        labels = np.concatenate([self._y_lab, pred_unl])
        return error_bounds(
            self._phi_uns(self._X_lab), self._y_lab, self._phi_uns(X_all), h,
            delta=delta, y_unl_pred=labels, max_iter=self.max_iter if max_iter is None else max_iter,
            batch_size=self.batch_size, step=self.step,
            random_state=self.random_state if random_state is None else random_state)
