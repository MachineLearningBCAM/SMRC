"""Supervised and unsupervised learning algorithms used to build SMRC features.

SMRCs can leverage general algorithms (Section 3.1 of the paper):

* an unsupervised feature learner ``A_uns^feat`` fitted with unlabeled
  instances, giving ``Psi_uns(x)``;
* a supervised feature learner ``A_sup^feat`` fitted with labeled samples,
  giving ``Psi_sup(x)``;
* a supervised classifier ``A_sup^clas`` fitted with labeled samples that
  provides labels ``h_sup(x)`` together with a confidence score, used to
  define the set ``X^conf`` of confidently classified instances.

Any scikit-learn transformer can be used as a feature learner (``fit`` /
``transform``) and any classifier with ``predict_proba`` can be wrapped in
:class:`ConfidentClassifier`.  The classes below implement the
general-purpose algorithms used in the paper's experiments.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def _nearest_centroid_one_hot(X, centers):
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ centers.T + (centers ** 2).sum(1)[None, :]
    out = np.zeros((X.shape[0], centers.shape[0]))
    out[np.arange(X.shape[0]), d2.argmin(axis=1)] = 1.0
    return out


# ----------------------------------------------------------------------
# Unsupervised feature learners
# ----------------------------------------------------------------------
class PCAKMeansFeatures(TransformerMixin, BaseEstimator):
    """Principal components and k-means clusters (paper's Table 1 setting).

    ``Psi_uns(x) = [1, z_1(x), ..., z_K1(x), c_1(x), ..., c_K2(x)]`` where
    ``z_j(x)`` are the projections onto the first ``K1`` principal components
    scaled to unit variance, and ``c(x)`` is the one-hot encoding of the
    nearest of ``K2`` k-means centroids.  The constant feature allows the
    uncertainty set to constrain the labels' marginal (Section 3).

    Parameters
    ----------
    n_components : int, default=50
        Number of principal components ``K1`` (0 to skip PCA features).
    n_clusters : int, default=50
        Number of k-means clusters ``K2`` (0 to skip cluster features).
    constant : bool, default=True
        Whether to include the constant feature.
    random_state : int or None
    """

    def __init__(self, n_components=50, n_clusters=50, constant=True,
                 random_state=None):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.constant = constant
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.n_components > 0:
            self.pca_ = PCA(n_components=self.n_components, whiten=True,
                            random_state=self.random_state).fit(X)
        if self.n_clusters > 0:
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=1,
                                  random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = []
        if self.constant:
            parts.append(np.ones((X.shape[0], 1)))
        if self.n_components > 0:
            parts.append(self.pca_.transform(X))
        if self.n_clusters > 0:
            parts.append(_nearest_centroid_one_hot(X, self.kmeans_.cluster_centers_))
        return np.hstack(parts)


class AutoencoderFeatures(TransformerMixin, BaseEstimator):
    """Autoencoder and k-means clusters (paper's Figure 1(b) setting).

    ``Psi_uns(x) = [1, a_1(x), ..., a_K5(x), c_1(x), ..., c_K2(x)]`` where
    ``a(x)`` is the hidden layer of a one-hidden-layer autoencoder with
    logistic sigmoid activations and ``c(x)`` is the one-hot encoding of the
    nearest k-means centroid.

    The MATLAB experiments use ``trainAutoencoder``; this implementation uses
    scikit-learn's :class:`~sklearn.neural_network.MLPRegressor` trained to
    reconstruct its (min-max scaled) input, which is similar but not
    identical.
    """

    def __init__(self, n_hidden=50, n_clusters=50, constant=True,
                 max_iter=500, alpha=1e-3, random_state=None):
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters
        self.constant = constant
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.scaler_ = MinMaxScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=(self.n_hidden,), activation="logistic",
            alpha=self.alpha, max_iter=self.max_iter,
            random_state=self.random_state).fit(Z, Z)
        if self.n_clusters > 0:
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=1,
                                  random_state=self.random_state).fit(X)
        return self

    def encode(self, X):
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        W, b = self.autoencoder_.coefs_[0], self.autoencoder_.intercepts_[0]
        return 1.0 / (1.0 + np.exp(-(Z @ W + b)))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = []
        if self.constant:
            parts.append(np.ones((X.shape[0], 1)))
        parts.append(self.encode(X))
        if self.n_clusters > 0:
            parts.append(_nearest_centroid_one_hot(X, self.kmeans_.cluster_centers_))
        return np.hstack(parts)


# ----------------------------------------------------------------------
# Supervised feature learners
# ----------------------------------------------------------------------
class NeuralNetworkFeatures(TransformerMixin, BaseEstimator):
    """Features given by the first layer of a feedforward neural network.

    A network with one hidden layer of size ``K3`` is trained with the
    labeled samples and ``Psi_sup(x)`` is given by the first-layer
    pre-activations ``W x`` standardized with the labeled samples' mean and
    standard deviation, as in the MATLAB implementation (``learn_sup.m`` and
    ``feat_sup.m``).  Setting ``activation=True`` uses instead the hidden
    layer's outputs ``ReLU(W x + b)`` (standardized in the same way).

    Parameters
    ----------
    n_hidden : int, default=200
        Size ``K3`` of the hidden layer.
    activation : bool, default=False
        Whether to apply the bias and ReLU activation of the hidden layer.
    alpha : float, default=0.0
        L2 regularization of the network (MATLAB's ``fitcnet`` default is 0).
    max_iter : int, default=1000
    random_state : int or None
    """

    def __init__(self, n_hidden=200, activation=False, alpha=0.0,
                 max_iter=1000, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def _raw(self, X):
        F = np.asarray(X, dtype=float) @ self.network_.coefs_[0]
        if self.activation:
            F = np.maximum(F + self.network_.intercepts_[0], 0.0)
        return F

    def fit(self, X, y):
        self.network_ = MLPClassifier(
            hidden_layer_sizes=(self.n_hidden,), solver="lbfgs",
            alpha=self.alpha, max_iter=self.max_iter,
            random_state=self.random_state).fit(X, y)
        F = self._raw(X)
        self.mean_ = F.mean(axis=0)
        std = F.std(axis=0, ddof=1)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        return (self._raw(X) - self.mean_) / self.scale_


class IdentityFeatures(TransformerMixin, BaseEstimator):
    """``Psi(x) = x``; e.g., to skip supervised feature learning when very
    few labeled samples are available (end of Section 3.1)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


# ----------------------------------------------------------------------
# Supervised classifier with confidence threshold
# ----------------------------------------------------------------------
def boosted_trees(n_estimators=200, random_state=None):
    """Boosted ensemble of shallow trees (similar to MATLAB's
    ``fitcensemble`` defaults: AdaBoost with trees of at most 10 splits)."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_leaf_nodes=11, random_state=random_state),
        n_estimators=n_estimators, random_state=random_state)


class ConfidentClassifier(ClassifierMixin, BaseEstimator):
    """Supervised classifier ``h_sup`` together with the set ``X^conf``.

    The confidence of the prediction for ``x`` is the probability that the
    classifier assigns to its predicted label, and ``X^conf`` is the set of
    instances whose confidence is larger than a threshold.

    Parameters
    ----------
    estimator : classifier with ``predict_proba``, optional
        Defaults to :func:`boosted_trees` with 200 trees.
    threshold : {"all", "cv"} or float, default="all"
        * ``"all"``: all instances are taken as confidently classified, so
          that ``Phi_sup(x, y) = e_y (x) Psi_sup(x)`` for every ``x``. This
          matches the MATLAB implementation, where, for simplicity, the
          confidence scores are obtained with the ``doublelogit`` transform;
          such scores may saturate at 1 for boosted ensembles, in which case
          all instances are taken as confidently classified.
        * ``"cv"``: threshold obtained by cross-validation over the labeled
          samples (Section 5.2): the cross-validated error ``e`` is estimated
          and the threshold is the ``100 e`` percentile of the
          cross-validated confidence scores.
        * float: fixed threshold.
    cv : int, default=10
        Number of cross-validation folds (only used if ``threshold="cv"``).
    random_state : int or None
    """

    def __init__(self, estimator=None, threshold="all", cv=10, random_state=None):
        self.estimator = estimator
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        est = (boosted_trees(random_state=self.random_state)
               if self.estimator is None else self.estimator)
        self.estimator_ = clone(est).fit(X, y)
        self.classes_ = self.estimator_.classes_
        if isinstance(self.threshold, str) and self.threshold == "all":
            self.threshold_ = -np.inf
        elif isinstance(self.threshold, str) and self.threshold == "cv":
            self.threshold_ = self._cv_threshold(est, X, y)
        elif isinstance(self.threshold, str):
            raise ValueError("threshold must be 'all', 'cv', or a number")
        else:
            self.threshold_ = float(self.threshold)
        return self

    def _cv_threshold(self, est, X, y):
        _, counts = np.unique(y, return_counts=True)
        n_folds = int(min(self.cv, counts.min()))
        if n_folds < 2:  # not enough samples for cross-validation
            self.cv_error_ = np.nan
            return -np.inf
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state)
        proba = cross_val_predict(clone(est), X, y, cv=folds, method="predict_proba")
        pred = self.classes_[proba.argmax(axis=1)]
        self.cv_error_ = float(np.mean(pred != np.asarray(y)))
        return float(np.percentile(proba.max(axis=1), 100 * self.cv_error_)) - 1e-4

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_confident(self, X):
        """Return predicted label indices and the mask of ``x in X^conf``."""
        proba = self.predict_proba(X)
        return proba.argmax(axis=1), proba.max(axis=1) > self.threshold_
