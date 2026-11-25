"""Semi-supervised minimax risk classifiers (SMRCs).

Python implementation of the methods in

    S. Mazuelas, "Leveraging Supervised and Unsupervised Methods for Minimax
    Semi-supervised Learning", Information and Inference: A Journal of the IMA.
"""

from ._smrc import SMRC
from ._phi import FeatureMap
from .features import (
    AutoencoderFeatures,
    ConfidentClassifier,
    IdentityFeatures,
    NeuralNetworkFeatures,
    PCAKMeansFeatures,
    boosted_trees,
)
from .data import load_mat, partition

__version__ = "1.0.0"

__all__ = [
    "SMRC",
    "FeatureMap",
    "AutoencoderFeatures",
    "ConfidentClassifier",
    "IdentityFeatures",
    "NeuralNetworkFeatures",
    "PCAKMeansFeatures",
    "boosted_trees",
    "load_mat",
    "partition",
]
