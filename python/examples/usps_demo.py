"""SMRC on the USPS dataset (Python counterpart of matlab/main.m).

Runs SMRCs with the settings used for Table 1 in the paper (l = 500 labeled
samples, u = 7000 unlabeled samples, 1000 test samples, K1 = K2 = 50,
K3 = K4 = 200) for one random partition of the data.

    python examples/usps_demo.py [--seed 1] [--bounds]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import smrc

DATA = Path(__file__).resolve().parents[2] / "data" / "usps.mat"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bounds", action="store_true",
                        help="also compute the performance bounds of Theorem 6")
    args = parser.parse_args()

    X, y = smrc.load_mat(DATA)
    lab, unl, test = smrc.partition(y, n_lab=500, n_unl=7000, n_test=1000,
                                    random_state=args.seed)
    scaler = StandardScaler().fit(np.vstack([X[lab], X[unl]]))
    X = scaler.transform(X)

    start = time.time()
    clf = smrc.SMRC(
        unsupervised_features=smrc.PCAKMeansFeatures(n_components=50, n_clusters=50,
                                                     random_state=args.seed),
        supervised_features=smrc.NeuralNetworkFeatures(n_hidden=200,
                                                       random_state=args.seed),
        supervised_classifier=smrc.ConfidentClassifier(
            smrc.boosted_trees(n_estimators=200, random_state=args.seed),
            random_state=args.seed),
        random_state=args.seed,
    )
    clf.fit(X[lab], y[lab], X[unl])
    print(f"Training time:           {time.time() - start:.0f} s")

    err = np.mean(clf.predict(X[test]) != y[test])
    err_sup = np.mean(clf.supervised_classifier_.predict(X[test])
                      != np.searchsorted(clf.classes_, y[test]))
    print(f"SMRC error:              {err:.3f}")
    print(f"Supervised error:        {err_sup:.3f}")
    print(f"Minimax risk R(V):       {clf.minimax_risk_:.3f}")

    if args.bounds:
        lower, upper = clf.error_bounds(delta=0.05)
        print(f"High-conf. upper bound:  {upper:.3f}")
        print(f"High-conf. lower bound:  {lower:.3f}")


if __name__ == "__main__":
    main()
