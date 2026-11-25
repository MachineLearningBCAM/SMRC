"""Reproduce the SMRC columns of Table 1 in the paper.

For each random stratified partition, an SMRC is learned with l labeled and
u unlabeled samples and evaluated on 1000 test samples. The script reports
the average and standard deviation of the error of the deterministic SMRC,
the error of the supervised classifier h_sup (SUP), and the minimax risk
R(V).

The dataset is a .mat file with a d x n matrix ``x`` and a vector ``y`` (as
``data/usps.mat``). In the paper, `MNIST`, `USPS`, and `MNIST4-9` use pixel
values, and the other datasets use the penultimate layer of a pre-trained
ResNet18 network.

    python examples/reproduce_table.py ../data/usps.mat --n-unl 7000 --reps 20
    python examples/reproduce_table.py cifar10_resnet18.mat --n-unl 10000
"""

import argparse
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

import smrc


def run(X, y, n_lab, n_unl, n_test, seed, k=(50, 50, 200, 200), **kwargs):
    lab, unl, test = smrc.partition(y, n_lab, n_unl, n_test, random_state=seed)
    Xs = StandardScaler().fit(np.vstack([X[lab], X[unl]])).transform(X)
    K1, K2, K3, K4 = k
    clf = smrc.SMRC(
        unsupervised_features=smrc.PCAKMeansFeatures(K1, K2, random_state=seed),
        supervised_features=smrc.NeuralNetworkFeatures(K3, random_state=seed),
        supervised_classifier=smrc.ConfidentClassifier(
            smrc.boosted_trees(K4, random_state=seed), random_state=seed),
        random_state=seed, **kwargs)
    clf.fit(Xs[lab], y[lab], Xs[unl])
    err = np.mean(clf.predict(Xs[test]) != y[test])
    err_sup = np.mean(clf.classes_[clf.supervised_classifier_.predict(Xs[test])] != y[test])
    return err, err_sup, clf.minimax_risk_


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help=".mat file with variables x (d x n) and y")
    parser.add_argument("--n-lab", type=int, default=500)
    parser.add_argument("--n-unl", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--split-unlabeled", action="store_true",
                        help="use half of the unlabeled samples to learn Psi_uns "
                             "and the other half for subgradients")
    args = parser.parse_args()

    X, y = smrc.load_mat(args.data)
    results = []
    for rep in range(1, args.reps + 1):
        start = time.time()
        res = run(X, y, args.n_lab, args.n_unl, args.n_test, seed=rep,
                  split_unlabeled=args.split_unlabeled)
        results.append(res)
        print(f"partition {rep:2d}: SMRC {res[0]:.3f}  SUP {res[1]:.3f}  "
              f"R(V) {res[2]:.3f}  ({time.time() - start:.0f} s)", flush=True)
    r = np.array(results)
    print(f"\nSMRC  {r[:, 0].mean():.3f} +- {r[:, 0].std():.3f}")
    print(f"SUP   {r[:, 1].mean():.3f} +- {r[:, 1].std():.3f}")
    print(f"R(V)  {r[:, 2].mean():.3f} +- {r[:, 2].std():.3f}")


if __name__ == "__main__":
    main()
