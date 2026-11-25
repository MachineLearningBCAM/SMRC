"""Reproduce Figure 1 in the paper: SMRC error and performance guarantees
as the capabilities of the supervised and unsupervised algorithms increase.

For each percentage P, SMRCs use K1 = K2 = 50P/100 (or K5 = 50P/100 hidden
units of an autoencoder with ``--autoencoder``), K3 = 200P/100, and
K4 = 200P/100, with l = 2000 labeled samples. The figure shows the SMRC
error, the minimax risk R(V), and the high-confidence upper and lower bounds
of Theorem 6 (delta = 0.05).

    python examples/reproduce_figure.py cifar10_resnet18.mat --reps 20
    python examples/reproduce_figure.py fashionmnist_resnet18.mat --autoencoder
"""

import argparse
import math

import numpy as np
from sklearn.preprocessing import StandardScaler

import smrc

PERCENTAGES = [1, 5, 10, 50, 80, 100]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help=".mat file with variables x (d x n) and y")
    parser.add_argument("--n-lab", type=int, default=2000)
    parser.add_argument("--n-unl", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--autoencoder", action="store_true")
    parser.add_argument("--output", default="figure_bounds.pdf")
    args = parser.parse_args()

    X, y = smrc.load_mat(args.data)
    res = np.zeros((args.reps, len(PERCENTAGES), 4))  # upper, R(V), error, lower
    for rep in range(args.reps):
        seed = rep + 1
        lab, unl, test = smrc.partition(y, args.n_lab, args.n_unl, args.n_test,
                                        random_state=seed)
        Xs = StandardScaler().fit(np.vstack([X[lab], X[unl]])).transform(X)
        for j, P in enumerate(PERCENTAGES):
            k12 = math.ceil(50 * P / 100)
            k34 = math.ceil(200 * P / 100)
            if args.autoencoder:
                uns = smrc.AutoencoderFeatures(k12, k12, random_state=seed)
            else:
                uns = smrc.PCAKMeansFeatures(k12, k12, random_state=seed)
            clf = smrc.SMRC(
                unsupervised_features=uns,
                supervised_features=smrc.NeuralNetworkFeatures(k34, random_state=seed),
                supervised_classifier=smrc.ConfidentClassifier(
                    smrc.boosted_trees(k34, random_state=seed), random_state=seed),
                random_state=seed)
            clf.fit(Xs[lab], y[lab], Xs[unl])
            lower, upper = clf.error_bounds(delta=0.05)
            err = np.mean(clf.predict(Xs[test]) != y[test])
            res[rep, j] = upper, clf.minimax_risk_, err, lower
            print(f"partition {seed:2d}, P = {P:3d}: error {err:.3f}  R(V) "
                  f"{clf.minimax_risk_:.3f}  bounds [{lower:.3f}, {upper:.3f}]",
                  flush=True)
        np.save(args.output.rsplit(".", 1)[0] + ".npy", res[:rep + 1])

    import matplotlib.pyplot as plt

    labels = ["High-conf. upper bound", "Minimax risk $R(\\mathcal{V})$",
              "SMRC error", "High-conf. lower bound"]
    colors = ["#007f00", "#0000ff", "#ff0000", "#00c000"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for k in range(4):
        mean, std = res[:, :, k].mean(0), res[:, :, k].std(0)
        ax.fill_between(PERCENTAGES, mean - std, mean + std, color=colors[k], alpha=0.15,
                        linewidth=0)
        ax.plot(PERCENTAGES, mean, color=colors[k], linewidth=2, label=labels[k])
    ax.set_xlabel("Percentage $P$")
    ax.set_ylabel("Error probability")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
