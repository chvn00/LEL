# File: forgetting_curve.py
# Experiment: Forgetting curve over the Split-MNIST task sequence
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Track how accuracy evolves as tasks arrive: the accuracy on the first task
#   (T0) and the average accuracy over all tasks seen so far. Vanilla local
#   learning degrades toward chance, while generative dreaming retains earlier
#   tasks throughout.
#
# Method:
#   - Class-incremental Split-MNIST with the local LEL classifier
#   - Two modes: vanilla and generative dreaming (mode 'gen')
#   - After each task, record T0 accuracy and the running average accuracy
#
# Outputs:
#   - Console: the T0 curve and the average curve for vanilla and for generative
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 forgetting_curve.py
#   (requires full_split_mnist.py in the same folder and /tmp/mnist.npz)
#
# Reproducibility:
#   Seed: 0 (NumPy default_rng)
#   Python 3, NumPy
#   Data: /tmp/mnist.npz (arrays Xtr, ytr, Xte, yte)
#
# Author:
#   Cesar Hernando Valencia Niño
#   Facultad de Ingeniería Mecatrónica
#   Universidad Santo Tomás, Seccional Bucaramanga, Colombia
#   Email: cesar.valencia@ustabuca.edu.co
#   ORCID: 0000-0001-6077-6458
#
# License:
#   MIT License, or the same license declared in the repository.
#
# Citation:
#   Valencia Niño, C.H. Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods.
# =============================================================================
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from full_split_mnist import (Xtr_all, ytr_all, Xte_all, yte_all, lel, g_train, g_sample, onehot, class_data, fwd, K)

H, EP, lr, g, T, nper, nd = 256, 35, 0.1, 0.2, 15, 2000, 1000


def acc_task(W, mu, a, b):
    """Accuracy on the two classes (a, b) of one task."""
    accs = []
    for c in (a, b):
        idx = np.where(yte_all == c)[0][:400]; accs.append(np.mean(np.argmax(fwd(W, Xte_all[idx] - mu), 1) == c))
    return float(np.mean(accs))


def run(mode, seed=0):
    """Run the full sequence; return (T0 curve, average-accuracy curve)."""
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr_all, ytr_all, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0); Xc = {c: Xc[c] - mu for c in range(K)}
    gens = {c: g_train(Xc[c], seed=seed + c) for c in range(K)} if mode == 'gen' else {}
    W = (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)), rng.normal(0, 1, (K, H)) * np.sqrt(1 / H), np.zeros((1, K)))
    seen = []; t0_curve = []; avg_curve = []
    for t in range(5):
        ca, cb = 2 * t, 2 * t + 1; Xcur = np.vstack([Xc[ca], Xc[cb]]); Ycur = onehot([ca] * len(Xc[ca]) + [cb] * len(Xc[cb]), seen); old = list(seen)
        if mode == 'gen' and old:
            Xd = np.vstack([g_sample(*gens[c], nd, rng) for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        else:
            Xt, Yt = Xcur, Ycur
        W = lel(W, Xt, Yt, EP, lr, g, T, seed=seed + 100 + t); seen = sorted(set(seen) | {ca, cb})
        t0_curve.append(acc_task(W, mu, 0, 1))
        avg_curve.append(float(np.mean([acc_task(W, mu, 2 * k, 2 * k + 1) for k in range(t + 1)])))
    return t0_curve, avg_curve


for mode in ['vanilla', 'gen']:
    t0, avg = run(mode)
    print(mode, "T0:", ["%.3f" % x for x in t0])
    print(mode, "AVG:", ["%.3f" % x for x in avg])
