# File: generator_capacity_ablation.py
# Experiment: Generator capacity vs. class-incremental retention (Split-MNIST)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Test the thesis that the quality (capacity) of the generator bounds
#   retention, not the continual mechanism. A LINEAR generator (the per-class
#   principal subspace, the fixed point of Oja's rule) is used because it is
#   unconditionally stable (SVD of the data, no iterative dynamics that can
#   diverge). Retention is measured as a function of the subspace dimension k:
#   more principal components give better coverage of the data manifold.
#
# Method:
#   - Class-incremental Split-MNIST, local LEL classifier
#   - Linear per-class generator (top-k principal components)
#   - Subspace dimension k swept over {1, 2, 4, 8, 16, 32, 64}
#
# Outputs:
#   - Console: k, class-incremental accuracy, and wall-clock time
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 generator_capacity_ablation.py
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
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from full_split_mnist import (Xtr_all, ytr_all, class_data, lel, onehot, evaluate, K)


def pca_gen(Xc, k):
    """Linear generator: class mean plus the top-k principal components."""
    m = Xc.mean(0); Z = Xc - m
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Vk = Vt[:k]; var = (S[:k] ** 2) / max(len(Xc) - 1, 1)
    return m, Vk, np.sqrt(var)


def pca_sample(gen, n, rng):
    """Draw n samples from the linear generator."""
    m, Vk, sd = gen
    z = rng.normal(0, 1, (n, len(sd))) * sd
    return m + z @ Vk


def run_k(k, seed=0, H=256, EP=35, lr=0.1, g=0.2, T=15, nper=2000, nd=1000):
    """Run one class-incremental sequence with a capacity-k generator."""
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr_all, ytr_all, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0)
    Xc = {c: Xc[c] - mu for c in range(K)}
    gens = {c: pca_gen(Xc[c], k) for c in range(K)}
    W = (rng.normal(0, 1, (H, 784)) * np.sqrt(1/784), np.zeros((1, H)),
         rng.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K)))
    tasks = [(2*t, 2*t+1) for t in range(5)]
    seen = []
    for t, (ca, cb) in enumerate(tasks):
        Xcur = np.vstack([Xc[ca], Xc[cb]])
        Ycur = onehot([ca]*len(Xc[ca]) + [cb]*len(Xc[cb]), seen)
        old = [c for c in seen]
        if not old:
            Xtr, Ytr = Xcur, Ycur
        else:
            Xd = np.vstack([pca_sample(gens[c], nd, rng) for c in old])
            Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xtr, Ytr = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        W = lel(W, Xtr, Ytr, EP, lr, g, T, seed=seed+100+t)
        seen = sorted(set(seen) | {ca, cb})
    return evaluate(W, mu, list(range(K)))


print("k   classIL%")
for k in [1, 2, 4, 8, 16, 32, 64]:
    t0 = time.time()
    acc = run_k(k)
    print(f"{k:<3} {acc*100:5.1f}   ({time.time()-t0:.0f}s)", flush=True)
