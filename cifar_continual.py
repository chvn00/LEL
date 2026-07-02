# File: cifar_continual.py
# Experiment: Split-CIFAR-10 class-incremental generalisation (no backpropagation)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Stress test on a harder dataset (colour, 3072 dimensions). A shallow MLP on
#   raw CIFAR pixels is modest even with backpropagation (the ceiling is low). We
#   report the joint-training ceiling as a reference and measure how much of it
#   generative dreaming recovers. Reuses the validated routines of
#   full_split_mnist.py. Everything is local, with no backpropagation.
#
# Method:
#   - Class-incremental Split-CIFAR-10 with the local LEL classifier
#   - Modes: ceiling (joint training), vanilla, oracle (real-image replay),
#     and generative (local dreaming)
#   - Mean centring and a smaller relaxation step for stability in 3072-D
#
# Outputs:
#   - Console: ceiling, vanilla, oracle and generative accuracies
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 cifar_continual.py
#   (requires full_split_mnist.py in the same folder and /tmp/cifar.npz)
#
# Reproducibility:
#   Seed: 0 (NumPy default_rng)
#   Python 3, NumPy
#   Data: /tmp/cifar.npz (arrays Xtr, ytr, Xte, yte)
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
import numpy as np, time, statistics as st
from full_split_mnist import lel, g_train, g_sample, onehot, class_data, fwd, K

tanh = np.tanh
d = np.load('/tmp/cifar.npz')
Xtr = d['Xtr'].astype(np.float64) / 255.0; ytr = d['ytr']
Xte = d['Xte'].astype(np.float64) / 255.0; yte = d['yte']
Din = Xtr.shape[1]
H, EP, lr, g, T, nper, nd = 512, 60, 0.05, 0.1, 15, 1500, 800
# mean centring (simpler and better than per-pixel standardisation on raw CIFAR)
# and a smaller relaxation step g for stability in 3072 dimensions
MU = Xtr.mean(0)


def norm(Z): return Z - MU


def evaluate(W, seen):
    """Class-incremental accuracy on the test set of seen classes."""
    a = []
    for c in seen:
        idx = np.where(yte == c)[0][:400]; a.append(np.mean(np.argmax(fwd(W, norm(Xte[idx])), 1) == c))
    return float(np.mean(a))


def setup(seed):
    rng = np.random.default_rng(seed)
    Xc = {c: norm(class_data(Xtr, ytr, c, nper, rng)) for c in range(K)}
    return Xc, rng


def newW(rng):
    return (rng.normal(0, 1, (H, Din)) * np.sqrt(1/Din), np.zeros((1, H)),
            rng.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K)))


def ceiling(seed):
    """Joint-training ceiling (all classes at once)."""
    Xc, rng = setup(seed)
    X = np.vstack([Xc[c] for c in range(K)]); Y = onehot([c for c in range(K) for _ in range(nper)], list(range(K)))
    return evaluate(lel(newW(rng), X, Y, EP, lr, g, T, seed=seed), list(range(K)))


def continual(mode, seed):
    """Run one class-incremental sequence under the given mode."""
    Xc, rng = setup(seed)
    gens = {c: g_train(Xc[c], dz=64, dh=256, EP=30, lr=0.03, gz=0.02, gh=0.05, Tg=25, pz=0.3, seed=seed+c) for c in range(K)} if mode == 'generative' else {}
    W = newW(rng); seen = []
    for t in range(5):
        ca, cb = 2*t, 2*t+1
        Xcur = np.vstack([Xc[ca], Xc[cb]]); Ycur = onehot([ca]*len(Xc[ca]) + [cb]*len(Xc[cb]), seen); old = list(seen)
        if mode == 'vanilla' or not old:
            Xt, Yt = Xcur, Ycur
        elif mode == 'oracle':
            Xd = np.vstack([Xc[c][rng.permutation(len(Xc[c]))[:nd]] for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        elif mode == 'generative':
            Xd = np.vstack([g_sample(*gens[c], nd, rng) for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        W = lel(W, Xt, Yt, EP, lr, g, T, seed=seed+100+t); seen = sorted(set(seen) | {ca, cb})
    return evaluate(W, list(range(K)))


if __name__ == "__main__":
    seeds = [0]
    t0 = time.time()
    print("=" * 60); print(" Split-CIFAR-10 class-incremental (LEL, no backpropagation)"); print("=" * 60)
    print(f" CEILING (joint)    = {st.mean([ceiling(s) for s in seeds]):.1%}", flush=True)
    print(f" VANILLA            = {st.mean([continual('vanilla', s) for s in seeds]):.1%}", flush=True)
    print(f" ORACLE             = {st.mean([continual('oracle', s) for s in seeds]):.1%}", flush=True)
    print(f" GENERATIVE (local) = {st.mean([continual('generative', s) for s in seeds]):.1%}", flush=True)
    print(f" time {time.time()-t0:.0f}s"); print("=" * 60)
