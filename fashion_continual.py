# File: fashion_continual.py
# Experiment: Split-Fashion-MNIST class-incremental generalisation (no backprop)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Run the same LEL pipeline (local classifier + local generative dreaming) on a
#   harder dataset (clothing). A generality test beyond MNIST. Reuses the
#   validated routines of full_split_mnist.py.
#
# Method:
#   - Class-incremental Split-Fashion-MNIST with the local LEL classifier
#   - Modes: ceiling (joint training), vanilla, oracle (real-image replay),
#     noise (noise replay), and generative (local dreaming)
#
# Outputs:
#   - Console: ceiling, oracle, vanilla, noise and generative accuracies
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 fashion_continual.py
#   (requires full_split_mnist.py in the same folder and /tmp/fashion.npz)
#
# Reproducibility:
#   Seeds: [0, 1] (averaged)
#   Python 3, NumPy
#   Data: /tmp/fashion.npz (arrays Xtr, ytr, Xte, yte)
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
#   Valencia Niño, C.H. Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges.
# =============================================================================
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, time, statistics as st
from full_split_mnist import lel, g_train, g_sample, onehot, class_data, fwd, K

d = np.load('/tmp/fashion.npz')
Xtr = d['Xtr'].astype(np.float64) / 255.0; ytr = d['ytr']
Xte = d['Xte'].astype(np.float64) / 255.0; yte = d['yte']
tanh = np.tanh


def evaluate(W, mu, seen):
    """Class-incremental accuracy on the test set of seen classes."""
    a = []
    for c in seen:
        idx = np.where(yte == c)[0][:400]
        a.append(np.mean(np.argmax(fwd(W, Xte[idx] - mu), 1) == c))
    return float(np.mean(a))


def setup(seed, nper):
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr, ytr, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0)
    return {c: Xc[c] - mu for c in range(K)}, mu, rng


def newW(rng, H):
    return (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
            rng.normal(0, 1, (K, H)) * np.sqrt(1 / H), np.zeros((1, K)))


def ceiling(seed, H=256, EP=35, lr=0.1, g=0.2, T=15, nper=2000):
    """Joint-training ceiling (all classes at once)."""
    Xc, mu, rng = setup(seed, nper)
    X = np.vstack([Xc[c] for c in range(K)]); Y = onehot([c for c in range(K) for _ in range(nper)], list(range(K)))
    return evaluate(lel(newW(rng, H), X, Y, EP, lr, g, T, seed=seed), mu, list(range(K)))


def continual(mode, seed, H=256, EP=35, lr=0.1, g=0.2, T=15, nper=2000, nd=1000):
    """Run one class-incremental sequence under the given mode."""
    Xc, mu, rng = setup(seed, nper)
    gens = {c: g_train(Xc[c], seed=seed + c) for c in range(K)} if mode == 'generative' else {}
    W = newW(rng, H); seen = []
    for t in range(5):
        ca, cb = 2 * t, 2 * t + 1
        Xcur = np.vstack([Xc[ca], Xc[cb]]); Ycur = onehot([ca] * len(Xc[ca]) + [cb] * len(Xc[cb]), seen)
        old = list(seen)
        if mode == 'vanilla' or not old:
            Xt, Yt = Xcur, Ycur
        elif mode == 'oracle':
            Xd = np.vstack([Xc[c][rng.permutation(len(Xc[c]))[:nd]] for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        elif mode == 'noise':
            Xn = rng.normal(0, Xcur.std(), (nd * len(old), 784)); Yn = onehot(np.argmax(fwd(W, Xn), 1).tolist(), seen)
            Xt, Yt = np.vstack([Xcur, Xn]), np.vstack([Ycur, Yn])
        elif mode == 'generative':
            Xd = np.vstack([g_sample(*gens[c], nd, rng) for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        W = lel(W, Xt, Yt, EP, lr, g, T, seed=seed + 100 + t); seen = sorted(set(seen) | {ca, cb})
    return evaluate(W, mu, list(range(K)))


if __name__ == "__main__":
    seeds = [0, 1]
    print("=" * 64); print(" Split-Fashion-MNIST class-incremental (LEL, no backpropagation)"); print("=" * 64)
    t0 = time.time()
    print(f" CEILING (joint)         = {st.mean([ceiling(s) for s in seeds]):.1%}", flush=True)
    print(f" ORACLE (real replay)    = {st.mean([continual('oracle', s) for s in seeds]):.1%}", flush=True)
    print(f" VANILLA                 = {st.mean([continual('vanilla', s) for s in seeds]):.1%}", flush=True)
    print(f" NOISE                   = {st.mean([continual('noise', s) for s in seeds]):.1%}", flush=True)
    g = st.mean([continual('generative', s) for s in seeds])
    print(f" GENERATIVE (local)      = {g:.1%}", flush=True)
    print(f" time {time.time()-t0:.0f}s"); print("=" * 64)
