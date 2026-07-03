# File: reinforce.py
# Experiment: Publication reinforcements: 5 seeds, Task-IL/Class-IL, nd ablation
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Strengthen the headline result: report mean +/- standard deviation over 5
#   seeds for generative dreaming, in both the class-incremental and the
#   task-incremental settings, plus an ablation of the number of dreams nd
#   (reusing the generators of seed 0). Reuses the validated routines of
#   full_split_mnist.py.
#
# Method:
#   - Class-incremental Split-MNIST with local LEL + generative dreaming
#   - Class-IL and Task-IL evaluation
#   - nd ablation on seed 0
#
# Outputs:
#   - Console: per-seed Class-IL/Task-IL, nd ablation, and 5-seed summary
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 reinforce.py
#   (requires full_split_mnist.py in the same folder and /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: [0, 1, 2, 3, 4]
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
#   Valencia Niño, C.H. Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges.
# =============================================================================
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, time, statistics as st
from full_split_mnist import (Xtr_all, ytr_all, Xte_all, yte_all,
                              lel, g_train, g_sample, onehot, class_data, fwd, K)

tanh = np.tanh
H, EP, lr, g, T, nper = 256, 35, 0.1, 0.2, 15, 2000


def class_il(W, mu):
    """Class-incremental accuracy (argmax over all 10 outputs)."""
    a = []
    for c in range(K):
        idx = np.where(yte_all == c)[0][:400]; a.append(np.mean(np.argmax(fwd(W, Xte_all[idx] - mu), 1) == c))
    return float(np.mean(a))


def task_il(W, mu):
    """Task-incremental accuracy (argmax restricted to each task's two classes)."""
    a = []
    for t in range(5):
        ca, cb = 2 * t, 2 * t + 1
        for c in (ca, cb):
            idx = np.where(yte_all == c)[0][:400]
            out = fwd(W, Xte_all[idx] - mu)[:, [ca, cb]]
            a.append(np.mean(np.array([ca, cb])[np.argmax(out, 1)] == c))
    return float(np.mean(a))


def prep(seed):
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr_all, ytr_all, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0); Xc = {c: Xc[c] - mu for c in range(K)}
    gens = {c: g_train(Xc[c], seed=seed + c) for c in range(K)}
    return Xc, mu, gens, rng


def W0(rng): return (rng.normal(0, 1, (H, 784)) * np.sqrt(1/784), np.zeros((1, H)),
                     rng.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K)))


def continual(Xc, gens, rng, nd, seed):
    """Run one class-incremental sequence with nd dreams per old class."""
    W = W0(rng); seen = []
    for t in range(5):
        ca, cb = 2*t, 2*t+1
        Xcur = np.vstack([Xc[ca], Xc[cb]]); Ycur = onehot([ca]*len(Xc[ca]) + [cb]*len(Xc[cb]), seen)
        old = list(seen)
        if nd > 0 and old:
            Xd = np.vstack([g_sample(*gens[c], nd, rng) for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        else:
            Xt, Yt = Xcur, Ycur
        W = lel(W, Xt, Yt, EP, lr, g, T, seed=seed+100+t); seen = sorted(set(seen) | {ca, cb})
    return W


if __name__ == "__main__":
    t0 = time.time(); seeds = [0, 1, 2, 3, 4]; CIL, TIL = [], []
    for s in seeds:
        Xc, mu, gens, rng = prep(s)
        W = continual(Xc, gens, rng, 1000, s)
        cil, til = class_il(W, mu), task_il(W, mu); CIL.append(cil); TIL.append(til)
        print(f" seed {s}: Class-IL={cil:.1%}  Task-IL={til:.1%}  ({time.time()-t0:.0f}s)", flush=True)
        if s == 0:  # nd ablation reusing the generators of seed 0
            print(" --- ablation of the number of dreams (nd), seed 0 ---", flush=True)
            for nd in [0, 250, 500, 2000]:
                Wn = continual(Xc, gens, rng, nd, s)
                print(f"   nd={nd:<5} Class-IL={class_il(Wn, mu):.1%}", flush=True)
    print("=" * 60)
    print(f" GENERATIVE Class-IL = {st.mean(CIL):.1%} +/- {st.pstdev(CIL):.1%}  (5 seeds)")
    print(f" GENERATIVE Task-IL  = {st.mean(TIL):.1%} +/- {st.pstdev(TIL):.1%}  (5 seeds)")
    print(f" total time {time.time()-t0:.0f}s")
    print("=" * 60)
