# File: close_gap.py
# Experiment: Closing the generator gap on Split-MNIST (generative -> oracle)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   The oracle (replay of real images) marks the ceiling of replay (about 90.2%).
#   Generative dreaming with nd=1000 reached 86.8%. Here we close the gap with two
#   levers: (1) more dreams (nd), and (2) sampling latent codes from a GAUSSIAN
#   MIXTURE (GMM) instead of a single Gaussian, to capture sub-modes of each class.
#   Generators are trained once per seed and reused across all variants.
#
# Method:
#   - Class-incremental Split-MNIST with the local LEL classifier
#   - Per-class local generators, sampled either from a single Gaussian or a GMM
#   - Sweep over the number of dreams nd
#
# Outputs:
#   - Console: Class-IL accuracy for each (sampler, nd) variant, with mean +/- std
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 close_gap.py
#   (requires full_split_mnist.py in the same folder and /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: [0, 1] (averaged)
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
                              lel, g_train, g_infer, onehot, class_data, fwd, K)

tanh = np.tanh
H, EP, lr, g, T, nper = 256, 35, 0.1, 0.2, 15, 2000
GZ, GH, TG, PZ, DZ = 0.1, 0.2, 35, 0.02, 48   # must match the defaults of g_train


def evaluate(W, mu):
    """Class-incremental accuracy on the test set of all classes."""
    a = []
    for c in range(K):
        idx = np.where(yte_all == c)[0][:400]; a.append(np.mean(np.argmax(fwd(W, Xte_all[idx] - mu), 1) == c))
    return float(np.mean(a))


def decode(G, z):
    """Decode latent codes z through the generator G."""
    W1, b1, W2, b2 = G; return tanh(z @ W1.T + b1) @ W2.T + b2


def fit_gmm(Z, k, rng, iters=25):
    """Fit a simple diagonal-covariance GMM to the latent codes Z (k-means + moments)."""
    c = Z[rng.choice(len(Z), k, replace=False)].copy(); a = np.zeros(len(Z), int)
    for _ in range(iters):
        a = ((Z[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = Z[a == j].mean(0)
    m, v, w = [], [], []
    for j in range(k):
        Zj = Z[a == j]
        if len(Zj) >= 2:
            m.append(Zj.mean(0)); v.append(Zj.var(0) + 1e-4); w.append(len(Zj))
    w = np.array(w) / sum(w); return w, np.array(m), np.array(v)


def gmm_sample(gmm, n, rng):
    """Draw n samples from the fitted GMM."""
    w, m, v = gmm; comp = rng.choice(len(w), n, p=w)
    return m[comp] + np.sqrt(v[comp]) * rng.normal(0, 1, (n, m.shape[1]))


def gauss_stats(Z): return Z.mean(0), np.cov(Z.T) + 1e-4 * np.eye(Z.shape[1])


def gauss_sample(st_, n, rng): zm, zc = st_; return rng.multivariate_normal(zm, zc, size=n)


def prep(seed):
    """Train one generator per class and precompute Gaussian and GMM latent stats."""
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr_all, ytr_all, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0); Xc = {c: Xc[c] - mu for c in range(K)}
    G = {}; gss = {}; gmm = {}
    for c in range(K):
        gc, _ = g_train(Xc[c], seed=seed + c); G[c] = gc
        Z = g_infer(Xc[c], gc, GZ, GH, TG, PZ, DZ)[0]
        gss[c] = gauss_stats(Z); gmm[c] = fit_gmm(Z, 5, rng)
    return Xc, mu, G, gss, gmm, rng


def W0(rng): return (rng.normal(0, 1, (H, 784)) * np.sqrt(1/784), np.zeros((1, H)),
                     rng.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K)))


def run(Xc, mu, G, stats, sampler, nd, rng, seed):
    """Run one class-incremental sequence with the given latent sampler."""
    W = W0(rng); seen = []
    for t in range(5):
        ca, cb = 2*t, 2*t+1
        Xcur = np.vstack([Xc[ca], Xc[cb]]); Ycur = onehot([ca]*len(Xc[ca]) + [cb]*len(Xc[cb]), seen); old = list(seen)
        if old:
            Xd = np.vstack([decode(G[c], sampler(stats[c], nd, rng)) for c in old]); Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xt, Yt = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        else:
            Xt, Yt = Xcur, Ycur
        W = lel(W, Xt, Yt, EP, lr, g, T, seed=seed+100+t); seen = sorted(set(seen) | {ca, cb})
    return evaluate(W, mu)


if __name__ == "__main__":
    seeds = [0, 1]; t0 = time.time()
    res = {}
    for s in seeds:
        Xc, mu, G, gss, gmm, rng = prep(s)
        for nd in [1000, 2000, 3000]:
            res.setdefault(('gauss', nd), []).append(run(Xc, mu, G, gss, gauss_sample, nd, rng, s))
        res.setdefault(('gmm', 2000), []).append(run(Xc, mu, G, gmm, gmm_sample, 2000, rng, s))
        res.setdefault(('gmm', 3000), []).append(run(Xc, mu, G, gmm, gmm_sample, 3000, rng, s))
        print(f" seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    print("=" * 56); print(" Oracle (ceiling of replay) = 90.2%")
    for (kind, nd), v in res.items():
        print(f" {kind:<6} nd={nd:<5} Class-IL = {st.mean(v):.1%} +/- {st.pstdev(v):.1%}")
    print("=" * 56)
