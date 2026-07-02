# File: baselines_si.py
# Experiment: Synaptic Intelligence (SI) backpropagation baseline on class-IL Split-MNIST
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Confirm that, like EWC, Synaptic Intelligence does not mitigate forgetting
#   in the class-incremental setting: the best result is about 19.4% (the same
#   as vanilla), and strong regularization collapses to chance (10%). This
#   contrasts with LEL plus generative dreaming (about 86.7%, see
#   full_split_mnist.py).
#
# Method:
#   - Matched backpropagation MLP (784 -> 256 -> 10)
#   - SI importance accumulated online along the optimisation path
#   - Penalty strength lambda swept over {0, 0.1, 1, 10, 100}
#   - Class-incremental Split-MNIST (five binary tasks in sequence)
#
# Outputs:
#   - Console accuracy for each lambda (averaged over seeds)
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 baselines_si.py
#   (requires the MNIST archive at /tmp/mnist.npz)
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
#   Valencia Niño, C.H. Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods.
# =============================================================================
import numpy as np, statistics as st

d = np.load('/tmp/mnist.npz')
Xtr = d['Xtr'].astype(np.float64) / 255.0; ytr = d['ytr']
Xte = d['Xte'].astype(np.float64) / 255.0; yte = d['yte']
tanh = np.tanh; K = 10


def cdata(c, n, rng):
    """Return n random training images of class c."""
    idx = np.where(ytr == c)[0]
    return Xtr[idx[rng.permutation(len(idx))[:n]]]


def onehot(labs):
    """One-hot targets in {-1, +1} over the K classes."""
    Y = -np.ones((len(labs), K))
    for i in range(len(labs)):
        Y[i, int(labs[i])] = 1.0
    return Y


def evalacc(P, mu):
    """Class-incremental accuracy (argmax over all K outputs) on the test set."""
    W1, b1, W2, b2 = P; a = []
    for c in range(K):
        idx = np.where(yte == c)[0][:400]
        o = tanh((Xte[idx] - mu) @ W1.T + b1) @ W2.T + b2; a.append(np.mean(np.argmax(o, 1) == c))
    return float(np.mean(a))


def run_si(seed, lam, H=256, EP=35, lr=0.1, bs=128, nper=2000, xi=0.1):
    """Run one class-incremental Split-MNIST sequence with SI; return final accuracy."""
    rng = np.random.default_rng(seed); Xc = {c: cdata(c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0); Xc = {c: Xc[c] - mu for c in range(K)}
    P = [rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
         rng.normal(0, 1, (K, H)) * np.sqrt(1 / H), np.zeros((1, K))]
    Omega = [np.zeros_like(p) for p in P]; ref = [p.copy() for p in P]
    for t in range(5):
        ca, cb = 2 * t, 2 * t + 1
        X = np.vstack([Xc[ca], Xc[cb]]); Y = onehot([ca] * len(Xc[ca]) + [cb] * len(Xc[cb]))
        w = [np.zeros_like(p) for p in P]; start = [p.copy() for p in P]; n = len(X)
        for ep in range(EP):
            pm = rng.permutation(n)
            for i in range(0, n, bs):
                xb = X[pm[i:i + bs]]; yb = Y[pm[i:i + bs]]; nb = len(xb)
                a1 = xb @ P[0].T + P[1]; h1 = tanh(a1); out = h1 @ P[2].T + P[3]; dout = (out - yb) / nb
                da = (dout @ P[2]) * (1 - h1 ** 2)
                g = [da.T @ xb, da.sum(0, keepdims=True), dout.T @ h1, dout.sum(0, keepdims=True)]
                for k in range(4):
                    tot = g[k] + lam * Omega[k] * (P[k] - ref[k])     # gradient with the SI penalty
                    dp = -lr * tot; w[k] += -g[k] * dp; P[k] += dp     # path integral with the task gradient
        for k in range(4):
            Omega[k] += w[k] / ((P[k] - start[k]) ** 2 + xi)
        ref = [p.copy() for p in P]
    return evalacc(P, mu)


if __name__ == "__main__":
    for lam in [0, 0.1, 1, 10, 100]:
        print(f"BACKPROP+SI lam={lam:<5} acc = {st.mean([run_si(s, lam) for s in [0, 1]]):.1%}")
