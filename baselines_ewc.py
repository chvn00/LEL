# File: baselines_ewc.py
# Experiment: EWC backpropagation baseline on class-incremental Split-MNIST
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Confirm the standard finding that Elastic Weight Consolidation (EWC) does
#   not mitigate forgetting in the class-incremental setting. Vanilla
#   backpropagation and backpropagation+EWC both reach about 19% (they predict
#   only the most recent task) for every value of lambda. This contrasts with
#   LEL plus generative dreaming (about 86.7%, see full_split_mnist.py).
#
# Method:
#   - Matched backpropagation MLP (784 -> 256 -> 10)
#   - EWC penalty weighted by the diagonal of the Fisher information
#   - Penalty strength lambda swept over {1, 10, 100, 1000}
#   - Class-incremental Split-MNIST (five binary tasks in sequence)
#
# Outputs:
#   - Console accuracy for vanilla backpropagation and for EWC at each lambda
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 baselines_ewc.py
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
EWC_LAM = 0.0   # set from the main loop


def cdata(c, n, rng):
    """Return n random training images of class c."""
    idx = np.where(ytr == c)[0]
    return Xtr[idx[rng.permutation(len(idx))[:n]]]


def onehot(labs):
    """One-hot targets in {-1, +1} over the K classes."""
    Y = -np.ones((len(labs), K))
    for i in range(len(labs)):
        Y[i, labs[i]] = 1.0
    return Y


def evalacc(W, mu):
    """Class-incremental accuracy (argmax over all K outputs) on the test set."""
    a = []
    for c in range(K):
        idx = np.where(yte == c)[0][:400]
        o = tanh((Xte[idx] - mu) @ W[0].T + W[1]) @ W[2].T + W[3]
        a.append(np.mean(np.argmax(o, 1) == c))
    return float(np.mean(a))


def train(W, X, Y, EP, lr, bs, rng, ewc=None):
    """Train the MLP by backpropagation, optionally with the EWC penalty."""
    W1, b1, W2, b2 = [w.copy() for w in W]; n = len(X)
    for ep in range(EP):
        p = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[p[i:i + bs]]; yb = Y[p[i:i + bs]]; nb = len(xb)
            a1 = xb @ W1.T + b1; h1 = tanh(a1); out = h1 @ W2.T + b2; dout = (out - yb) / nb
            dW2 = dout.T @ h1; db2 = dout.sum(0, keepdims=True)
            da = (dout @ W2) * (1 - h1 ** 2); dW1 = da.T @ xb; db1 = da.sum(0, keepdims=True)
            if ewc:
                # quadratic anchor toward each stored task, weighted by its Fisher
                for (F, Wr) in ewc:
                    dW1 += EWC_LAM * F[0] * (W1 - Wr[0]); db1 += EWC_LAM * F[1] * (b1 - Wr[1])
                    dW2 += EWC_LAM * F[2] * (W2 - Wr[2]); db2 += EWC_LAM * F[3] * (b2 - Wr[3])
            W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return (W1, b1, W2, b2)


def fisher(W, X, Y, bs):
    """Diagonal Fisher information of the current weights on (X, Y)."""
    W1, b1, W2, b2 = W; F = [np.zeros_like(w) for w in W]; cnt = 0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]; yb = Y[i:i + bs]; nb = len(xb)
        a1 = xb @ W1.T + b1; h1 = tanh(a1); out = h1 @ W2.T + b2; dout = (out - yb) / nb
        da = (dout @ W2) * (1 - h1 ** 2)
        for k, gg in enumerate([da.T @ xb, da.sum(0, keepdims=True), dout.T @ h1, dout.sum(0, keepdims=True)]):
            F[k] += gg ** 2
        cnt += 1
    return [f / cnt for f in F]


def run(mode, seed, H=256, EP=35, lr=0.1, bs=128, nper=2000):
    """Run one class-incremental Split-MNIST sequence; return final accuracy."""
    rng = np.random.default_rng(seed); Xc = {c: cdata(c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0); Xc = {c: Xc[c] - mu for c in range(K)}
    W = (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
         rng.normal(0, 1, (K, H)) * np.sqrt(1 / H), np.zeros((1, K)))
    ewc = [] if mode == 'ewc' else None
    for t in range(5):
        ca, cb = 2 * t, 2 * t + 1
        X = np.vstack([Xc[ca], Xc[cb]]); Y = onehot([ca] * len(Xc[ca]) + [cb] * len(Xc[cb]))
        W = train(W, X, Y, EP, lr, bs, rng, ewc=ewc)
        if mode == 'ewc':
            ewc.append((fisher(W, X, Y, bs), [w.copy() for w in W]))
    return evalacc(W, mu)


if __name__ == "__main__":
    print(f"BACKPROP vanilla        acc = {st.mean([run('vanilla', s) for s in [0, 1]]):.1%}")
    for lam in [1, 10, 100, 1000]:
        EWC_LAM = lam
        print(f"BACKPROP+EWC lam={lam:<5} acc = {st.mean([run('ewc', s) for s in [0, 1]]):.1%}")
