# File: dreaming_lel.py
# Experiment: Defeating catastrophic forgetting by generative dreaming (Python)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Biological motivation: the hippocampus replays experiences during sleep to
#   consolidate memory. Here, after learning task A, the OLD model labels random
#   inputs (its "dreams") and those dreams are mixed with task B. No real data of
#   A is stored: A lives only in the weights. Across 6 seeds, at convergence,
#   forgetting of A drops from about 33% to about 1% and its variance collapses.
#   This is the Python replica of continual_dreaming_lel.m.
#
# Method:
#   - Sequential task learning (two spatially separated two-moons tasks)
#   - Local Equilibrium Learning (relaxation to equilibrium + local Hebbian rule)
#   - Generative replay: random pseudo-inputs labelled by the frozen old model
#   - Multi-seed evaluation at convergence
#
# Outputs:
#   - Console table: per-seed forgetting and accuracy on B for vanilla and for
#     dreaming, plus mean and standard deviation across seeds
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 dreaming_lel.py
#
# Reproducibility:
#   Seeds: [1, 7, 13, 21, 42, 99]
#   Python 3, NumPy
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
import numpy as np
import statistics as st


def make_moons(rng, n, noise=0.12, shift=(0, 0)):
    no, ni = n // 2, n - n // 2
    to = np.linspace(0, np.pi, no); ti = np.linspace(0, np.pi, ni)
    Xo = np.c_[np.cos(to), np.sin(to)]
    Xi = np.c_[1 - np.cos(ti), 1 - np.sin(ti) - 0.5]
    X = np.vstack([Xo, Xi]) + rng.normal(0, noise, (n, 2)) + np.array(shift)
    y = np.r_[-np.ones(no), np.ones(ni)].reshape(-1, 1)
    p = rng.permutation(n)
    return X[p], y[p]


tanh = np.tanh


def forward(W, X):
    W1, b1, W2, b2 = W
    return tanh(X @ W1.T + b1) @ W2.T + b2


def acc(W, X, Y):
    return float(np.mean(np.sign(forward(W, X)) == np.sign(Y)))


def relax(W, X, Y, gamma, T):
    """Inference = relax the hidden states to the free-energy equilibrium."""
    W1, b1, W2, b2 = W
    x1 = tanh(X @ W1.T + b1).copy()
    for _ in range(T):
        pred1 = tanh(X @ W1.T + b1)
        e1 = x1 - pred1
        e2 = Y - (x1 @ W2.T + b2)
        x1 = x1 - gamma * (e1 - e2 @ W2)
    return x1


def train_lel(W, X, Y, EP, lr, gamma, T):
    """Local Equilibrium Learning: a fully local Hebbian rule, no backpropagation."""
    W1, b1, W2, b2 = [w.copy() for w in W]; n = len(X)
    for _ in range(EP):
        x1 = relax((W1, b1, W2, b2), X, Y, gamma, T)
        a1 = X @ W1.T + b1
        e1 = x1 - tanh(a1)
        e2 = Y - (x1 @ W2.T + b2)
        g1 = e1 * (1 - tanh(a1) ** 2)
        W1 += lr * (g1.T @ X) / n;  b1 += lr * g1.sum(0, keepdims=True) / n
        W2 += lr * (e2.T @ x1) / n; b2 += lr * e2.sum(0, keepdims=True) / n
    return (W1, b1, W2, b2)


def run(seed, H=32, EP=1200, lr=0.05, gamma=0.2, T=40, Ndream=400):
    rng = np.random.default_rng(seed)
    XA, YA = make_moons(rng, 800, shift=(-2.5, 0)); XAte, YAte = make_moons(rng, 400, shift=(-2.5, 0))
    XB, YB = make_moons(rng, 800, shift=(2.5, 0));  XBte, YBte = make_moons(rng, 400, shift=(2.5, 0))
    mu = np.vstack([XA, XB]).mean(0); sd = np.vstack([XA, XB]).std(0)
    XA, XAte = (XA - mu) / sd, (XAte - mu) / sd
    XB, XBte = (XB - mu) / sd, (XBte - mu) / sd
    W0 = (rng.normal(0, 1, (H, 2)) * np.sqrt(1 / 2), np.zeros((1, H)),
          rng.normal(0, 1, (1, H)) * np.sqrt(1 / H), np.zeros((1, 1)))

    WA = train_lel(W0, XA, YA, EP, lr, gamma, T)
    a1 = acc(WA, XAte, YAte)

    # VANILLA: learn B (forgets A)
    WBv = train_lel(WA, XB, YB, EP, lr, gamma, T)
    fv, bv = a1 - acc(WBv, XAte, YAte), acc(WBv, XBte, YBte)

    # DREAM: the old model labels noise -> generative replay, no data of A
    Xd = rng.normal(0, 1.6, (Ndream, 2)); Yd = forward(WA, Xd)
    WBd = train_lel(WA, np.vstack([XB, Xd]), np.vstack([YB, Yd]), EP, lr, gamma, T)
    fd, bd = a1 - acc(WBd, XAte, YAte), acc(WBd, XBte, YBte)
    return fv, bv, fd, bd


if __name__ == "__main__":
    seeds = [1, 7, 13, 21, 42, 99]
    FV, BV, FD, BD = [], [], [], []
    print("=" * 66)
    print(" Catastrophic forgetting: VANILLA vs LEL+DREAMING (multi-seed)")
    print("=" * 66)
    print(f" {'seed':<7}| {'VANILLA forget/accB':<22}| {'DREAM forget/accB':<22}")
    print("-" * 66)
    for s in seeds:
        fv, bv, fd, bd = run(s)
        FV.append(fv); BV.append(bv); FD.append(fd); BD.append(bd)
        print(f" {s:<7}| {fv:7.2%}  {bv:7.2%}        | {fd:7.2%}  {bd:7.2%}")
    print("-" * 66)
    print(f" VANILLA       forget = {st.mean(FV):6.1%} +/- {st.pstdev(FV):.1%}   accB = {st.mean(BV):.1%}")
    print(f" LEL + DREAM   forget = {st.mean(FD):6.1%} +/- {st.pstdev(FD):.1%}   accB = {st.mean(BD):.1%}")
    print("=" * 66)
