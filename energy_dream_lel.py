# File: energy_dream_lel.py
# Experiment: Energy-based dreaming (sampling from the network's own free energy F)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Instead of labelling NOISE (pseudo-rehearsal), the old model GENERATES its
#   dreams by sampling from its free energy F via Langevin dynamics, with the
#   input CONFINED to the data region (a hard prior: a ball of radius R, valid
#   because the data are standardised). The output is clamped to a random class
#   and input+hidden are relaxed; the resulting dream is self-labelled by the old
#   model. Findings (2D two-moons, multi-seed): it is STABLE when confined
#   (without confinement it DIVERGES, since the discriminative model has no real
#   prior over x); it concentrates dreams toward region A more than noise (higher
#   fracA); and it TIES noise dreaming in forgetting/accuracy, because in 2D the
#   space is tiny and noise already covers A. The expected advantage of the
#   energy-based variant is in HIGH DIMENSION (MNIST), where noise stops looking
#   like the data. Honest limitation for scaling: confinement to a ball
#   approximates the data region in 2D but not in 784D; scaling needs a real
#   generative (top-down) path so that F has minima on the data manifold.
#
# Method:
#   - Sequential two-moons tasks, local LEL classifier
#   - Three replay variants: vanilla, noise dreaming, energy-based dreaming
#
# Outputs:
#   - Console table: per-seed forgetting for each variant, with fracA and means
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 energy_dream_lel.py
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


def fwd(W, X):
    W1, b1, W2, b2 = W
    return tanh(X @ W1.T + b1) @ W2.T + b2


def acc(W, X, Y):
    return float(np.mean(np.sign(fwd(W, X)) == np.sign(Y)))


def relax(W, X, Y, gamma, T):
    """Inference: relax the hidden states to the free-energy equilibrium."""
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
        e1 = x1 - tanh(a1); e2 = Y - (x1 @ W2.T + b2)
        g1 = e1 * (1 - tanh(a1) ** 2)
        W1 += lr * (g1.T @ X) / n; b1 += lr * g1.sum(0, keepdims=True) / n
        W2 += lr * (e2.T @ x1) / n; b2 += lr * e2.sum(0, keepdims=True) / n
    return (W1, b1, W2, b2)


def energy_dream(W, rng, n, eta=0.2, K=80, R=2.5, gin=0.2, inner=8, tau=0.02):
    """Sample inputs from F via Langevin dynamics, confined to the ball of radius R."""
    W1, b1, W2, b2 = W
    x0 = rng.normal(0, 1, (n, 2))
    t = np.where(rng.random((n, 1)) < 0.5, -1.0, 1.0)      # target class (output clamp)
    x1 = np.clip(tanh(x0 @ W1.T + b1), -1.5, 1.5)
    for _ in range(K):
        for _ in range(inner):                              # relax hidden states
            a1 = x0 @ W1.T + b1
            e1 = x1 - tanh(a1); e2 = t - (x1 @ W2.T + b2)
            x1 = np.clip(x1 - gin * (e1 - e2 @ W2), -1.5, 1.5)
        a1 = x0 @ W1.T + b1; e1 = x1 - tanh(a1)
        gx0 = -(e1 * (1 - tanh(a1) ** 2)) @ W1              # dF/dx0
        x0 = x0 - eta * gx0 + np.sqrt(2 * eta * tau) * rng.normal(0, 1, x0.shape)
        nrm = np.linalg.norm(x0, axis=1, keepdims=True)     # confine (hard prior)
        x0 = x0 * np.minimum(1.0, R / (nrm + 1e-9))
    return x0, fwd(W, x0)


def run(seed, H=32, EP=1200, lr=0.05, gamma=0.2, T=40, Ndream=400):
    rng = np.random.default_rng(seed)
    XA, YA = make_moons(rng, 800, shift=(-2.5, 0)); XAte, YAte = make_moons(rng, 400, shift=(-2.5, 0))
    XB, YB = make_moons(rng, 800, shift=(2.5, 0));  XBte, YBte = make_moons(rng, 400, shift=(2.5, 0))
    mu = np.vstack([XA, XB]).mean(0); sd = np.vstack([XA, XB]).std(0)
    XA, XAte = (XA - mu) / sd, (XAte - mu) / sd
    XB, XBte = (XB - mu) / sd, (XBte - mu) / sd
    W0 = (rng.normal(0, 1, (H, 2)) * np.sqrt(1 / 2), np.zeros((1, H)),
          rng.normal(0, 1, (1, H)) * np.sqrt(1 / H), np.zeros((1, 1)))
    WA = train_lel(W0, XA, YA, EP, lr, gamma, T); a1 = acc(WA, XAte, YAte)

    WBv = train_lel(WA, XB, YB, EP, lr, gamma, T)
    fv, bv = a1 - acc(WBv, XAte, YAte), acc(WBv, XBte, YBte)

    # noise dreaming: the old model labels noise
    Xn = rng.normal(0, 1.6, (Ndream, 2)); Yn = fwd(WA, Xn)
    WBn = train_lel(WA, np.vstack([XB, Xn]), np.vstack([YB, Yn]), EP, lr, gamma, T)
    fn, bn = a1 - acc(WBn, XAte, YAte), acc(WBn, XBte, YBte)

    # energy-based dreaming: sample from F, confined to the data region
    Xd, Yd = energy_dream(WA, rng, Ndream)
    fracA = float(np.mean(Xd[:, 0] < 0))
    WBd = train_lel(WA, np.vstack([XB, Xd]), np.vstack([YB, Yd]), EP, lr, gamma, T)
    fe, be = a1 - acc(WBd, XAte, YAte), acc(WBd, XBte, YBte)
    return fv, bv, fn, bn, fe, be, fracA


if __name__ == "__main__":
    seeds = [1, 7, 13, 21, 42, 99]
    FV, BV, FN, BN, FE, BE, FR = ([] for _ in range(7))
    print("=" * 70)
    print(" Forgetting: VANILLA vs NOISE-DREAMING vs ENERGY-DREAMING")
    print("=" * 70)
    print(f" {'seed':<5}| {'vanilla':<8}| {'noise':<8}| {'energy':<11}| {'fracA':<6}")
    print("-" * 70)
    for s in seeds:
        fv, bv, fn, bn, fe, be, fr = run(s)
        FV.append(fv); BV.append(bv); FN.append(fn); BN.append(bn); FE.append(fe); BE.append(be); FR.append(fr)
        print(f" {s:<5}| {fv:7.1%} | {fn:7.1%} | {fe:9.1%}  | {fr:5.0%}")
    print("-" * 70)
    print(f" VANILLA          forget = {st.mean(FV):5.1%} +/- {st.pstdev(FV):.1%}  accB={st.mean(BV):.0%}")
    print(f" NOISE DREAMING   forget = {st.mean(FN):5.1%} +/- {st.pstdev(FN):.1%}  accB={st.mean(BN):.0%}")
    print(f" ENERGY DREAMING  forget = {st.mean(FE):5.1%} +/- {st.pstdev(FE):.1%}  accB={st.mean(BE):.0%}  fracA={st.mean(FR):.0%}")
    print("=" * 70)
    print(" In 2D, noise and energy TIE (tiny space). The advantage of energy")
    print(" dreaming (dreams on the data manifold) is decided in HIGH DIMENSION.")
    print("=" * 70)
