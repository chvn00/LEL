# File: manifold_dream_lel.py
# Experiment: The structural advantage: generative vs. noise dreaming in HIGH DIMENSION
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Setup: the data live on a 2D manifold (two-moons) embedded in R^30 by a random
#   orthonormal basis, plus a small ambient noise (a "thin" manifold). Tasks A and
#   B are the same manifold in different regions (continual learning). Two ways to
#   "dream" task A while learning B (without storing data):
#     - noise:      label N(0, sigma^2 I) in R^30 with the old model
#     - generative: learn the manifold of A with a LOCAL rule (the principal
#                   subspace, the fixed point of Oja's rule, here via SVD), sample
#                   codes and DECODE -> samples ON the manifold
#   Hypothesis (confirmed): in 2D both tie, but in high dimension noise falls OFF
#   the manifold (unreliable labels) and fails, while the generative variant falls
#   ON the manifold and removes forgetting. This is the structural advantage of LEL
#   as an energy-based model: it can dream on the manifold. Key diagnostic: the
#   distance of each dream to the true manifold (residual to the principal subspace).
#
# Method:
#   - 2D two-moons embedded in R^30, continual tasks A and B
#   - Local LEL classifier; noise and generative (PCA/Oja) dreaming
#   - Off-manifold distance reported for each dreaming strategy
#
# Outputs:
#   - Console table: per-seed forgetting and off-manifold distance for each variant
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 manifold_dream_lel.py
#
# Reproducibility:
#   Seeds: [1, 7, 13, 21, 42]
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
    Xo = np.c_[np.cos(to), np.sin(to)]; Xi = np.c_[1 - np.cos(ti), 1 - np.sin(ti) - 0.5]
    X = np.vstack([Xo, Xi]) + rng.normal(0, noise, (n, 2)) + np.array(shift)
    y = np.r_[-np.ones(no), np.ones(ni)].reshape(-1, 1)
    p = rng.permutation(n); return X[p], y[p]


tanh = np.tanh


def fwd(W, X):
    W1, b1, W2, b2 = W; return tanh(X @ W1.T + b1) @ W2.T + b2


def acc(W, X, Y): return float(np.mean(np.sign(fwd(W, X)) == np.sign(Y)))


def relax(W, X, Y, gamma, T):
    """Inference: relax the hidden states to the free-energy equilibrium."""
    W1, b1, W2, b2 = W; x1 = tanh(X @ W1.T + b1).copy()
    for _ in range(T):
        pred1 = tanh(X @ W1.T + b1)
        e1 = x1 - pred1; e2 = Y - (x1 @ W2.T + b2)
        x1 = x1 - gamma * (e1 - e2 @ W2)
    return x1


def lel(W, X, Y, EP, lr, gamma, T):
    """Local Equilibrium Learning: a fully local Hebbian rule, no backpropagation."""
    W1, b1, W2, b2 = [w.copy() for w in W]; n = len(X)
    for _ in range(EP):
        x1 = relax((W1, b1, W2, b2), X, Y, gamma, T); a1 = X @ W1.T + b1
        e1 = x1 - tanh(a1); e2 = Y - (x1 @ W2.T + b2); g1 = e1 * (1 - tanh(a1) ** 2)
        W1 += lr * (g1.T @ X) / n; b1 += lr * g1.sum(0, keepdims=True) / n
        W2 += lr * (e2.T @ x1) / n; b2 += lr * e2.sum(0, keepdims=True) / n
    return (W1, b1, W2, b2)


def run(seed, D=30, H=32, EP=1000, lr=0.05, gamma=0.2, T=40, Ndream=800):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(0, 1, (D, 2)))          # the manifold (2D subspace)
    embed = lambda L: L @ Q.T + 0.05 * rng.normal(0, 1, (L.shape[0], D))
    LA, YA = make_moons(rng, 800, shift=(-2.5, 0)); LAte, YAte = make_moons(rng, 400, shift=(-2.5, 0))
    LB, YB = make_moons(rng, 800, shift=(2.5, 0));  LBte, YBte = make_moons(rng, 400, shift=(2.5, 0))
    XA, XAte, XB, XBte = embed(LA), embed(LAte), embed(LB), embed(LBte)
    muv = np.vstack([XA, XB]).mean(0)
    XA, XAte, XB, XBte = [Z - muv for Z in (XA, XAte, XB, XBte)]
    W0 = (rng.normal(0, 1, (H, D)) * np.sqrt(1 / D), np.zeros((1, H)),
          rng.normal(0, 1, (1, H)) * np.sqrt(1 / H), np.zeros((1, 1)))
    WA = lel(W0, XA, YA, EP, lr, gamma, T); a1 = acc(WA, XAte, YAte)

    # manifold generator (principal subspace of A = fixed point of Oja's rule)
    muA = XA.mean(0); Xc = XA - muA
    V = np.linalg.svd(Xc, full_matrices=False)[2][:2].T
    Cz = np.cov((Xc @ V).T)
    offman = lambda Xs: float(np.linalg.norm((Xs - muA) - ((Xs - muA) @ V) @ V.T, axis=1).mean())

    WBv = lel(WA, XB, YB, EP, lr, gamma, T)
    fv, bv = a1 - acc(WBv, XAte, YAte), acc(WBv, XBte, YBte)

    Xn = rng.normal(0, 1.2, (Ndream, D)); Yn = fwd(WA, Xn)
    WBn = lel(WA, np.vstack([XB, Xn]), np.vstack([YB, Yn]), EP, lr, gamma, T)
    fn, bn, dn = a1 - acc(WBn, XAte, YAte), acc(WBn, XBte, YBte), offman(Xn)

    Zs = rng.multivariate_normal(np.zeros(2), Cz, size=Ndream)
    Xg = Zs @ V.T + muA; Yg = fwd(WA, Xg)
    WBg = lel(WA, np.vstack([XB, Xg]), np.vstack([YB, Yg]), EP, lr, gamma, T)
    fg, bg, dg = a1 - acc(WBg, XAte, YAte), acc(WBg, XBte, YBte), offman(Xg)
    return fv, fn, bn, dn, fg, bg, dg


if __name__ == "__main__":
    seeds = [1, 7, 13, 21, 42]
    FV, FN, BN, DN, FG, BG, DG = ([] for _ in range(7))
    print("=" * 72)
    print(" Structural advantage in HIGH DIMENSION (2D manifold in R^30)")
    print("=" * 72)
    print(f" {'seed':<5}| {'vanilla':<8}| {'NOISE forg/dist':<16}| {'GENERATIVE forg/dist':<18}")
    print("-" * 72)
    for s in seeds:
        fv, fn, bn, dn, fg, bg, dg = run(s)
        FV.append(fv); FN.append(fn); BN.append(bn); DN.append(dn); FG.append(fg); BG.append(bg); DG.append(dg)
        print(f" {s:<5}| {fv:7.1%} | {fn:6.1%}  d={dn:4.1f} | {fg:7.1%}  d={dg:4.2f}")
    print("-" * 72)
    print(f" VANILLA      forget = {st.mean(FV):5.1%} +/- {st.pstdev(FV):.1%}")
    print(f" NOISE        forget = {st.mean(FN):5.1%} +/- {st.pstdev(FN):.1%}   accB={st.mean(BN):.0%}   off_dist={st.mean(DN):.1f}")
    print(f" GENERATIVE   forget = {st.mean(FG):5.1%} +/- {st.pstdev(FG):.1%}   accB={st.mean(BG):.0%}   off_dist={st.mean(DG):.2f}")
    print("=" * 72)
    print(" In high dimension, NOISE falls off the manifold and fails; GENERATIVE")
    print(" falls on it and removes forgetting. This is the advantage specific to LEL.")
    print("=" * 72)
