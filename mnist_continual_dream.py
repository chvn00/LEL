# File: mnist_continual_dream.py
# Experiment: Split-MNIST dreaming on REAL data (generative vs. noise vs. vanilla)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Scale the toy result to real images. Continual learning with a shared head:
#   task A = MNIST {0,1} -> task B = MNIST {2,3} (without revisiting A). Local LEL
#   classifier 784 -> 128 -> 1 (local rule, no backprop), targets in {-1, +1}.
#   Two ways to "dream" A while learning B (without storing images of A):
#     - noise:      N(0, sigma^2) in R^784, labelled by the old model
#     - generative: a per-class manifold generator learned with a LOCAL rule
#                   (principal subspace = fixed point of Oja's rule, via SVD);
#                   sample codes and DECODE -> pseudo-digits
#   Hypothesis: in R^784 noise falls far from the digit manifold (unreliable
#   labels) and fails; the generative variant falls on the manifold and curbs
#   forgetting. Diagnostic: mean distance of the dreams to the manifold of A
#   (residual to its 50-D principal subspace).
#
# Method:
#   - Shared-head continual learning A={0,1} -> B={2,3}, local LEL classifier
#   - Noise and generative (PCA/Oja) dreaming, with off-manifold distance
#
# Outputs:
#   - Console table: per-seed forgetting and off-manifold distance for each variant
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 mnist_continual_dream.py
#   (requires the MNIST archive at /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: [1, 7, 13]
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
import numpy as np, statistics as st

D = np.load('/tmp/mnist.npz')
Xtr_all = D['Xtr'].astype(np.float64) / 255.0; ytr_all = D['ytr']
Xte_all = D['Xte'].astype(np.float64) / 255.0; yte_all = D['yte']

tanh = np.tanh


def fwd(W, X): W1, b1, W2, b2 = W; return tanh(X @ W1.T + b1) @ W2.T + b2


def acc(W, X, Y): return float(np.mean(np.sign(fwd(W, X)) == np.sign(Y)))


def relax(W, X, Y, g, T):
    """Inference: relax the hidden states to the free-energy equilibrium."""
    W1, b1, W2, b2 = W; x1 = tanh(X @ W1.T + b1).copy()
    for _ in range(T):
        p1 = tanh(X @ W1.T + b1); e1 = x1 - p1; e2 = Y - (x1 @ W2.T + b2)
        x1 = x1 - g * (e1 - e2 @ W2)
    return x1


def lel(W, X, Y, EP, lr, g, T, bs=200, seed=1):
    """Local Equilibrium Learning: relaxation + local Hebbian update, no backprop."""
    W1, b1, W2, b2 = [w.copy() for w in W]; n = len(X); rng = np.random.default_rng(seed)
    for ep in range(EP):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]; xb = X[j]; yb = Y[j]; nb = len(xb)
            x1 = relax((W1, b1, W2, b2), xb, yb, g, T); a1 = xb @ W1.T + b1
            e1 = x1 - tanh(a1); e2 = yb - (x1 @ W2.T + b2); g1 = e1 * (1 - tanh(a1) ** 2)
            W1 += lr * (g1.T @ xb) / nb; b1 += lr * g1.sum(0, keepdims=True) / nb
            W2 += lr * (e2.T @ x1) / nb; b2 += lr * e2.sum(0, keepdims=True) / nb
    return (W1, b1, W2, b2)


def get_task(X, y, c_neg, c_pos, n, rng):
    """Build a binary task (c_neg vs c_pos), targets in {-1, +1}."""
    m = (y == c_neg) | (y == c_pos); Xs, ys = X[m], y[m]
    idx = rng.permutation(len(Xs))[:n]
    return Xs[idx], np.where(ys[idx] == c_pos, 1.0, -1.0).reshape(-1, 1), ys[idx]


def pca_generator(Xc, k, rng, n_samp):
    """Manifold generator for one class: principal subspace (Oja/SVD) + Gaussian codes."""
    mc = Xc.mean(0); Z = Xc - mc
    V = np.linalg.svd(Z, full_matrices=False)[2][:k].T
    Cz = np.cov((Z @ V).T)
    S = rng.multivariate_normal(np.zeros(k), Cz, size=n_samp)
    return S @ V.T + mc


def run(seed, H=128, EP=15, lr=0.1, g=0.2, T=20, nper=1000, Ndream=2000, k=40):
    rng = np.random.default_rng(seed)
    XA, YA, lA = get_task(Xtr_all, ytr_all, 0, 1, 2 * nper, rng)
    XAte, YAte, _ = get_task(Xte_all, yte_all, 0, 1, 1000, rng)
    XB, YB, _ = get_task(Xtr_all, ytr_all, 2, 3, 2 * nper, rng)
    XBte, YBte, _ = get_task(Xte_all, yte_all, 2, 3, 1000, rng)
    mu = np.vstack([XA, XB]).mean(0)
    XA, XAte, XB, XBte = [Z - mu for Z in (XA, XAte, XB, XBte)]

    W0 = (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
          rng.normal(0, 1, (1, H)) * np.sqrt(1 / H), np.zeros((1, 1)))
    WA = lel(W0, XA, YA, EP, lr, g, T); a1 = acc(WA, XAte, YAte)

    # subspace of the manifold of A (for the distance diagnostic)
    muA = XA.mean(0); Vd = np.linalg.svd(XA - muA, full_matrices=False)[2][:50].T
    offman = lambda Xs: float(np.linalg.norm((Xs - muA) - ((Xs - muA) @ Vd) @ Vd.T, axis=1).mean())

    # VANILLA
    WBv = lel(WA, XB, YB, EP, lr, g, T); fv, bv = a1 - acc(WBv, XAte, YAte), acc(WBv, XBte, YBte)

    # NOISE
    sig = XA.std()
    Xn = rng.normal(0, sig, (Ndream, 784)); Yn = fwd(WA, Xn)
    WBn = lel(WA, np.vstack([XB, Xn]), np.vstack([YB, Yn]), EP, lr, g, T)
    fn, bn, dn = a1 - acc(WBn, XAte, YAte), acc(WBn, XBte, YBte), offman(Xn)

    # GENERATIVE (per class: dream 0s and 1s)
    X0 = XA[(lA[:len(XA)] == 0)]; X1 = XA[(lA[:len(XA)] == 1)]
    Xg = np.vstack([pca_generator(X0, k, rng, Ndream // 2),
                    pca_generator(X1, k, rng, Ndream // 2)])
    Yg = fwd(WA, Xg)
    WBg = lel(WA, np.vstack([XB, Xg]), np.vstack([YB, Yg]), EP, lr, g, T)
    fg, bg, dg = a1 - acc(WBg, XAte, YAte), acc(WBg, XBte, YBte), offman(Xg)
    return a1, fv, fn, bn, dn, fg, bg, dg, offman(XA)


if __name__ == "__main__":
    seeds = [1, 7, 13]
    A1, FV, FN, BN, DN, FG, BG, DG, DR = ([] for _ in range(9))
    print("=" * 76)
    print(" Split-MNIST  (A={0,1} -> B={2,3}, shared head, LEL no backprop)")
    print("=" * 76)
    print(f" {'seed':<5}| {'accA0':<6}| {'vanilla':<8}| {'NOISE forg/dist':<16}| {'GENERATIVE forg/dist':<18}")
    print("-" * 76)
    for s in seeds:
        a1, fv, fn, bn, dn, fg, bg, dg, dr = run(s)
        A1.append(a1); FV.append(fv); FN.append(fn); BN.append(bn); DN.append(dn); FG.append(fg); BG.append(bg); DG.append(dg); DR.append(dr)
        print(f" {s:<5}| {a1:5.1%}| {fv:7.1%} | {fn:6.1%}  d={dn:4.1f} | {fg:7.1%}  d={dg:4.1f}")
    print("-" * 76)
    print(f" initial acc A = {st.mean(A1):.1%}")
    print(f" VANILLA      forget = {st.mean(FV):5.1%} +/- {st.pstdev(FV):.1%}")
    print(f" NOISE        forget = {st.mean(FN):5.1%} +/- {st.pstdev(FN):.1%}   accB={st.mean(BN):.0%}   off_dist={st.mean(DN):.1f}")
    print(f" GENERATIVE   forget = {st.mean(FG):5.1%} +/- {st.pstdev(FG):.1%}   accB={st.mean(BG):.0%}   off_dist={st.mean(DG):.1f}")
    print(f" (real data A: off-manifold distance to its own manifold = {st.mean(DR):.1f})")
    print("=" * 76)
