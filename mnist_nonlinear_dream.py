# File: mnist_nonlinear_dream.py
# Experiment: Split-MNIST dreaming with a LOCAL NON-LINEAR generator (generative LEL)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Complete the "LEL dreams from its own energy" pipeline WITHOUT backprop:
#     - LEL classifier 784 -> 128 -> 1 (local rule)
#     - per-class manifold generator = a NON-LINEAR predictive-coding autoencoder
#       z -> h=tanh(W1 z) -> x=W2 h, trained with LOCAL rules (no backprop) and
#       sampled from its latent prior (LEL in generative mode)
#   Four ways to "dream" task A ({0,1}) while learning B ({2,3}) are compared:
#   vanilla | noise | LINEAR generator (PCA/Oja) | NON-LINEAR generator (this one).
#   Metric: forgetting of A at convergence + distance of the dreams to the manifold.
#
# Method:
#   - Shared-head continual learning A={0,1} -> B={2,3}, local LEL classifier
#   - Local non-linear and linear generators, with off-manifold distance
#
# Outputs:
#   - Console table: per-seed forgetting and off-manifold distance for each variant
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 mnist_nonlinear_dream.py
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


def dtanh(a): return 1 - np.tanh(a) ** 2


# ---------------- LEL classifier ----------------
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


# ---------------- local NON-LINEAR generator (predictive-coding AE) ----------------
def g_infer(xb, G, gz, gh, Tg, pz, dz):
    """Relax the latent z and hidden h of the generator to their equilibrium."""
    W1, b1, W2, b2 = G; nb, Dx = xb.shape; z = np.zeros((nb, dz)); h = (xb @ W2) / Dx
    for _ in range(Tg):
        a1 = z @ W1.T + b1; eh = h - tanh(a1); ex = xb - (h @ W2.T + b2)
        z = z - gz * (pz * z - (eh * dtanh(a1)) @ W1); h = h - gh * (eh - ex @ W2)
    return z, h


def g_train(X, dz=24, dh=128, EP=30, lr=0.05, gz=0.05, gh=0.1, Tg=20, pz=0.1, bs=200, seed=0):
    """Train a local predictive-coding autoencoder generator on class data X."""
    rng = np.random.default_rng(seed); n, Dx = X.shape
    G = [rng.normal(0, 1, (dh, dz)) * np.sqrt(1 / dz), np.zeros((1, dh)),
         rng.normal(0, 1, (Dx, dh)) * np.sqrt(1 / dh), np.zeros((1, Dx))]
    for ep in range(EP):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]]; nb = len(xb)
            z, h = g_infer(xb, G, gz, gh, Tg, pz, dz)
            a1 = z @ G[0].T + G[1]; eh = h - tanh(a1); ex = xb - (h @ G[2].T + G[3]); gh1 = eh * dtanh(a1)
            G[0] += lr * (gh1.T @ z) / nb; G[1] += lr * gh1.sum(0, keepdims=True) / nb
            G[2] += lr * (ex.T @ h) / nb; G[3] += lr * ex.sum(0, keepdims=True) / nb
    zs, _ = g_infer(X[:1000], G, gz, gh, Tg, pz, dz)
    return G, (zs.mean(0), np.cov(zs.T) + 1e-4 * np.eye(dz))


def g_sample(G, stats, n, rng):
    """Sample n inputs from the generator by drawing latent codes from its prior."""
    W1, b1, W2, b2 = G; zm, zc = stats; z = rng.multivariate_normal(zm, zc, size=n)
    return tanh(z @ W1.T + b1) @ W2.T + b2


# ---------------- LINEAR generator (PCA/Oja) ----------------
def pca_sample(Xc, k, rng, n):
    """Sample n inputs from the principal subspace of class data Xc."""
    mc = Xc.mean(0); Z = Xc - mc; V = np.linalg.svd(Z, full_matrices=False)[2][:k].T
    S = rng.multivariate_normal(np.zeros(k), np.cov((Z @ V).T), size=n)
    return S @ V.T + mc


def get_task(X, y, cn, cp, n, rng):
    """Build a binary task (cn vs cp), targets in {-1, +1}."""
    m = (y == cn) | (y == cp); Xs, ys = X[m], y[m]; idx = rng.permutation(len(Xs))[:n]
    return Xs[idx], np.where(ys[idx] == cp, 1.0, -1.0).reshape(-1, 1), ys[idx]


def run(seed, H=128, EP=15, lr=0.1, g=0.2, T=20, nper=1000, Ndream=2000):
    rng = np.random.default_rng(seed)
    XA, YA, lA = get_task(Xtr_all, ytr_all, 0, 1, 2 * nper, rng)
    XAte, YAte, _ = get_task(Xte_all, yte_all, 0, 1, 1000, rng)
    XB, YB, _ = get_task(Xtr_all, ytr_all, 2, 3, 2 * nper, rng)
    XBte, YBte, _ = get_task(Xte_all, yte_all, 2, 3, 1000, rng)
    mu = np.vstack([XA, XB]).mean(0); XA, XAte, XB, XBte = [Z - mu for Z in (XA, XAte, XB, XBte)]
    W0 = (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
          rng.normal(0, 1, (1, H)) * np.sqrt(1 / H), np.zeros((1, 1)))
    WA = lel(W0, XA, YA, EP, lr, g, T); a1 = acc(WA, XAte, YAte)
    muA = XA.mean(0); Vd = np.linalg.svd(XA - muA, full_matrices=False)[2][:50].T
    offman = lambda Xs: float(np.linalg.norm((Xs - muA) - ((Xs - muA) @ Vd) @ Vd.T, axis=1).mean())
    X0 = XA[lA[:len(XA)] == 0]; X1 = XA[lA[:len(XA)] == 1]

    def cont(Xd):
        """Learn B with the given dreams Xd mixed in; return (forgetting, accB, off-dist)."""
        Yd = fwd(WA, Xd)
        WB = lel(WA, np.vstack([XB, Xd]), np.vstack([YB, Yd]), EP, lr, g, T)
        return a1 - acc(WB, XAte, YAte), acc(WB, XBte, YBte), offman(Xd)

    fv = a1 - acc(lel(WA, XB, YB, EP, lr, g, T), XAte, YAte)
    fn, bn, dn = cont(rng.normal(0, XA.std(), (Ndream, 784)))
    Xp = np.vstack([pca_sample(X0, 40, rng, Ndream // 2), pca_sample(X1, 40, rng, Ndream // 2)])
    fp, bp, dp = cont(Xp)
    G0, s0 = g_train(X0, seed=seed); G1, s1 = g_train(X1, seed=seed + 1)
    Xg = np.vstack([g_sample(G0, s0, Ndream // 2, rng), g_sample(G1, s1, Ndream // 2, rng)])
    fg, bg, dg = cont(Xg)
    return a1, fv, fn, dn, fp, dp, fg, bg, dg


if __name__ == "__main__":
    seeds = [1, 7, 13]
    A1, FV, FN, DN, FP, DP, FG, BG, DG = ([] for _ in range(9))
    print("=" * 78)
    print(" Split-MNIST: local NON-LINEAR generator (generative LEL, no backprop)")
    print("=" * 78)
    print(f" {'seed':<5}| {'vanilla':<8}| {'noise':<8}| {'PCA(lin) forg/d':<15}| {'NON-LIN forg/d':<14}")
    print("-" * 78)
    for s in seeds:
        a1, fv, fn, dn, fp, dp, fg, bg, dg = run(s)
        A1.append(a1); FV.append(fv); FN.append(fn); DN.append(dn); FP.append(fp); DP.append(dp); FG.append(fg); BG.append(bg); DG.append(dg)
        print(f" {s:<5}| {fv:7.1%} | {fn:7.1%} | {fp:6.1%} d={dp:3.1f} | {fg:6.1%} d={dg:3.1f}")
    print("-" * 78)
    print(f" initial acc A = {st.mean(A1):.1%}")
    print(f" VANILLA              forget = {st.mean(FV):5.1%} +/- {st.pstdev(FV):.1%}")
    print(f" NOISE                forget = {st.mean(FN):5.1%} +/- {st.pstdev(FN):.1%}   dist={st.mean(DN):.1f}")
    print(f" LINEAR GEN (PCA/Oja) forget = {st.mean(FP):5.1%} +/- {st.pstdev(FP):.1%}   dist={st.mean(DP):.1f}")
    print(f" NON-LINEAR GEN (LEL) forget = {st.mean(FG):5.1%} +/- {st.pstdev(FG):.1%}   accB={st.mean(BG):.0%} dist={st.mean(DG):.1f}")
    print("=" * 78)
