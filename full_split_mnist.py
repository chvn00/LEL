# File: full_split_mnist.py
# Experiment: Full class-incremental Split-MNIST (5 tasks), backpropagation-free
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Standard and hard scenario: a local LEL classifier (784 -> 256 -> 10, one-hot
#   targets in {-1, +1}). Classes arrive two at a time (T0={0,1}, T1={2,3}, ...,
#   T4={8,9}) in sequence, training only on the current task. We report the mean
#   accuracy over ALL classes seen so far (argmax over the 10 outputs) and the
#   forgetting of T0. Three conditions are compared:
#     - vanilla    : no defence against forgetting
#     - noise      : replay by labelling noise with the old classifier
#     - generative : replay with local non-linear per-class generators
#                    (a predictive-coding autoencoder, sampled from its prior)
#   Everything is local: neither the classifier nor the generators use backprop.
#   This module also exports the building blocks reused by the other scripts.
#
# Method:
#   - Local Equilibrium Learning classifier (relaxation + local Hebbian rule)
#   - Per-class non-linear generative replay
#   - Class-incremental Split-MNIST, averaged over seeds
#
# Outputs:
#   - When run directly: console table of final accuracy and T0 forgetting for
#     each condition
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 full_split_mnist.py
#   (requires the MNIST archive at /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: [0, 1, 2] (averaged)
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
import numpy as np, time, statistics as st

D = np.load('/tmp/mnist.npz')
Xtr_all = D['Xtr'].astype(np.float64) / 255.0; ytr_all = D['ytr']
Xte_all = D['Xte'].astype(np.float64) / 255.0; yte_all = D['yte']
tanh = np.tanh


def dtanh(a): return 1 - np.tanh(a) ** 2


# ---------------- LEL classifier (K outputs) ----------------
def fwd(W, X):
    """Plain feedforward pass (used at test time)."""
    W1, b1, W2, b2 = W; return tanh(X @ W1.T + b1) @ W2.T + b2


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


# ---------------- local non-linear per-class generator ----------------
def g_infer(xb, G, gz, gh, Tg, pz, dz):
    """Relax the latent z and hidden h of the generator to their equilibrium."""
    W1, b1, W2, b2 = G; nb, Dx = xb.shape; z = np.zeros((nb, dz)); h = (xb @ W2) / Dx
    for _ in range(Tg):
        a1 = z @ W1.T + b1; eh = h - tanh(a1); ex = xb - (h @ W2.T + b2)
        z = z - gz * (pz * z - (eh * dtanh(a1)) @ W1); h = h - gh * (eh - ex @ W2)
    return z, h


def g_train(X, dz=48, dh=256, EP=60, lr=0.08, gz=0.1, gh=0.2, Tg=35, pz=0.02, bs=200, seed=0):
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


# ---------------- utilities ----------------
K = 10


def onehot(labels, classes):
    """One-hot targets in {-1, +1} over the K classes."""
    Y = -np.ones((len(labels), K))
    for i, c in enumerate(labels):
        Y[i, c] = 1.0
    return Y


def class_data(X, y, c, n, rng):
    """Return n random samples of class c from (X, y)."""
    idx = np.where(y == c)[0]; idx = idx[rng.permutation(len(idx))[:n]]; return X[idx]


def evaluate(W, mu, seen):
    """Class-incremental accuracy (argmax over 10) on the test set of seen classes."""
    accs = []
    for c in seen:
        idx = np.where(yte_all == c)[0][:400]
        pred = np.argmax(fwd(W, Xte_all[idx] - mu), axis=1)
        accs.append(np.mean(pred == c))
    return float(np.mean(accs))


def run(mode, seed=0, H=256, EP=35, lr=0.1, g=0.2, T=15, nper=2000, nd=1000):
    """Run one class-incremental Split-MNIST sequence under the given mode."""
    rng = np.random.default_rng(seed)
    Xc = {c: class_data(Xtr_all, ytr_all, c, nper, rng) for c in range(K)}
    mu = np.vstack(list(Xc.values())).mean(0)
    Xc = {c: Xc[c] - mu for c in range(K)}
    gens = {}
    if mode == 'generative':
        for c in range(K):
            gens[c] = g_train(Xc[c], seed=seed + c)
    W = (rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784), np.zeros((1, H)),
         rng.normal(0, 1, (K, H)) * np.sqrt(1 / H), np.zeros((1, K)))
    tasks = [(2 * t, 2 * t + 1) for t in range(5)]
    seen = []; t0_initial = None
    for t, (ca, cb) in enumerate(tasks):
        Xcur = np.vstack([Xc[ca], Xc[cb]])
        Ycur = onehot([ca] * len(Xc[ca]) + [cb] * len(Xc[cb]), seen)
        old = [c for c in seen]
        if mode == 'vanilla' or not old:
            Xtr, Ytr = Xcur, Ycur
        elif mode == 'noise':
            Xn = rng.normal(0, np.vstack([Xc[ca], Xc[cb]]).std(), (nd * len(old), 784))
            pn = np.argmax(fwd(W, Xn), axis=1); Yn = onehot(pn, seen)
            Xtr, Ytr = np.vstack([Xcur, Xn]), np.vstack([Ycur, Yn])
        elif mode == 'generative':
            Xd = np.vstack([g_sample(*gens[c], nd, rng) for c in old])
            Yd = onehot([c for c in old for _ in range(nd)], seen)
            Xtr, Ytr = np.vstack([Xcur, Xd]), np.vstack([Ycur, Yd])
        W = lel(W, Xtr, Ytr, EP, lr, g, T, seed=seed + 100 + t)
        seen = sorted(set(seen) | {ca, cb})
        if t == 0:
            t0_initial = evaluate(W, mu, [0, 1])
    acc_final = evaluate(W, mu, list(range(K)))
    acc_t0_final = evaluate(W, mu, [0, 1])
    return acc_final, t0_initial, acc_t0_final


if __name__ == "__main__":
    print("=" * 70)
    print(" Full Split-MNIST (5 tasks, class-incremental, no backpropagation)")
    print(" classifier 784->256->10 | metric: mean accuracy over the 10 classes")
    print("=" * 70)
    for mode in ['vanilla', 'noise', 'generative']:
        t0 = time.time()
        AF, F0 = [], []
        for s in [0, 1, 2]:
            af, t0i, t0f = run(mode, seed=s)
            AF.append(af); F0.append(t0i - t0f)
        print(f" {mode:<11} | final acc (10 classes) = {st.mean(AF):5.1%} +/- {st.pstdev(AF):.1%}"
              f" | T0 forgetting = {st.mean(F0):5.1%} | {time.time()-t0:4.0f}s")
    print("=" * 70)
    print(" Chance = 10%. Vanilla collapses (remembers only the last task ~20%).")
    print("=" * 70)
