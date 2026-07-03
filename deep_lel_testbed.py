# File: deep_lel_testbed.py
# Experiment: Depth/error-imbalance testbed for the LEL network
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Show the depth barrier of local learning on an easy task (0 vs 1). As depth
#   grows, the per-layer equilibrium errors become exponentially imbalanced (tiny
#   near the input, larger near the output), so the deepest layers receive almost
#   no learning signal and accuracy degrades.
#
# Method:
#   - Binary task (digits 0 vs 1)
#   - Multilayer LEL trained from scratch with the local rule
#   - Depth D varied over {2, 4, 6, 8, 10}; the mean per-layer error is reported
#
# Outputs:
#   - Console: for each depth D, test accuracy and the mean |error| per layer
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 deep_lel_testbed.py
#   (requires the MNIST archive at /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: data 0, weights 1, training 0 (NumPy default_rng)
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
import numpy as np
d = np.load('/tmp/mnist.npz'); Xtr = d['Xtr'].astype(np.float64)/255.0; ytr = d['ytr']; Xte = d['Xte'].astype(np.float64)/255.0; yte = d['yte']
tanh = np.tanh


def task(X, y, n, rng):
    """Binary task: digits 0 vs 1, targets in {-1, +1}."""
    m = (y == 0) | (y == 1); Xs, ys = X[m], y[m]; idx = rng.permutation(len(Xs))[:n]
    return Xs[idx], np.where(ys[idx] == 1, 1.0, -1.0).reshape(-1, 1)


def build(D, Din, H, K, rng):
    """Build a depth-D LEL network (D weight layers)."""
    sizes = [Din] + [H]*(D-1) + [K]; W = [None]; B = [None]
    for l in range(1, D+1):
        W.append(rng.normal(0, 1, (sizes[l], sizes[l-1]))*np.sqrt(1/sizes[l-1])); B.append(np.zeros((1, sizes[l])))
    return W, B, D


def fwd_states(X, W, B, D):
    """Feedforward states (tanh hidden, linear output)."""
    x = [X]
    for l in range(1, D+1):
        a = x[-1]@W[l].T+B[l]; x.append(tanh(a) if l < D else a)
    return x


def train(W, B, D, X, Y, EP, lr, g, T, bs=128, seed=0):
    """Train the deep LEL network with the local rule; return final per-layer errors."""
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(EP):
        p = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[p[i:i+bs]]; yb = Y[p[i:i+bs]]; nb = len(xb)
            x = fwd_states(xb, W, B, D); x[D] = yb.copy()      # clamp output
            for _ in range(T):                                 # relax hidden states
                a = [None]+[x[l-1]@W[l].T+B[l] for l in range(1, D+1)]
                mu = [None]+[tanh(a[l]) if l < D else a[l] for l in range(1, D+1)]
                e = [None]+[x[l]-mu[l] for l in range(1, D+1)]
                for l in range(1, D):
                    fp = (1-tanh(a[l+1])**2) if (l+1) < D else 1.0
                    x[l] = x[l]-g*(e[l]-(fp*e[l+1])@W[l+1])
            a = [None]+[x[l-1]@W[l].T+B[l] for l in range(1, D+1)]
            mu = [None]+[tanh(a[l]) if l < D else a[l] for l in range(1, D+1)]
            e = [None]+[x[l]-mu[l] for l in range(1, D+1)]
            for l in range(1, D+1):
                fp = (1-tanh(a[l])**2) if l < D else 1.0
                gl = fp*e[l]
                W[l] = W[l]+lr*(gl.T@x[l-1])/nb; B[l] = B[l]+lr*gl.sum(0, keepdims=True)/nb
    return W, B, e


def acc(W, B, D, X, Y):
    out = fwd_states(X, W, B, D)[D]; return float(np.mean(np.sign(out) == np.sign(Y)))


rng = np.random.default_rng(0)
Xtr2, Ytr2 = task(Xtr, ytr, 1600, rng); Xte2, Yte2 = task(Xte, yte, 800, rng)
mu = Xtr2.mean(0); Xtr2 = Xtr2-mu; Xte2 = Xte2-mu
print("Easy task (0 vs 1). How LEL degrades as it gets deeper:")
print(f"{'layers D':<10}{'test acc':<12}{'per-layer error (mean|e_l|) ->'}")
for D in [2, 4, 6, 8, 10]:
    W, B, Dd = build(D, 784, 64, 1, np.random.default_rng(1))
    W, B, e = train(W, B, D, Xtr2, Ytr2, 12, 0.05, 0.1, 15, seed=0)
    a = acc(W, B, D, Xte2, Yte2)
    per = [f"{np.mean(np.abs(e[l])):.2e}" for l in range(1, D+1)]
    print(f"{D:<10}{a:<12.1%}{' '.join(per)}")
