# File: beta_sweep_theorem2.py
# Experiment: Fine beta sweep for Theorem 2 (local update -> backprop as beta -> 0)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Visualise Theorem 2: the nudged local update converges to the
#   backpropagation gradient as the nudge beta -> 0. Produces (beta, cosine)
#   pairs for the figure in the manuscript.
#
# Method:
#   - Single fixed mini-batch (256 MNIST images), 784 -> 64 -> 10 network
#   - Nudged relaxation for a logarithmic grid of beta
#   - Cosine between the local update and the negative backprop gradient
#
# Outputs:
#   - Console: beta and the corresponding cosine
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 beta_sweep_theorem2.py
#   (requires the MNIST archive at /tmp/mnist.npz)
#
# Reproducibility:
#   Seed: 0 (NumPy default_rng)
#   Python 3, NumPy
#   Data: /tmp/mnist.npz (arrays Xtr, ytr)
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
import numpy as np

d = np.load('/tmp/mnist.npz'); Xtr = d['Xtr'].astype(np.float64) / 255.0; ytr = d['ytr']
tanh = np.tanh; K = 10; H = 64
rng = np.random.default_rng(0)
idx = rng.permutation(len(Xtr))[:256]; X = Xtr[idx] - Xtr.mean(0)
Y = -np.ones((256, K))
for i in range(256):
    Y[i, ytr[idx][i]] = 1.0
W1 = rng.normal(0, 1, (H, 784)) * np.sqrt(1/784); b1 = np.zeros((1, H))
W2 = rng.normal(0, 1, (K, H)) * np.sqrt(1/H); b2 = np.zeros((1, K))
a1 = X @ W1.T + b1; h1 = tanh(a1); out = h1 @ W2.T + b2; dout = (out - Y)/len(X)
gW2 = dout.T @ h1; da = (dout @ W2) * (1 - h1**2); gW1 = da.T @ X
bp = -np.concatenate([gW1.ravel(), gW2.ravel()])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nudged_update(beta, T=300, g=0.1):
    """Local update with the output gently nudged toward the target by beta."""
    x1 = tanh(X @ W1.T + b1).copy(); x2 = (x1 @ W2.T + b2).copy()
    for _ in range(T):
        aa = X @ W1.T + b1; e1 = x1 - tanh(aa); e2 = x2 - (x1 @ W2.T + b2)
        x2 = x2 - g * (e2 + beta * (x2 - Y))
        x1 = x1 - g * (e1 - e2 @ W2)
    aa = X @ W1.T + b1; e1 = x1 - tanh(aa); e2 = x2 - (x1 @ W2.T + b2); g1 = e1 * (1 - tanh(aa)**2)
    return np.concatenate([((g1.T @ X)/len(X)).ravel(), ((e2.T @ x1)/len(X)).ravel()])


print("beta  cosine")
for beta in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
    print(f"{beta:<6} {cos(nudged_update(beta), bp):.4f}")
