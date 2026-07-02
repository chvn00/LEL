# File: theorem_validation.py
# Experiment: Numerical validation of Theorem 2 (LEL <-> backpropagation)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Measure the cosine between the local LEL update and the negative
#   backpropagation gradient of the loss L = 1/2||y - mu_L||^2, in a
#   784 -> 64 -> 10 network. Two regimes:
#     - With the output fully CLAMPED, the cosine saturates near 0.62: LEL
#       descends the free energy E (Theorem 1, exact), a different objective
#       from the task loss.
#     - With a NUDGE beta -> 0, the cosine -> 1.0000: LEL recovers exactly the
#       backpropagation gradient. The equivalence is a LIMIT, not the default.
#
# Method:
#   - Single fixed mini-batch (256 MNIST images)
#   - Clamped-output relaxation for several step counts T
#   - Nudged relaxation for several nudge strengths beta
#
# Outputs:
#   - Console: cosine to the backpropagation gradient for each T (clamped)
#     and for each beta (nudged)
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 theorem_validation.py
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
W1 = rng.normal(0, 1, (H, 784)) * np.sqrt(1 / 784); b1 = np.zeros((1, H))
W2 = rng.normal(0, 1, (K, H)) * np.sqrt(1 / H); b2 = np.zeros((1, K))

# backpropagation descent direction on L = 1/2||out - Y||^2
a1 = X @ W1.T + b1; h1 = tanh(a1); out = h1 @ W2.T + b2; dout = (out - Y) / len(X)
gW2 = dout.T @ h1; da = (dout @ W2) * (1 - h1 ** 2); gW1 = da.T @ X
bp = -np.concatenate([gW1.ravel(), gW2.ravel()])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def clamped_update(T, g=0.2):
    """Local update with the output fully clamped to the target (T relaxation steps)."""
    x1 = tanh(X @ W1.T + b1).copy()
    for _ in range(T):
        p1 = tanh(X @ W1.T + b1); e1 = x1 - p1; e2 = Y - (x1 @ W2.T + b2)
        x1 = x1 - g * (e1 - e2 @ W2)
    aa = X @ W1.T + b1; e1 = x1 - tanh(aa); e2 = Y - (x1 @ W2.T + b2); g1 = e1 * (1 - tanh(aa) ** 2)
    return np.concatenate([((g1.T @ X) / len(X)).ravel(), ((e2.T @ x1) / len(X)).ravel()])


def nudged_update(beta, T=200, g=0.1):
    """Local update with the output gently nudged toward the target by beta."""
    x1 = tanh(X @ W1.T + b1).copy(); x2 = (x1 @ W2.T + b2).copy()
    for _ in range(T):
        aa = X @ W1.T + b1; e1 = x1 - tanh(aa); e2 = x2 - (x1 @ W2.T + b2)
        x2 = x2 - g * (e2 + beta * (x2 - Y))      # soft nudge of the output
        x1 = x1 - g * (e1 - e2 @ W2)
    aa = X @ W1.T + b1; e1 = x1 - tanh(aa); e2 = x2 - (x1 @ W2.T + b2); g1 = e1 * (1 - tanh(aa) ** 2)
    return np.concatenate([((g1.T @ X) / len(X)).ravel(), ((e2.T @ x1) / len(X)).ravel()])


if __name__ == "__main__":
    print("CLAMPED OUTPUT (cosine LEL vs -grad backprop):")
    for T in [0, 1, 5, 20, 100]:
        print(f"  T={T:<4} -> cos = {cos(clamped_update(T), bp):.4f}")
    print("NUDGE beta->0 (cosine LEL vs -grad backprop):")
    for beta in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
        print(f"  beta={beta:<6} -> cos = {cos(nudged_update(beta), bp):.4f}")
