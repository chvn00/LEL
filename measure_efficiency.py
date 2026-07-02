# File: measure_efficiency.py
# Experiment: Honest efficiency comparison, LEL vs. backprop (time, FLOPs, memory)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   On von Neumann hardware (CPU/GPU) LEL is MORE expensive, because the
#   relaxation costs T forward-like passes. Its advantages are (1) lower
#   activation memory, since it stores no computation graph (O(width) vs.
#   O(depth x width)), and (2) locality, which makes it mappable to in-memory
#   and neuromorphic hardware where the dominant cost (moving data) is avoided.
#   Point (2) is a PROJECTION (not measured here); see Horowitz (2014) for the
#   per-operation energy figures.
#
# Method:
#   - Measure wall-clock time of one training run for backprop and for LEL
#   - Count forward-equivalent FLOPs per sample
#   - Compare activation memory
#
# Outputs:
#   - Console: wall-clock time, relative compute, and activation memory
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 measure_efficiency.py
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
import numpy as np, time

d = np.load('/tmp/mnist.npz'); Xtr = d['Xtr'].astype(np.float64) / 255.0; ytr = d['ytr']
tanh = np.tanh; K = 10; Din = 784; H = 256; nper = 1000; EP = 20; bs = 128; T = 15
rng = np.random.default_rng(0)


def cdata(c, n):
    """Return n random training images of class c."""
    idx = np.where(ytr == c)[0]
    return Xtr[idx[rng.permutation(len(idx))[:n]]]


Xc = [cdata(c, nper) for c in range(K)]; X = np.vstack(Xc) - np.vstack(Xc).mean(0)
Y = -np.ones((len(X), K))
for i, c in enumerate([c for c in range(K) for _ in range(nper)]):
    Y[i, c] = 1.0


def init():
    """Fresh weights for a 784 -> 256 -> 10 network."""
    return [rng.normal(0, 1, (H, Din)) * np.sqrt(1/Din), np.zeros((1, H)),
            rng.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K))]


def bp(EP):
    """One backpropagation training run (timing only)."""
    W1, b1, W2, b2 = init(); n = len(X)
    for ep in range(EP):
        p = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[p[i:i+bs]]; yb = Y[p[i:i+bs]]; nb = len(xb)
            a1 = xb@W1.T+b1; h1 = tanh(a1); out = h1@W2.T+b2; dout = (out-yb)/nb
            dW2 = dout.T@h1; da = (dout@W2)*(1-h1**2); dW1 = da.T@xb
            W2 -= 0.1*dW2; W1 -= 0.1*dW1


def lel(EP):
    """One LEL training run (timing only)."""
    W1, b1, W2, b2 = init(); n = len(X)
    for ep in range(EP):
        p = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[p[i:i+bs]]; yb = Y[p[i:i+bs]]; nb = len(xb); x1 = tanh(xb@W1.T+b1)
            for _ in range(T):
                p1 = tanh(xb@W1.T+b1); e1 = x1-p1; e2 = yb-(x1@W2.T+b2); x1 = x1-0.2*(e1-e2@W2)
            a1 = xb@W1.T+b1; e1 = x1-tanh(a1); e2 = yb-(x1@W2.T+b2); g1 = e1*(1-tanh(a1)**2)
            W1 += 0.1*(g1.T@xb)/nb; W2 += 0.1*(e2.T@x1)/nb


if __name__ == "__main__":
    t0 = time.time(); bp(EP); tb = time.time()-t0
    t0 = time.time(); lel(EP); tl = time.time()-t0
    fwd = 2*(Din*H + H*K)
    print(f"TIME ({EP} epochs, 784-256-10): Backprop={tb:.2f}s  LEL={tl:.2f}s  -> LEL {tl/tb:.1f}x slower (CPU)")
    print(f"FLOPs/sample: Backprop ~{3*fwd:,} (3x fwd) | LEL ~{(T+1)*fwd:,} ((T+1)x fwd) -> {(T+1)/3:.0f}x more compute")
    print(f"Activation memory (batch={bs}): Backprop {bs*(Din+H):,} (O(depth*width)) | LEL {bs*H:,} (O(width))")
