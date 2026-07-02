# File: equilibrium_vs_backprop.py
# Experiment: Can a network learn WITHOUT backpropagation? (prototype 0)
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Test whether a neural network can be trained without backpropagation of the
#   gradient, using purely LOCAL updates, and still match backpropagation. On the
#   same data and the same architecture (2 -> H -> 1) we compare:
#     (A) BACKPROP    : classic MLP, global backward gradient (the 1986 rule)
#     (B) EQUILIBRIUM : predictive-coding / free-energy network, where inference
#                       is relaxation to an equilibrium and learning is a local
#                       Hebbian rule, with no backward pass and no stored graph.
#   Everything is plain NumPy (no autodiff), so every gradient of method (B) is
#   written by hand and is demonstrably local. If (B) learns, backpropagation is
#   shown to be just ONE way of assigning credit, not a necessity.
#
# Method:
#   - Non-linear two-moons classification
#   - Shared initial weights and data for a fair comparison
#   - (A) global backward gradient; (B) relaxation to equilibrium + local rule
#
# Outputs:
#   - Console test accuracy for backpropagation and for the equilibrium method
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 equilibrium_vs_backprop.py
#
# Reproducibility:
#   Seed: 7 (NumPy default_rng)
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
#   Valencia Niño, C.H. Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods.
# =============================================================================
import numpy as np

rng = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Data: two-moons (two interleaving half-moons), not linearly separable
# --------------------------------------------------------------------------
def make_moons(n, noise=0.12):
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    Xo = np.c_[np.cos(t_out), np.sin(t_out)]
    Xi = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
    X = np.vstack([Xo, Xi])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[-np.ones(n_out), np.ones(n_in)]      # targets in {-1, +1}
    perm = rng.permutation(n)
    return X[perm], y[perm].reshape(-1, 1)


Xtr, Ytr = make_moons(800)
Xte, Yte = make_moons(400)
# standardise with the training statistics
mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

H = 32            # hidden units
EPOCHS = 400


def tanh(z):  return np.tanh(z)
def dtanh(a): return 1.0 - np.tanh(a) ** 2     # derivative as a function of the preactivation
def acc(out, Y): return float(np.mean(np.sign(out) == np.sign(Y)))


# initial weights: generated ONCE so both methods start from the SAME copy
def init():
    W1 = rng.normal(0, 1, (H, 2)) * np.sqrt(1 / 2)
    b1 = np.zeros((1, H))
    W2 = rng.normal(0, 1, (1, H)) * np.sqrt(1 / H)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


INIT = init()   # same starting point for both methods (fair comparison)


# ==========================================================================
# (A) BACKPROP -- the classic rule: global gradient flowing back through the net
# ==========================================================================
def train_backprop(lr=0.2):
    W1, b1, W2, b2 = (w.copy() for w in INIT)
    for ep in range(EPOCHS):
        a1 = Xtr @ W1.T + b1; h1 = tanh(a1)        # forward
        out = h1 @ W2.T + b2
        d_out = (out - Ytr) / len(Xtr)             # backward (global)
        dW2 = d_out.T @ h1;            db2 = d_out.sum(0, keepdims=True)
        d_a1 = (d_out @ W2) * dtanh(a1)            # <-- the backward pass
        dW1 = d_a1.T @ Xtr;            db1 = d_a1.sum(0, keepdims=True)
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    out_te = tanh(Xte @ W1.T + b1) @ W2.T + b2
    return acc(out_te, Yte)


# ==========================================================================
# (B) EQUILIBRIUM / PREDICTIVE CODING -- no backprop, LOCAL rule
# --------------------------------------------------------------------------
# Free energy:  F = 1/2||e1||^2 + 1/2||e2||^2
#   e1 = x1 - tanh(W1 x0 + b1)         (hidden-layer prediction error)
#   e2 = x2 - (W2 x1 + b2)             (output prediction error)
# INFERENCE (relaxation to equilibrium):  x1 <- x1 - g * dF/dx1
#   dF/dx1 = e1 - e2 @ W2
# LEARNING (local rule, at the equilibrium):  dW = -dF/dW
#   dW1 ~ (e1 . tanh'(a1))^T @ x0      <- local error x presynaptic activity
#   dW2 ~  e2^T @ x1                   <- local error x presynaptic activity
# No term needs information from the whole network: only from the two neurons
# that each weight connects. That is exactly what backpropagation does not do.
# ==========================================================================
def train_equilibrium(lr=0.5, gamma=0.3, T=120):
    W1, b1, W2, b2 = (w.copy() for w in INIT)
    for ep in range(EPOCHS):
        a1 = Xtr @ W1.T + b1
        pred1 = tanh(a1)
        x1 = pred1.copy()                 # warm start = feedforward prediction
        x2 = Ytr                          # output "clamped" to the target
        for _ in range(T):                # relax to equilibrium (inference)
            a1 = Xtr @ W1.T + b1
            pred1 = tanh(a1)
            e1 = x1 - pred1
            e2 = x2 - (x1 @ W2.T + b2)
            x1 = x1 - gamma * (e1 - e2 @ W2)
        # at the equilibrium -> local Hebbian update (no backward pass)
        e1 = x1 - tanh(Xtr @ W1.T + b1)
        e2 = x2 - (x1 @ W2.T + b2)
        g1 = e1 * dtanh(Xtr @ W1.T + b1)
        dW1 = g1.T @ Xtr / len(Xtr);  db1 = g1.sum(0, keepdims=True) / len(Xtr)
        dW2 = e2.T @ x1 / len(Xtr);   db2 = e2.sum(0, keepdims=True) / len(Xtr)
        W1 += lr * dW1; b1 += lr * db1
        W2 += lr * dW2; b2 += lr * db2
    # test: forward only (no targets)
    out_te = tanh(Xte @ W1.T + b1) @ W2.T + b2
    return acc(out_te, Yte)


# ==========================================================================
print("=" * 66)
print(" Can a network learn WITHOUT backpropagation?")
print("=" * 66)
print(f" Task: two-moons (non-linear) | architecture 2 -> {H} -> 1 | {EPOCHS} epochs")
print("-" * 66)
acc_bp = train_backprop()
acc_eq = train_equilibrium()
print(f" (A) BACKPROP   (global gradient, 1986)       test accuracy = {acc_bp:6.2%}")
print(f" (B) EQUILIBRIUM(local rule, no backprop)     test accuracy = {acc_eq:6.2%}")
print("-" * 66)
print(" Note: (B) never computes a backward pass and never stores the graph.")
print("       Each weight is updated only with information from the 2 neurons it connects.")
print("=" * 66)
