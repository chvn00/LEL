# File: t_relaxation_ablation.py
# Experiment: Ablation of the number of relaxation steps T in the LEL classifier
#
# Paper:
#   Energy-Based Credit Assignment for Continual Learning:
#   A Review of Backpropagation-Free Methods
#
# Purpose:
#   Study the inference-compute vs. accuracy trade-off: how many local relaxation
#   steps T are needed to assign credit well. The result supports the honest
#   "local but slower" reading: accuracy is nearly flat in T while wall-clock
#   time grows roughly linearly, so the measured slowdown over backpropagation is
#   tied to large T rather than being intrinsic.
#
# Method:
#   - Single-task 10-class MNIST classifier (784 -> 256 -> 10)
#   - Fixed training budget; only the number of relaxation steps T is varied
#   - Reuses the LEL training routine from full_split_mnist.py
#
# Outputs:
#   - Console: T, test accuracy, and wall-clock time
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 t_relaxation_ablation.py
#   (requires full_split_mnist.py in the same folder and /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: data 0, weights 7, training 1 (NumPy default_rng)
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
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from full_split_mnist import Xtr_all, ytr_all, Xte_all, yte_all, lel, fwd, onehot, K

rng = np.random.default_rng(0)
# 1500 per class for training, full test (first 400 per class)
nper = 1500
idx_tr = np.hstack([np.where(ytr_all == c)[0][:nper] for c in range(K)])
Xtr = Xtr_all[idx_tr]; ytr = ytr_all[idx_tr]
mu = Xtr.mean(0); Xtr = Xtr - mu
Ytr = onehot(ytr, list(range(K)))
idx_te = np.hstack([np.where(yte_all == c)[0][:400] for c in range(K)])
Xte = Xte_all[idx_te] - mu; yte = yte_all[idx_te]

H = 256


def fresh():
    """Fresh classifier weights (same init for every T, for a fair comparison)."""
    r = np.random.default_rng(7)
    return (r.normal(0, 1, (H, 784)) * np.sqrt(1/784), np.zeros((1, H)),
            r.normal(0, 1, (K, H)) * np.sqrt(1/H), np.zeros((1, K)))


print("T  acc%   time_s")
for T in [1, 2, 4, 8, 15, 25]:
    W = fresh()
    t0 = time.time()
    W = lel(W, Xtr, Ytr, EP=20, lr=0.1, g=0.2, T=T, seed=1)
    dt = time.time() - t0
    acc = float(np.mean(np.argmax(fwd(W, Xte), axis=1) == yte))
    print(f"{T:<2} {acc*100:5.1f}  {dt:5.1f}")
