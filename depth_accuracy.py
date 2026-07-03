# File: depth_accuracy.py
# Experiment: Depth D vs. accuracy (single-task MNIST, multilayer LEL from scratch)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Quantify the EFFECT of the depth barrier on ACCURACY. This complements the
#   per-layer error figure (fig:depth), which shows the CAUSE. As depth grows,
#   accuracy falls, because the layers near the input receive a vanishing
#   equilibrium error and therefore barely learn.
#
# Method:
#   - Single-task 10-class MNIST, trained from scratch with the local rule
#   - Multilayer LEL of architecture 784 -> [128] x D -> 10
#   - Test accuracy measured for D in {1, 2, 3, 4, 6, 8}
#
# Outputs:
#   - Console: hidden-layer count D, test accuracy, and wall-clock time
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 depth_accuracy.py
#   (requires the MNIST archive at /tmp/mnist.npz)
#
# Reproducibility:
#   Seeds: weights 7, training 1 (NumPy default_rng)
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
import numpy as np, time

tanh = np.tanh


def dtanh(a): return 1 - np.tanh(a)**2


D = np.load('/tmp/mnist.npz')
Xtr = D['Xtr'].astype(np.float64)/255.0; ytr = D['ytr']
Xte = D['Xte'].astype(np.float64)/255.0; yte = D['yte']
K = 10
nper = 1000
idx = np.hstack([np.where(ytr == c)[0][:nper] for c in range(K)])
X = Xtr[idx]; y = ytr[idx]
mu = X.mean(0); X = X - mu
Y = -np.ones((len(y), K)); Y[np.arange(len(y)), y] = 1.0
idxte = np.hstack([np.where(yte == c)[0][:400] for c in range(K)])
Xt = Xte[idxte] - mu; yt = yte[idxte]


def init(sizes):
    """Initialise weights and biases for the given layer sizes."""
    r = np.random.default_rng(7)
    Ws = [r.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(1/sizes[i]) for i in range(len(sizes)-1)]
    bs = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
    return Ws, bs


def forward(Ws, bs, Z):
    """Plain feedforward pass with tanh hidden units and linear output."""
    h = Z
    for i in range(len(Ws)-1):
        h = tanh(h@Ws[i].T + bs[i])
    return h@Ws[-1].T + bs[-1]


def train(sizes, EP=20, lr=0.1, g=0.2, T=20, bsz=200, seed=1):
    """Train a multilayer LEL network with the local rule (no backpropagation)."""
    Ws, bs = init(sizes); L = len(Ws)  # weight matrices 0..L-1 ; states 0..L
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(EP):
        perm = rng.permutation(n)
        for i in range(0, n, bsz):
            j = perm[i:i+bsz]; xb = X[j]; yb = Y[j]; nb = len(xb)
            x = [xb]; h = xb
            for l in range(L-1):
                h = tanh(h@Ws[l].T + bs[l]); x.append(h.copy())
            x.append(yb)                       # clamped output x[L]
            for t in range(T):                  # inference relaxation
                a = [None]*(L+1); e = [None]*(L+1)
                for l in range(1, L+1):
                    a[l] = x[l-1]@Ws[l-1].T + bs[l-1]
                    mu_l = a[l] if l == L else tanh(a[l])
                    e[l] = x[l] - mu_l
                for l in range(1, L):           # update hidden states only
                    fp = np.ones_like(a[l+1]) if (l+1) == L else dtanh(a[l+1])
                    x[l] = x[l] - g*(e[l] - (fp*e[l+1])@Ws[l])
            for l in range(1, L+1):             # local weight update
                a_l = x[l-1]@Ws[l-1].T + bs[l-1]
                fp = np.ones_like(a_l) if l == L else dtanh(a_l)
                e_l = x[l] - (a_l if l == L else tanh(a_l))
                gl = fp*e_l
                Ws[l-1] += lr*(gl.T@x[l-1])/nb
                bs[l-1] += lr*gl.sum(0, keepdims=True)/nb
    return Ws, bs


print("D  acc%")
H = 128
for Dh in [1, 2, 3, 4, 6, 8]:
    sizes = [784] + [H]*Dh + [10]
    t0 = time.time()
    Ws, bs = train(sizes)
    acc = float(np.mean(np.argmax(forward(Ws, bs, Xt), axis=1) == yt))
    print(f"{Dh:<2} {acc*100:5.1f}  ({time.time()-t0:.0f}s)", flush=True)
