# Backpropagation-Free Continual Learning
### Reproducible Benchmark Code

**Paper:** Valencia Niño, C.H. *Backpropagation-Free Continual Learning: A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges.* MDPI Analytics (submitted).

**Author:** Cesar Hernando Valencia Niño  
Facultad de Ingeniería de Telecomunicaciones, Universidad Santo Tomás, Seccional Bucaramanga, Colombia  
📧 cesar.valencia@ustabuca.edu.co · ORCID: [0000-0001-6077-6458](https://orcid.org/0000-0001-6077-6458)

---

## Overview

This repository contains all scripts used in the empirical case study of the paper. The experiments are diagnostic and auditable: they verify theoretical claims about Local Equilibrium Learning (LEL) and provide an honest, reproducible baseline for backpropagation-free continual learning. No automatic differentiation package is used in the LEL experiments — every error computation, state relaxation, and weight update is explicit.

The central finding is that local generative dreaming converts class-incremental Split-MNIST performance from chance level (≈19%, shared with EWC and SI) to **86.8 ± 0.2%**, reaching 96% of the joint-training ceiling, without storing any raw data of past tasks and without backpropagation.

---

## Requirements

**Python scripts:** Python 3, NumPy only. No deep learning framework required.

```bash
pip install numpy
```

**MATLAB scripts:** Base MATLAB (R2023b or later recommended). No toolboxes required.

**Datasets:** MNIST, Fashion-MNIST, and CIFAR-10 are downloaded automatically by script or expected as a NumPy archive at `/tmp/mnist.npz`. See each script's *How to run* header for details.

---

## Repository structure

Scripts are grouped by what they test, matching the paper's section structure.

### 1 · Theoretical validation (Section 4 of the paper)

| Script | What it does |
|--------|--------------|
| `equilibrium_vs_backprop.py` | Compares LEL against a matched backpropagation MLP on a nonlinear two-moons task. Shows parity in accuracy without a backward pass. |
| `theorem_validation.py` | Numerically validates Theorem 2: the cosine between the LEL update and the backpropagation gradient approaches 1 as the nudge β → 0 (nudged regime), while clamping saturates below 1 (LEL descends its own free energy, not the task loss). |
| `beta_sweep_theorem2.py` | Sweeps the nudge parameter β and records (β, cosine) pairs, producing the figure that visualises Theorem 2. |

---

### 2 · Headline continual-learning results (Section 6)

| Script | What it does |
|--------|--------------|
| `full_split_mnist.py` | **Core module.** Full class-incremental Split-MNIST (5 binary tasks in sequence). Compares three conditions: vanilla LEL, noise replay, and local generative dreaming. Exports the LEL building blocks reused by all other scripts. |
| `reinforce.py` | Strengthens the headline result: reports mean ± std over 5 seeds in both class-incremental and task-incremental settings, plus an ablation of the dream budget *n_d*. |
| `fashion_continual.py` | Applies the same LEL pipeline to Fashion-MNIST as a generality test beyond MNIST. |
| `cifar_continual.py` | Stress test on raw-pixel CIFAR-10 (3072 dimensions). Reports the joint-training ceiling alongside dreaming performance, documenting the base-classifier limitation honestly. |
| `forgetting_curve.py` | Tracks accuracy on the first task and average accuracy over the full task sequence as tasks arrive, contrasting vanilla LEL with generative dreaming. |
| `per_task_accuracy.py` | After all five Split-MNIST tasks, reports per-task accuracy for each condition. Shows that vanilla LEL retains only the most recent task; dreaming retains all. |

---

### 3 · Manifold analysis and generator ablations

| Script | What it does |
|--------|--------------|
| `manifold_dream_lel.py` | Two-moons manifold embedded in ℝ³⁰. Measures the orthogonal distance of noise samples versus local generator samples from the data manifold, explaining geometrically why noise replay fails and dreaming works. |
| `mnist_continual_dream.py` | Scales the two-task dreaming result to real MNIST images. Compares noise and local linear generator (Oja/PCA subspace) across multiple seeds. |
| `mnist_nonlinear_dream.py` | Replaces the linear generator with a nonlinear predictive-coding autoencoder trained with local rules. Compares vanilla, noise, linear, and nonlinear generator conditions. |
| `energy_dream_lel.py` | Attempts Langevin-dynamics energy sampling (input-confined) as an alternative generator. Documents a negative result: the approach is stable only when confined to a data-scale ball; without confinement it diverges. |
| `generator_capacity_ablation.py` | Sweeps the subspace dimension *k* of the linear generator and measures class-incremental retention. Shows that retention is bounded by generator fidelity, not by the continual mechanism. |
| `close_gap.py` | Closes the gap between dreaming and the oracle bound using two levers: increased dream budget *n_d* and Gaussian mixture latent sampling instead of a single Gaussian. |

---

### 4 · Baselines (negative results)

| Script | What it does |
|--------|--------------|
| `baselines_ewc.py` | EWC (Elastic Weight Consolidation) on a matched backpropagation MLP. Confirms that EWC does not mitigate forgetting in the class-incremental setting for any value of λ. |
| `baselines_si.py` | Synaptic Intelligence on the same architecture. Confirms failure in class-incremental learning; strong regularisation collapses to chance. |

---

### 5 · Depth barrier

| Script | What it does |
|--------|--------------|
| `deep_lel_testbed.py` | Trains deep LEL networks (up to 10 hidden layers) on a binary MNIST task and logs per-layer equilibrium errors. Shows that errors become exponentially imbalanced: large near the output, vanishing near the input. |
| `depth_accuracy.py` | Quantifies the effect on accuracy as depth grows, complementing the per-layer error figure by showing the practical consequence of the error-imbalance barrier. |

---

### 6 · Efficiency and hardware projection

| Script | What it does |
|--------|--------------|
| `measure_efficiency.py` | Measures wall-clock time and activation memory of LEL versus backpropagation on CPU. Reports the ~7.5× slowdown and the O(width) vs O(depth × width) memory contrast. |
| `energy_estimate.py` | Projects per-sample training energy using the per-operation figures of Horowitz (2014, 45 nm). On von Neumann hardware LEL costs more; on in-memory substrates the dominant data-movement term is eliminated, projecting a 120–1200× reduction. This is a projection, not a device measurement. |
| `t_relaxation_ablation.py` | Sweeps the number of relaxation steps *T* and measures accuracy and wall-clock time. Accuracy is nearly flat across *T*; cost grows linearly, quantifying the source of the measured slowdown. |

---

### 7 · Independent MATLAB reproduction

| Script | What it does |
|--------|--------------|
| `continual_learning_lel_vs_backprop.m` | Baseline comparison of unprotected LEL versus backpropagation on a sequential two-task problem. Reports forgetting at few epochs and at convergence, controlling for the undertraining artefact. |
| `continual_dreaming_lel.m` | **Independent reproduction** of the headline dreaming result in MATLAB (base toolboxes only, no deep learning toolbox). Confirms that generative replay eliminates forgetting across six random seeds. |

---

## Reproducing the headline result

```bash
# Install dependency
pip install numpy

# Download MNIST to /tmp/mnist.npz (see script header for instructions)
# Then run:
python full_split_mnist.py      # three conditions, single seed
python reinforce.py             # 5-seed mean ± std, task-IL and class-IL
python baselines_ewc.py         # EWC comparison
python baselines_si.py          # SI comparison
```

In MATLAB:
```matlab
continual_learning_lel_vs_backprop
continual_dreaming_lel
```

Every script prints results to the console. No files are written unless stated otherwise in the script header. Fixed random seeds and all hyperparameters are declared inside each script and match the values in Table 5 of the paper.

---

## Citation

```bibtex
@article{valencianino2026lel,
  author  = {Valencia Niño, Cesar Hernando},
  title   = {Backpropagation-Free Continual Learning:
             A Review of Backpropagation-Free Methods},
  journal = {Analytics},
  year    = {2026},
  publisher = {MDPI}
}
```

---

## License

MIT License. See individual script headers for the full declaration.
