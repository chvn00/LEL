# File: energy_estimate.py
# Experiment: Back-of-envelope energy estimate (LEL vs. backprop, data movement)
#
# Paper:
#   Backpropagation-Free Continual Learning:
#   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
#
# Purpose:
#   Estimate the per-sample energy of one training update with the per-operation
#   figures from Sze et al. (2017, 45 nm node). On von Neumann hardware LEL costs more
#   (about 5x the MACs of backpropagation), and data movement (DRAM access)
#   dominates the budget. In an in-memory regime, where the weight is the device
#   and no data is moved, the same local computation is projected to be far
#   cheaper. This is a projection from per-operation figures, not a device
#   measurement; backpropagation cannot run in-memory because it needs a
#   transported copy of the weights.
#
# Method:
#   - Count MACs per sample for backprop and for LEL (T relaxation steps)
#   - Apply per-operation energies (MAC, SRAM access, DRAM access, in-memory MAC)
#
# Outputs:
#   - Console: per-sample energy in the SRAM and DRAM regimes, and the in-memory
#     projection (optimistic and conservative)
#   - No files are written (results are printed to the console)
#
# How to run:
#   python3 energy_estimate.py
#
# Reproducibility:
#   Deterministic (no randomness); analytical estimate only
#   Python 3
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
P = 784*256 + 256*10          # number of weights
T = 15
mac_bp  = 3*P                 # MACs/sample for backprop (forward + 2 in the backward pass)
mac_lel = (T+1)*P             # MACs/sample for LEL (T-step relaxation)
# per-operation energies (pJ)
E_MAC   = 4.6                 # digital FP32 multiply-accumulate
E_SRAM  = 5.0                 # on-chip SRAM access
E_DRAM  = 640.0               # off-chip DRAM access (dominant)
E_INMEM = 0.1                 # analog in-memory MAC (the weight is the device); fJ-pJ range
E_INMEM_cons = 1.0            # conservative variant


def e(macs, eop):
    """Energy per sample in microjoules, given a MAC count and a per-op energy (pJ)."""
    return macs*eop/1e6


print(f"Weights P={P:,} | MACs/sample: backprop={mac_bp:,}  LEL={mac_lel:,} ({mac_lel/mac_bp:.0f}x)")
print("\n--- von Neumann (GPU/CPU): each MAC pays compute + memory access ---")
print(f" Backprop (SRAM):  {e(mac_bp, E_MAC+E_SRAM):.2f} uJ/sample")
print(f" LEL      (SRAM):  {e(mac_lel,E_MAC+E_SRAM):.2f} uJ/sample  -> LEL {(mac_lel/mac_bp):.0f}x WORSE")
print(f" Backprop (DRAM):  {e(mac_bp, E_MAC+E_DRAM):.1f} uJ/sample")
print(f" LEL      (DRAM):  {e(mac_lel,E_MAC+E_DRAM):.1f} uJ/sample  -> LEL {(mac_lel/mac_bp):.0f}x WORSE")
print("\n--- PROJECTION: LEL on IN-MEMORY hardware (stationary weight, no data movement) ---")
print(" (backprop CANNOT run here: it needs weight transport)")
for ei, lbl in [(E_INMEM, 'optimistic 0.1pJ'), (E_INMEM_cons, 'conservative 1pJ')]:
    lel_im = e(mac_lel, ei)
    bp_vn  = e(mac_bp, E_MAC+E_DRAM)
    print(f" LEL in-memory ({lbl}): {lel_im:.3f} uJ/sample  ->  {bp_vn/lel_im:.0f}x LESS energy than backprop(DRAM)")
