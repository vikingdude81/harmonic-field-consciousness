# Figures for "A Harmonic Field Model of Consciousness"

This directory contains all figure-generation code for the paper.

## Quick Start

```bash
python generate_figures.py
```

**Requirements:** `numpy`, `matplotlib`, `networkx`

## Generated Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| Figure 1 | `fig1_harmonic_modes.png` | First three harmonic modes $\psi_1, \psi_2, \psi_3$ on a 24-node toy graph |
| Figure 2 | `fig2_mode_power_states.png` | Mode power distributions for wake, NREM unconscious, NREM dreaming, anesthesia |
| Figure 3 | `fig3_consciousness_components.png` | Five components of $C(t)$ plus combined functional across four states |
| Figure 4 | `fig4_delta_paradox.png` | Delta paradox schematic: similar delta power, different $C(t)$ |

## Paper Section Mapping

| Section | Figure | Label |
|---------|--------|-------|
| 2 (Geometry) | Fig 1 | `fig:harmonic-modes` |
| 4 (Consciousness Functional) | Fig 3 | `fig:consciousness-components` |
| 5 (Delta Paradox) | Fig 2 | `fig:mode-power-states` |
| 5 (Delta Paradox) | Fig 4 | `fig:delta-paradox` |

## Four Brain States

All figures use the same four canonical states:
- **Wake:** Broad mode distribution, high entropy
- **NREM Unconscious:** Concentrated in low-index modes, low entropy
- **NREM Dreaming:** Low-mode power present but broader distribution
- **Anesthesia:** Strongly concentrated in lowest modes

## Mathematical Notation

- $\psi_k$: Laplacian eigenmode (connectome harmonic)
- $\lambda_k$: Eigenvalue for mode $k$
- $p_k$: Normalized power in mode $k$
- $H_{\text{mode}}$: Mode entropy $= -\sum_k p_k \log p_k$
- $PR$: Participation ratio $= 1/\sum_k p_k^2$
- $R$: Phase coherence
- $\dot{S}$: Entropy production rate
- $\kappa$: Criticality index
- $C(t)$: Consciousness functional

## Notes

- All figures use fixed random seeds for reproducibility
- Saved at 300 DPI for publication quality
- Script uses only standard libraries (numpy, matplotlib, networkx)
