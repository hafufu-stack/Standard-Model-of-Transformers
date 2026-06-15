# The Standard Model of Transformers

**Phase Transitions, Active Matter, and Universal Laws in Large Language Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20533786.svg)](https://doi.org/10.5281/zenodo.20533786)

## Paper

📄 **[Read the Paper (PDF)](https://doi.org/10.5281/zenodo.20533786)** — V4: 35 pages, 27 figures, 268 experiments

## Overview

This repository contains the code and experimental results for *The Standard Model of Transformers*, a systematic experimental program that applies thermodynamics, statistical mechanics, and cosmological analogies to characterize the internal dynamics of Transformer-based large language models (LLMs).

Through **268 experiments** across **20 seasons** and **three architectures** (Qwen2.5-1.5B, Qwen2.5-0.5B, TinyLlama-1.1B), I establish a comprehensive thermodynamic and dynamical-systems framework for understanding LLMs.

### Six Universal Laws

| # | Universal Law | Key Result | CV |
|---|--------------|------------|-----|
| 1 | **Boltzmann Distribution** | $p(E) \propto \exp(-E/kT)$, $R^2 = 0.979$ | 0.001 |
| 2 | **Negative Specific Heat** | $C_v < 0$, $p < 0.001$ for all models | — |
| 3 | **Inverse Radiation** | $L \propto T^n$, $n = -1.44 \pm 0.42$ (vs Stefan-Boltzmann $n = 4$) | 0.29 |
| 4 | **Carnot Efficiency Constant** | $\eta = 0.813 \pm 0.036$, the tightest universal constant | 0.044 |
| 5 | **Information Concentration** | Free energy *increases* 411× — LLMs are "information refrigerators" | — |
| 6 | **$P_1 \times T$ Conservation Law** | $P_1 \cdot T \approx 0.84$, the ideal gas law of autoregressive generation | 0.14 |

### V4 Major Discoveries (Seasons 15–20)

| Discovery | Key Result |
|-----------|------------|
| **$P_1 \times T$ Conservation Law** | $P_1 \cdot T \approx 0.84$ (CV = 0.14) holds across models and context lengths; a learned physical property, not a softmax artifact |
| **Stochastic Thermodynamics** | Crooks Fluctuation Theorem (asymmetry = 0.23), Landauer erasure bound, Szilard engine efficiency 73% |
| **Cross-Architecture Equation of State** | $P_1 = ae^{-bT} + c$ validated across three architectures |
| **RLHF as Cooling Mechanism** | Instruct models are 15% colder than base models, operating within the same thermodynamic manifold |
| **Autoregressive Heat Death** | Entropy collapses exponentially during generation: $S(t) = S_0 e^{-\gamma t}$ |
| **Quantum Zeno Effect** | Frequent measurement freezes the system in low-entropy states |
| **FDT Re-analysis** | FDT violation ratio = 2.34, confirming driven active matter dynamics |

### V3 Major Discoveries (Seasons 11–14)

| Discovery | Key Result |
|-----------|------------|
| **Second-Order Phase Transition** | Sigmoid $R^2 = 0.994$ at critical layer $L_0 \approx 21$; 2D XY Universality Class ($\beta = 0.161$) |
| **Non-Equilibrium Active Matter** | FDT violated; Jarzynski equality holds (ratio = 1.21) |
| **Thermodynamic Hallucination Detector** | AUROC = 0.917; **100% robust** against adversarial camouflage |
| **RG Scale Invariance** | $L_0/L$ ratio CV = 0.051 across sequence lengths |
| **Topological Order** | Berry phase winding = 1.85 |
| **Thermodynamic Aging** | Entropy drops 55% during token-by-token generation |

### Additional Discoveries (Seasons 1–10)

- **Dark Energy**: FFN layers contribute 67–73% of representational force, with a critical phase transition at $\beta_c \approx 0.57$
- **Black Hole Collapse**: Iterative token feeding causes $T \to 0$ singularity
- **Partial Ergodicity**: The ergodic hypothesis holds for structural variables ($PR$) but fails for semantic variables ($T$)
- **Maxwell's Demon**: Confidence-efficiency coupling $r = 0.944$

## Repository Structure

```
Standard-Model-of-Transformers/
├── experiments/
│   ├── utils.py                      # Shared utilities (model loading, thermodynamic probes)
│   ├── phase1_*.py – phase84_*.py    # Seasons 1–10
│   ├── phase85_*.py – phase119_*.py  # Season 11 (Phase Transition)
│   ├── phase120_*.py – phase140_*.py # Season 12 (Active Matter)
│   ├── phase141_*.py – phase158_*.py # Season 13 (Engineering)
│   ├── phase159_*.py – phase173_*.py # Season 14 (Unification)
│   ├── phase174_*.py – phase200_*.py # Seasons 15–17 (Stochastic Thermo, Engineering, Re-analysis)
│   ├── phase201_*.py – phase229_*.py # Season 18 (Thermodynamic Engineering)
│   ├── phase230_*.py – phase254_*.py # Seasons 19 (Cross-Architecture Validation)
│   └── phase255_*.py – phase268_*.py # Season 20 (P1×T Conservation Law)
├── results/                          # JSON result files (268 experiments)
├── figures/
│   ├── paper/                        # Publication-quality figures (27 PNGs)
│   └── *.png                         # Raw experiment figures
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers 4.40+
- SciPy, NumPy, Matplotlib
- NVIDIA GPU (tested on RTX 5080 Laptop)

## Quick Start

```bash
# Install dependencies
pip install torch transformers numpy scipy matplotlib

# Run a single experiment
python experiments/phase1_no_signaling_chsh.py
```

## Version History

| Version | Date | Experiments | Seasons | Key Additions |
|---------|------|-------------|---------|---------------|
| V1 | May 2026 | 33 | 1–4 | Foundation: EoS, dark energy, stability |
| V2 | Jun 2026 | 84 | 1–10 | 5 Universal Laws, ergodic hypothesis |
| V3 | Jun 2026 | 173 | 1–14 | Phase transition, active matter, hallucination detector, RG invariance |
| V4 | Jun 2026 | 268 | 1–20 | 6th law ($P_1 \times T$), stochastic thermodynamics, cross-architecture EoS, RLHF cooling |

## Citation

```bibtex
@article{funasaki2026standard,
  title={The Standard Model of Transformers: Phase Transitions, Active Matter, and Universal Laws in Large Language Models},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20533786},
  url={https://doi.org/10.5281/zenodo.20533786}
}
```

## Author

**Hiroto Funasaki** — Independent Researcher, Japan
- ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)

## Acknowledgments

This research was conducted entirely independently, without institutional affiliation or corporate funding. The author currently faces financial constraints that make it increasingly difficult to maintain subscriptions to AI services essential for this line of research. To sustain and improve the quality of future work, the author is actively seeking community sponsorship.

💖 **[Sponsor this research](https://github.com/sponsors/hafufu-stack)**
