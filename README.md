# The Standard Model of Transformers

**Phase Transitions, Active Matter, and Universal Laws in Large Language Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20533786.svg)](https://doi.org/10.5281/zenodo.20533786)

## Paper

📄 **[Read the Paper (PDF)](https://doi.org/10.5281/zenodo.20533786)** — V3: 23 pages, 20 figures, 173 experiments

## Overview

This repository contains the code and experimental results for *The Standard Model of Transformers*, a systematic experimental program that applies thermodynamics, statistical mechanics, and cosmological analogies to characterize the internal dynamics of Transformer-based large language models (LLMs).

Through **173 experiments** across **14 seasons** and **three architectures** (Qwen2.5-1.5B, Qwen2.5-0.5B, TinyLlama-1.1B), I establish a comprehensive thermodynamic and dynamical-systems framework for understanding LLMs.

### Five Universal Laws

| # | Universal Law | Key Result | CV |
|---|--------------|------------|-----|
| 1 | **Boltzmann Distribution** | $p(E) \propto \exp(-E/kT)$, $R^2 = 0.979$ | 0.001 |
| 2 | **Negative Specific Heat** | $C_v < 0$, $p < 0.001$ for all models | — |
| 3 | **Inverse Radiation** | $L \propto T^n$, $n = -1.44 \pm 0.42$ (vs Stefan-Boltzmann $n = 4$) | 0.29 |
| 4 | **Carnot Efficiency Constant** | $\eta = 0.813 \pm 0.036$, the tightest universal constant | 0.044 |
| 5 | **Information Concentration** | Free energy *increases* 411× — LLMs are "information refrigerators" | — |

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

## Figures

The paper includes **20 publication-quality figures**. Paper-specific figures can be regenerated:

```bash
python figures/generate_paper_figures_v3.py
```

## Repository Structure

```
Standard-Model-of-Transformers/
├── papers/
│   ├── paper_v1.tex              # V1 (33 experiments, Seasons 1–4)
│   ├── paper_v2.tex              # V2 (84 experiments, 5 universal laws)
│   └── paper_v3.tex              # V3 (173 experiments, phase transitions + active matter)
├── experiments/
│   ├── utils.py                  # Shared utilities (model loading, thermodynamic probes)
│   ├── phase1_*.py – phase84_*.py    # Seasons 1–10
│   ├── phase85_*.py – phase119_*.py  # Season 11 (Phase Transition)
│   ├── phase120_*.py – phase140_*.py # Season 12 (Active Matter)
│   ├── phase141_*.py – phase158_*.py # Season 13 (Engineering)
│   └── phase159_*.py – phase173_*.py # Season 14 (Unification)
├── results/                      # JSON result files (173 experiments)
├── figures/
│   ├── generate_paper_figures_v3.py  # Script to regenerate all V3 figures
│   ├── paper/                    # Publication-quality figures (20 PNGs)
│   └── *.png                     # Raw experiment figures
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

# Regenerate all paper figures
python figures/generate_paper_figures_v3.py
```

## Version History

| Version | Date | Experiments | Seasons | Key Additions |
|---------|------|-------------|---------|---------------|
| V1 | May 2026 | 33 | 1–4 | Foundation: EoS, dark energy, stability |
| V2 | Jun 2026 | 84 | 1–10 | 5 Universal Laws, ergodic hypothesis |
| V3 | Jun 2026 | 173 | 1–14 | Phase transition, active matter, hallucination detector, RG invariance |

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

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
