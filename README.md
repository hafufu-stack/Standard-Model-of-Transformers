# The Standard Model of Transformers

**Five Universal Laws of Thermodynamic Computation in Large Language Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20533786.svg)](https://doi.org/10.5281/zenodo.20533786)

## Paper

📄 **[Read the Paper (PDF)](https://doi.org/10.5281/zenodo.20533786)**

## Overview

This repository contains the code and experimental results for *The Standard Model of Transformers*, a systematic experimental program that applies thermodynamics, dynamical systems theory, and cosmological analogies to characterize the internal dynamics of Transformer-based large language models (LLMs).

Through **84 experiments** across **three architectures** (Qwen2.5-1.5B, Qwen2.5-0.5B, TinyLlama-1.1B), I discover **five universal laws** governing Transformer computation:

### Five Universal Laws

| # | Universal Law | Key Result | CV |
|---|--------------|------------|-----|
| 1 | **Boltzmann Distribution** | $p(E) \propto \exp(-E/kT)$, $R^2 = 0.978$ | 0.001 |
| 2 | **Negative Specific Heat** | $C_v < 0$, $p < 0.001$ for all models | — |
| 3 | **Inverse Radiation** | $L \propto T^n$, $n = -1.44 \pm 0.42$ (vs Stefan-Boltzmann $n = 4$) | 0.29 |
| 4 | **Carnot Efficiency Constant** | $\eta = 0.813 \pm 0.036$, the tightest universal constant | 0.044 |
| 5 | **Information Concentration** | Free energy *increases* — LLMs are "information refrigerators" | — |

### Additional Discoveries

- **Dark Energy**: FFN layers contribute 67–73% of representational force, with a critical phase transition at $\beta_c \approx 0.57$
- **Black Hole Collapse**: Iterative token feeding causes $T \to 0$ singularity — a computational analogue of gravitational collapse
- **Partial Ergodicity**: The ergodic hypothesis holds for structural variables (participation ratio) but fails for semantic variables (temperature)

## Figures

The paper includes **12 publication-quality figures**. These can be regenerated from the data:

```bash
python figures/generate_paper_figures.py
```

## Repository Structure

```
Standard-Model-of-Transformers/
├── papers/
│   ├── paper_v1.tex          # V1 LaTeX source (33 experiments)
│   └── paper_v2.tex          # V2 LaTeX source (84 experiments, 5 universal laws)
├── experiments/
│   ├── utils.py              # Shared utilities (model loading, thermodynamic probes)
│   ├── phase1_*.py           # Phase 1–84 experiment scripts
│   └── ...
├── results/                  # JSON result files
├── figures/
│   ├── generate_paper_figures.py  # Script to regenerate all paper figures
│   ├── paper/                # Publication-quality figures (12 PNGs)
│   └── *.png                 # Raw experiment figures
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers 5.0+
- SciPy, NumPy, Matplotlib
- NVIDIA GPU (tested on RTX 5080 Laptop)

## Quick Start

```bash
# Install dependencies
pip install torch transformers numpy scipy matplotlib

# Run a single experiment
python experiments/phase1_no_signaling_chsh.py

# Regenerate all paper figures
python figures/generate_paper_figures.py
```

## Citation

```bibtex
@article{funasaki2026standard,
  title={The Standard Model of Transformers: Five Universal Laws of Thermodynamic Computation in Large Language Models},
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
