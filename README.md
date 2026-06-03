# The Standard Model of Transformers

**A Thermodynamic and Dynamical-Systems Framework for Understanding Large Language Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20533786.svg)](https://doi.org/10.5281/zenodo.20533786)

## Paper

📄 **[Read the Paper (PDF)](https://doi.org/10.5281/zenodo.20533786)**

## Overview

This repository contains the code and experimental results for *The Standard Model of Transformers*, a systematic experimental program that applies thermodynamics, dynamical systems theory, and cosmological analogies to characterize the internal dynamics of Transformer-based large language models (LLMs).

Through **33 experiments** on Qwen2.5 models (0.5B and 1.5B parameters), we establish a unified physical framework:

### Key Discoveries

| # | Finding | Key Result |
|---|---------|------------|
| 1 | **Attention = Gravity** | Negative specific heat $dU/dT \approx -18$, universal across model scales |
| 2 | **FFN = Dark Energy** | 67–73% of representational force, matching cosmological dark energy (68%) |
| 3 | **Stable Attractor** | Lyapunov exponent $\lambda = -0.05$ — perturbations decay exponentially |
| 4 | **Anti-Lensing** | Information repels from high-norm tokens ($\cos = -0.15$) |
| 5 | **Thermodynamic Firewall** | Hallucination detection AUC = 0.88 via $PR \times T$ variance |
| 6 | **Critical Phase Transition** | Dark energy suppression threshold $\beta_c = 0.57$ |

### The Seven Laws

1. **Attention = Gravity.** Contracts representations, reduces entropy, drives cooling.
2. **FFN = Dark Energy.** Expands representation space; below $\beta_c = 0.57$, output collapses.
3. **Residual Stream = Spacetime.** High inertia, stable attractor, anti-lensing fabric.
4. **$|dU/dT| \approx 18$ is Universal.** Independent of model dimension ($d^{0.04}$ scaling).
5. **Dark Energy Fraction is Universal.** 67–73%, independent of model size.
6. **Phase is Input-Selected.** Thermodynamic state varies 10× across semantic categories.
7. **Tokens Evolve from Free to Bound.** Virial ratio transitions with event horizon at final layers.

## Repository Structure

```
Standard-Model-of-Transformers/
├── papers/
│   └── paper_v1.tex          # LaTeX source
├── experiments/
│   ├── utils.py              # Shared utilities
│   ├── phase1_*.py           # Phase 1–33 experiment scripts
│   ├── ...
│   └── runner*.py            # Sequential experiment runners
├── results/                  # JSON result files
└── figures/                  # Generated figures (PNG)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers 5.0+
- NVIDIA GPU (tested on RTX 5080 Laptop)

## Quick Start

```bash
# Install dependencies
pip install torch transformers numpy scipy matplotlib

# Run a single experiment
python experiments/phase1_no_signaling_chsh.py

# Run all experiments sequentially
python experiments/runner.py
```

## Citation

```bibtex
@article{funasaki2026standard,
  title={The Standard Model of Transformers: A Thermodynamic and Dynamical-Systems Framework for Understanding Large Language Models},
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
