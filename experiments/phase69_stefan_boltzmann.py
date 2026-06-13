# -*- coding: utf-8 -*-
"""
Phase 69: Stefan-Boltzmann Radiation Law
Does luminosity (output probability mass) scale as T^4?
L = sigma * T^4 is the radiation law for blackbodies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 69: Stefan-Boltzmann Radiation Law (L ~ T^n)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration",
        "Quantum mechanics describes the behavior of particles at atomic scale",
        "The human genome contains three billion base pairs encoding all genetic",
        "Neural networks learn representations through layers of interconnected nodes",
        "Black holes form when massive stars exhaust their nuclear fuel",
        "The periodic table organizes elements by atomic number and electron",
        "Evolution by natural selection acts on heritable variation in populations",
        "Climate models simulate atmospheric dynamics using partial differential equations",
        "Photosynthesis converts light energy into chemical energy stored in glucose",
        "The standard model classifies all known elementary particles into fermions",
        "Cryptographic protocols ensure secure communication through mathematical guarantees",
        "The cosmic microwave background is remnant radiation from the early universe",
        "Machine learning discovers patterns in data without explicit programming of",
        "Protein folding determines the three dimensional structure from amino acid sequence",
        "The Turing test evaluates whether machines exhibit intelligent behavior that is",
        "General relativity describes gravity as curvature of spacetime caused by mass",
    ]

    all_T = []
    all_L = []  # luminosity = information output rate
    all_layers = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(len(out.hidden_states)):
            with torch.no_grad():
                normed = model.model.norm(out.hidden_states[li][:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()

            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T) or T < 0.01:
                continue

            # Luminosity = total probability flux to top-k tokens
            # = how "bright" the output is (concentrated probability)
            top_k_probs = torch.topk(probs, 10).values
            L = top_k_probs.sum().item()  # fraction of prob in top 10

            # Alternative: L = max probability (peak brightness)
            L_peak = probs.max().item()

            all_T.append(T)
            all_L.append(L_peak)  # use peak luminosity
            all_layers.append(li)

    all_T = np.array(all_T)
    all_L = np.array(all_L)

    # Fit L = sigma * T^n
    # Use log-log regression
    valid = (all_T > 0.1) & (all_L > 1e-6)
    log_T = np.log(all_T[valid])
    log_L = np.log(all_L[valid])

    slope, intercept, r_val, p_val, _ = stats.linregress(log_T, log_L)
    n_exponent = slope  # L ~ T^n
    sigma = np.exp(intercept)

    print(f"\n=== Stefan-Boltzmann Analysis ===")
    print(f"  L ~ T^{n_exponent:.2f} (ideal: T^4 for SB, T^(-k) for inverse)")
    print(f"  sigma = {sigma:.4f}")
    print(f"  R2 = {r_val**2:.3f}, p = {p_val:.2e}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Log-log scatter
    sc = axes[0, 0].scatter(log_T[::3], log_L[::3], c=[all_layers[i] for i, v in enumerate(valid) if v][::3],
                           cmap='viridis', s=15, alpha=0.5)
    t_fit = np.linspace(log_T.min(), log_T.max(), 50)
    axes[0, 0].plot(t_fit, slope * t_fit + intercept, 'r--', linewidth=2,
                    label=f'n={n_exponent:.2f}')
    axes[0, 0].set_xlabel('log(T)')
    axes[0, 0].set_ylabel('log(L)')
    axes[0, 0].set_title(f'(a) log-log: L~T^{n_exponent:.2f} (R2={r_val**2:.3f})')
    axes[0, 0].legend()
    plt.colorbar(sc, ax=axes[0, 0], label='Layer')

    # (b) T vs L scatter
    axes[0, 1].scatter(all_T[::3], all_L[::3], s=10, alpha=0.3, color='#e74c3c')
    t_range = np.linspace(all_T.min(), all_T.max(), 100)
    axes[0, 1].plot(t_range, sigma * t_range ** n_exponent, 'b--', linewidth=2,
                    label=f'L = {sigma:.3f} * T^{n_exponent:.2f}')
    axes[0, 1].set_xlabel('Temperature T')
    axes[0, 1].set_ylabel('Luminosity L (peak prob)')
    axes[0, 1].set_title('(b) T vs Luminosity')
    axes[0, 1].legend(fontsize=8)

    # (c) Exponent comparison with theory
    theories = {'Stefan-Boltzmann': 4, 'Wien': 3, 'Measured': n_exponent, 'Linear': 1}
    colors_t = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
    axes[0, 2].bar(list(theories.keys()), list(theories.values()),
                   color=colors_t, alpha=0.8)
    axes[0, 2].axhline(y=n_exponent, color='red', linestyle='--')
    axes[0, 2].set_ylabel('Exponent n')
    axes[0, 2].set_title('(c) Exponent Comparison')

    # (d) Residuals
    predicted = slope * log_T + intercept
    residuals = log_L - predicted
    axes[1, 0].scatter(log_T[::3], residuals[::3], s=10, alpha=0.3, color='#9b59b6')
    axes[1, 0].axhline(y=0, color='black', linewidth=1)
    axes[1, 0].set_xlabel('log(T)')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('(d) Residuals (log-log fit)')

    # (e) L profile per layer (averaged)
    layer_set = sorted(set(all_layers))
    mean_L_per_layer = {}
    for l in layer_set:
        idxs = [i for i, ll in enumerate(all_layers) if ll == l]
        mean_L_per_layer[l] = np.mean([all_L[i] for i in idxs])
    axes[1, 1].plot(list(mean_L_per_layer.keys()),
                    list(mean_L_per_layer.values()),
                    'o-', color='#f39c12', linewidth=2, markersize=3)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Mean Luminosity')
    axes[1, 1].set_title('(e) Luminosity per Layer')

    # (f) T profile per layer
    mean_T_per_layer = {}
    for l in layer_set:
        idxs = [i for i, ll in enumerate(all_layers) if ll == l]
        mean_T_per_layer[l] = np.mean([all_T[i] for i in idxs])
    axes[1, 2].plot(list(mean_T_per_layer.keys()),
                    list(mean_T_per_layer.values()),
                    'o-', color='#3498db', linewidth=2, markersize=3)
    axes[1, 2].set_xlabel('Layer')
    axes[1, 2].set_ylabel('Mean Temperature')
    axes[1, 2].set_title('(f) Temperature per Layer')

    fig.suptitle(f'Phase 69: Stefan-Boltzmann Law (L ~ T^{n_exponent:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase69_stefan_boltzmann')
    plt.close()

    is_sb = abs(n_exponent - 4) < 1.5

    print(f"\n{'='*70}")
    print(f"VERDICT: L ~ T^{n_exponent:.2f} (R2={r_val**2:.3f}). "
          f"{'Stefan-Boltzmann-like' if is_sb else f'Exponent n={n_exponent:.1f} (not T^4)'}. "
          f"sigma={sigma:.4f}.")
    print(f"{'='*70}")

    save_results('phase69_stefan_boltzmann', {
        'experiment': 'Stefan-Boltzmann Radiation Law',
        'summary': {
            'exponent': float(n_exponent),
            'sigma': float(sigma),
            'r_squared': float(r_val**2),
            'is_stefan_boltzmann': bool(is_sb),
        }
    })


if __name__ == '__main__':
    main()
