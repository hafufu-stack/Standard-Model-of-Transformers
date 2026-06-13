# -*- coding: utf-8 -*-
"""
Phase 68: Equipartition Theorem
In thermal equilibrium, each DOF carries kT/2 energy.
For LLM: does each hidden dimension have ~kT/2 = T/2 mean energy?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 68: Equipartition Theorem (E_dim = kT/2)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental laws of physics describe how the universe operates at every scale",
        "Machine learning algorithms can learn complex patterns from large datasets without",
        "The human brain contains approximately eighty six billion neurons connected by",
        "Evolution by natural selection explains the diversity of life on Earth through",
        "Quantum entanglement allows particles to be correlated regardless of distance",
        "The greenhouse effect traps heat in the atmosphere causing global temperatures to",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy",
        "The standard model of particle physics classifies all known elementary particles into",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(2, len(out.hidden_states) - 1):
            h = out.hidden_states[li][0, -1, :].float()
            hidden_dim = h.shape[0]

            # Energy per dimension: E_dim = h_i^2 / 2
            E_per_dim = (h ** 2) / 2
            mean_E_dim = E_per_dim.mean().item()

            # Temperature from logits
            with torch.no_grad():
                normed = model.model.norm(out.hidden_states[li][:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0

            # Equipartition: mean_E_dim should equal kT/2
            # k is Boltzmann constant (our units: k=1 or fitted)
            # So ratio = mean_E_dim / (T/2) should be constant
            kT_half = T / 2
            ratio = mean_E_dim / (kT_half + 1e-10) if kT_half > 0 else 0

            # Variance of E per dim (should be small if equipartition holds)
            E_cv = E_per_dim.std().item() / (E_per_dim.mean().item() + 1e-10)

            all_results.append({
                'layer': li,
                'mean_E_dim': float(mean_E_dim),
                'T': float(T),
                'kT_half': float(kT_half),
                'ratio': float(ratio),
                'E_cv': float(E_cv),
                'hidden_dim': hidden_dim,
            })

    # Aggregate
    layers = sorted(set(r['layer'] for r in all_results))
    mean_ratios = {l: np.mean([r['ratio'] for r in all_results if r['layer'] == l])
                   for l in layers}
    overall_ratio = np.mean([r['ratio'] for r in all_results])
    std_ratio = np.std([r['ratio'] for r in all_results])

    print(f"\n=== Equipartition Analysis ===")
    print(f"  Overall: E_dim / (T/2) = {overall_ratio:.2f} +/- {std_ratio:.2f}")
    print(f"  (ideal = constant k_eff)")
    for l in layers[::7]:
        print(f"  Layer {l}: ratio = {mean_ratios[l]:.2f}")

    # The k_eff (effective Boltzmann constant)
    k_eff = overall_ratio  # ratio = mean_E / (T/2) = k * T/2 / (T/2) = k
    is_equipartition = std_ratio / (overall_ratio + 1e-10) < 0.5  # CV < 50%

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Ratio per layer
    ratios_by_layer = [mean_ratios[l] for l in layers]
    axes[0, 0].plot(layers, ratios_by_layer, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    axes[0, 0].axhline(y=overall_ratio, color='blue', linestyle='--',
                       label=f'k_eff = {overall_ratio:.2f}')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('E_dim / (T/2)')
    axes[0, 0].set_title(f'(a) Equipartition Ratio (k_eff={overall_ratio:.2f})')
    axes[0, 0].legend()

    # (b) E_dim vs T scatter
    all_E = [r['mean_E_dim'] for r in all_results]
    all_T = [r['T'] for r in all_results]
    slope, intercept, r_val, p_val, _ = stats.linregress(all_T, all_E)
    axes[0, 1].scatter(all_T, all_E, s=10, alpha=0.3, color='#e74c3c')
    t_fit = np.linspace(min(all_T), max(all_T), 50)
    axes[0, 1].plot(t_fit, slope * t_fit + intercept, 'b--', linewidth=2,
                    label=f'slope={slope:.2f} (ideal=k/2)')
    axes[0, 1].set_xlabel('Temperature T')
    axes[0, 1].set_ylabel('Mean E per dim')
    axes[0, 1].set_title(f'(b) E vs T (R2={r_val**2:.3f})')
    axes[0, 1].legend(fontsize=8)

    # (c) Distribution of ratios
    all_ratios = [r['ratio'] for r in all_results]
    axes[0, 2].hist(all_ratios, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=overall_ratio, color='red', linewidth=2,
                       label=f'Mean={overall_ratio:.2f}')
    axes[0, 2].set_xlabel('E_dim / (T/2)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('(c) Ratio Distribution')
    axes[0, 2].legend()

    # (d) E per dimension CV (flatness)
    mean_cv = {l: np.mean([r['E_cv'] for r in all_results if r['layer'] == l])
               for l in layers}
    axes[1, 0].plot(layers, [mean_cv[l] for l in layers], 'o-',
                    color='#2ecc71', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('CV of E per dimension')
    axes[1, 0].set_title('(d) Energy Uniformity (low = equipartition)')

    # (e) E_dim and T/2 side by side
    mean_E_by_l = {l: np.mean([r['mean_E_dim'] for r in all_results if r['layer'] == l])
                   for l in layers}
    mean_kT2_by_l = {l: np.mean([r['kT_half'] for r in all_results if r['layer'] == l])
                     for l in layers}
    axes[1, 1].plot(layers, [mean_E_by_l[l] for l in layers], 'r-', linewidth=2,
                    label='Mean E_dim')
    axes[1, 1].plot(layers, [mean_kT2_by_l[l] * k_eff for l in layers], 'b--', linewidth=2,
                    label=f'k_eff * T/2 (k={k_eff:.1f})')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('(e) E_dim vs k*T/2')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    axes[1, 2].bar(['k_eff', 'E-T R^2', 'Ratio CV'],
                   [overall_ratio, r_val**2, std_ratio / (overall_ratio + 1e-10)],
                   color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 68: Equipartition Theorem (k_eff={overall_ratio:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase68_equipartition')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: k_eff={overall_ratio:.2f}, E~T R2={r_val**2:.3f}. "
          f"Equipartition {'CONFIRMED' if is_equipartition else 'NOT confirmed'} "
          f"(ratio CV={std_ratio/overall_ratio:.2f}).")
    print(f"{'='*70}")

    save_results('phase68_equipartition', {
        'experiment': 'Equipartition Theorem',
        'summary': {
            'k_eff': float(overall_ratio),
            'E_T_R2': float(r_val**2),
            'is_equipartition': bool(is_equipartition),
        }
    })


if __name__ == '__main__':
    main()
