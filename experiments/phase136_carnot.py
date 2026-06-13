# -*- coding: utf-8 -*-
"""
Phase 136: Carnot Efficiency Limit
Does eta_max approach the Carnot limit? 
eta_Carnot = 1 - T_cold/T_hot
Compare measured eta with the Carnot prediction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 136: Carnot Efficiency Limit")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # For each prompt, at each layer, compute:
    # T_hot = max entropy seen so far
    # T_cold = current entropy
    # eta_Carnot = 1 - T_cold/T_hot
    # eta_measured = our eta from earlier phases

    all_eta_carnot = [[] for _ in range(n_layers)]
    all_eta_measured = [[] for _ in range(n_layers)]
    all_T_hot = [[] for _ in range(n_layers)]
    all_T_cold = [[] for _ in range(n_layers)]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_vals.append(S if not np.isnan(S) else 0)

        for li in range(n_layers):
            T_subset = S_vals[:li+1]
            T_hot = max(T_subset) if T_subset else 0
            T_cold = S_vals[li]

            all_T_hot[li].append(T_hot)
            all_T_cold[li].append(T_cold)

            # Carnot efficiency
            if T_hot > 0.1:
                eta_carnot = 1.0 - T_cold / T_hot
            else:
                eta_carnot = 0
            all_eta_carnot[li].append(eta_carnot)

            # Measured eta (same definition as throughout)
            if len(T_subset) >= 4:
                T_cold_half = min(T_subset[len(T_subset)//2:])
                if T_hot > 0.01:
                    eta_m = 1.0 - T_cold_half / T_hot
                else:
                    eta_m = 0
            else:
                eta_m = 0
            all_eta_measured[li].append(eta_m)

    # Averages
    avg = lambda x: [np.mean(v) if v else 0 for v in x]
    eta_c = avg(all_eta_carnot)
    eta_m = avg(all_eta_measured)
    T_hot = avg(all_T_hot)
    T_cold = avg(all_T_cold)

    # Efficiency ratio: eta_measured / eta_Carnot
    eff_ratio = [em / (ec + 1e-10) for em, ec in zip(eta_m, eta_c)]

    layers = np.arange(n_layers)

    # Correlation
    r_corr, p_corr = sp_stats.pearsonr(eta_c[4:], eta_m[4:])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Eta comparison
    axes[0,0].plot(layers, eta_c, 'o-', color='#c0392b', markersize=3, linewidth=2,
                   label='$\\eta_{Carnot}$')
    axes[0,0].plot(layers, eta_m, 's-', color='#2980b9', markersize=3, linewidth=2,
                   label='$\\eta_{measured}$')
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Carnot vs Measured')
    axes[0,0].legend(fontsize=8)

    # (b) T_hot and T_cold
    axes[0,1].plot(layers, T_hot, 'o-', color='#c0392b', markersize=3, label='$T_{hot}$')
    axes[0,1].plot(layers, T_cold, 's-', color='#2980b9', markersize=3, label='$T_{cold}$')
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_ylabel('$T$ (entropy)')
    axes[0,1].set_title('(b) Hot and Cold Reservoirs')
    axes[0,1].legend()

    # (c) Efficiency ratio
    axes[0,2].plot(layers[4:], eff_ratio[4:], 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=1, color='black', linestyle='--', label='Carnot limit')
    axes[0,2].set_ylabel('$\\eta_{meas} / \\eta_{Carnot}$')
    axes[0,2].set_title('(c) Efficiency Ratio (1 = Carnot limit)')
    axes[0,2].legend()

    # (d) Scatter
    axes[1,0].scatter(eta_c[4:], eta_m[4:], c=layers[4:], cmap='coolwarm',
                      s=60, edgecolors='black')
    xr = np.linspace(0, max(eta_c), 100)
    axes[1,0].plot(xr, xr, 'k--', alpha=0.5, label='$\\eta_m = \\eta_C$')
    axes[1,0].set_xlabel('$\\eta_{Carnot}$')
    axes[1,0].set_ylabel('$\\eta_{measured}$')
    axes[1,0].set_title(f'(d) Correlation ($r={r_corr:.3f}$)')
    axes[1,0].legend()

    # (e) Deficit from Carnot
    deficit = np.array(eta_c) - np.array(eta_m)
    colors_d = ['#27ae60' if d > 0 else '#c0392b' for d in deficit[4:]]
    axes[1,1].bar(layers[4:], deficit[4:], color=colors_d, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('$\\eta_C - \\eta_m$ (Carnot deficit)')
    axes[1,1].set_title('(e) Thermodynamic Inefficiency')

    # (f) Summary
    final_ratio = eff_ratio[-1] if eff_ratio else 0
    max_ratio = max(eff_ratio[4:]) if len(eff_ratio) > 4 else 0
    summary = (
        f"Carnot Efficiency Analysis\n\n"
        f"Correlation: r={r_corr:.3f}\n"
        f"Final eta_C: {eta_c[-1]:.3f}\n"
        f"Final eta_m: {eta_m[-1]:.3f}\n"
        f"Final ratio: {final_ratio:.3f}\n\n"
        f"Max ratio: {max_ratio:.3f} (at L{np.argmax(eff_ratio[4:])+4})\n\n"
        f"{'APPROACHES' if max_ratio > 0.8 else 'BELOW'} Carnot limit\n"
        f"Deficit = {np.mean(deficit[4:]):.3f} avg"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 136: Carnot Efficiency Limit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase136_carnot')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Carnot correlation: r={r_corr:.3f}")
    print(f"Final ratio: {final_ratio:.3f}")
    print(f"Max ratio: {max_ratio:.3f}")
    print(f"{'='*70}")

    save_results('phase136_carnot', {
        'experiment': 'Carnot Efficiency Limit',
        'eta_carnot': eta_c,
        'eta_measured': eta_m,
        'eff_ratio': eff_ratio,
        'summary': {
            'correlation': float(r_corr),
            'final_ratio': float(final_ratio),
            'max_ratio': float(max_ratio),
        }
    })


if __name__ == '__main__':
    main()
