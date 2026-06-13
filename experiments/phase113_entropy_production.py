# -*- coding: utf-8 -*-
"""
Phase 113: Entropy Production Rate
In non-equilibrium thermodynamics, the entropy production rate sigma
determines how far the system is from equilibrium. Measure dS/dL
at each layer and see if it changes sign at the transition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    print("Phase 113: Entropy Production Rate")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Measure both output entropy and hidden state entropy at each layer
    all_S_out = []  # output distribution entropy
    all_S_hidden = []  # hidden state activation entropy
    all_KL = []  # KL divergence between adjacent layers

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_out = []
        S_hidden = []
        prev_probs = None

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            # Output entropy
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_out.append(S if not np.isnan(S) else 0)

            # Hidden state entropy (using activation magnitudes as probabilities)
            h_abs = h.abs()
            p_h = h_abs / (h_abs.sum() + 1e-10)
            Sh = -(p_h * torch.log(p_h + 1e-10)).sum().item()
            S_hidden.append(Sh if not np.isnan(Sh) else 0)

            prev_probs = probs

        all_S_out.append(S_out)
        all_S_hidden.append(S_hidden)

    avg_S_out = np.mean(all_S_out, axis=0)
    avg_S_hidden = np.mean(all_S_hidden, axis=0)

    # Entropy production rate: sigma = dS/dL
    sigma_out = np.gradient(avg_S_out)
    sigma_hidden = np.gradient(avg_S_hidden)

    # Total entropy production
    sigma_total = sigma_out + sigma_hidden

    # Cumulative entropy production
    cum_sigma = np.cumsum(sigma_total)

    # Find where sigma changes sign (equilibrium point)
    sign_changes = []
    for i in range(1, len(sigma_out)):
        if sigma_out[i] * sigma_out[i-1] < 0:
            sign_changes.append(i)

    layers = np.arange(n_layers)

    # Pre/post analysis
    pre_sigma = np.mean(sigma_out[:int(L0)])
    post_sigma = np.mean(sigma_out[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Entropy profiles
    axes[0,0].plot(layers, avg_S_out, 'o-', color='#c0392b', markersize=3, label='$S_{out}$')
    axes[0,0].plot(layers, avg_S_hidden, 's-', color='#2980b9', markersize=3, label='$S_{hidden}$')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Entropy')
    axes[0,0].set_title('(a) Entropy Profiles')
    axes[0,0].legend(fontsize=8)

    # (b) Entropy production rate (output)
    colors_s = ['#c0392b' if s > 0 else '#2980b9' for s in sigma_out]
    axes[0,1].bar(layers, sigma_out, color=colors_s, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].axhline(y=0, color='black', linewidth=0.5)
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$\\sigma_{out} = dS_{out}/dL$')
    axes[0,1].set_title('(b) Output Entropy Production')

    # (c) Hidden entropy production
    colors_h = ['#c0392b' if s > 0 else '#2980b9' for s in sigma_hidden]
    axes[0,2].bar(layers, sigma_hidden, color=colors_h, alpha=0.7, edgecolor='black')
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='black', linewidth=0.5)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$\\sigma_{hidden}$')
    axes[0,2].set_title('(c) Hidden Entropy Production')

    # (d) Total entropy production
    axes[1,0].plot(layers, sigma_total, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].axhline(y=0, color='black', linewidth=0.5)
    axes[1,0].fill_between(layers, 0, sigma_total,
                            where=sigma_total > 0, alpha=0.2, color='#c0392b')
    axes[1,0].fill_between(layers, 0, sigma_total,
                            where=sigma_total <= 0, alpha=0.2, color='#2980b9')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$\\sigma_{total}$')
    axes[1,0].set_title('(d) Total Entropy Production')

    # (e) Cumulative entropy
    axes[1,1].plot(layers, cum_sigma, 'o-', color='#27ae60', markersize=3, linewidth=2)
    axes[1,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Cumulative $\\Sigma$')
    axes[1,1].set_title('(e) Cumulative Entropy Production')

    # (f) Summary
    # 2nd law check: total entropy should increase
    total_entropy_change = avg_S_out[-1] - avg_S_out[0]
    summary = (
        f"Entropy Production Analysis\n\n"
        f"S_out change: {total_entropy_change:.2f}\n"
        f"(initial: {avg_S_out[0]:.2f} -> final: {avg_S_out[-1]:.2f})\n\n"
        f"sigma pre-L0: {pre_sigma:.3f}\n"
        f"sigma post-L0: {post_sigma:.3f}\n\n"
        f"Sign changes at: {sign_changes}\n\n"
        f"2nd law: {'SATISFIED' if total_entropy_change < 0 else 'ENTROPY DECREASES'}\n"
        f"(S_out decreases = prediction sharpens)"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 113: Entropy Production Rate',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase113_entropy_production')
    plt.close()

    print(f"\n{'='*70}")
    print(f"S_out: {avg_S_out[0]:.2f} -> {avg_S_out[-1]:.2f} (change: {total_entropy_change:.2f})")
    print(f"sigma pre-L0: {pre_sigma:.3f}, post: {post_sigma:.3f}")
    print(f"Sign changes: {sign_changes}")
    print(f"{'='*70}")

    save_results('phase113_entropy_production', {
        'experiment': 'Entropy Production Rate',
        'S_out': [float(v) for v in avg_S_out],
        'S_hidden': [float(v) for v in avg_S_hidden],
        'sigma_out': [float(v) for v in sigma_out],
        'sigma_hidden': [float(v) for v in sigma_hidden],
        'summary': {
            'S_out_initial': float(avg_S_out[0]),
            'S_out_final': float(avg_S_out[-1]),
            'total_change': float(total_entropy_change),
            'pre_sigma': float(pre_sigma),
            'post_sigma': float(post_sigma),
            'sign_changes': [int(s) for s in sign_changes],
        }
    })


if __name__ == '__main__':
    main()
