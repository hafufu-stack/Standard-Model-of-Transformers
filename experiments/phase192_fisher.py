# -*- coding: utf-8 -*-
"""
Phase 192: Fisher Information Landscape
=========================================
Fisher information F(l) measures the sensitivity of the output
distribution to changes at layer l. High F = critical layer.
F(l) = sum_x p_l(x) * [d log p_l(x) / dl]^2

This maps the "curvature of the statistical manifold" and identifies
which layers are doing the most statistically significant work.
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
    "Cryptographic hash functions ensure data integrity",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "Superconductors carry current with zero resistance",
    "Artificial neural networks are inspired by biological neurons",
]


def main():
    print("=" * 70)
    print("Phase 192: Fisher Information Landscape")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_fisher = []
    all_kl = []
    all_js = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get log-probabilities at each layer
        log_probs = []
        probs_all = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            p = torch.softmax(logits, dim=-1)
            lp = torch.log_softmax(logits, dim=-1)
            probs_all.append(p)
            log_probs.append(lp)

        # Fisher information at each layer transition
        # F(l) = sum_x p_l(x) * (log p_{l+1}(x) - log p_l(x))^2
        fisher_vals = []
        kl_vals = []
        js_vals = []

        for i in range(n_layers - 1):
            p = probs_all[i]
            q = probs_all[i + 1]
            lp = log_probs[i]
            lq = log_probs[i + 1]

            # Fisher information approximation
            d_log_p = lq - lp  # gradient of log-probability
            F = (p * d_log_p ** 2).sum().item()
            fisher_vals.append(F if not np.isnan(F) else 0)

            # KL divergence (forward)
            kl = (p * (lp - lq)).sum().item()
            kl_vals.append(kl if not np.isnan(kl) else 0)

            # Jensen-Shannon
            m = 0.5 * (p + q)
            lm = torch.log(m + 1e-10)
            js = 0.5 * (p * (lp - lm)).sum().item() + 0.5 * (q * (lq - lm)).sum().item()
            js_vals.append(js if not np.isnan(js) else 0)

        all_fisher.append(fisher_vals)
        all_kl.append(kl_vals)
        all_js.append(js_vals)

    fisher_mean = np.mean(all_fisher, axis=0)
    fisher_std = np.std(all_fisher, axis=0)
    kl_mean = np.mean(all_kl, axis=0)
    js_mean = np.mean(all_js, axis=0)

    layers_t = np.arange(n_layers - 1) + 0.5

    # Critical layer (peak Fisher information)
    critical_layer = np.argmax(fisher_mean)

    # Cumulative Fisher
    fisher_cumul = np.cumsum(fisher_mean)

    # Pre/post L0 analysis
    pre_fisher = np.mean(fisher_mean[:L0])
    post_fisher = np.mean(fisher_mean[L0:])

    # Fisher-KL relationship (should be F ~ 2*KL for small perturbations)
    fk_ratio = [f / (2 * max(abs(k), 1e-10)) for f, k in zip(fisher_mean, kl_mean)]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Fisher information profile
    axes[0, 0].fill_between(layers_t, fisher_mean - fisher_std, fisher_mean + fisher_std,
                            alpha=0.3, color='#e74c3c')
    axes[0, 0].plot(layers_t, fisher_mean, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].axvline(x=critical_layer + 0.5, color='#8e44ad', linewidth=2, linestyle=':',
                        label=f'Peak (L={critical_layer})')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Fisher Information $F$')
    axes[0, 0].set_title('(a) Fisher Information Landscape')
    axes[0, 0].legend(fontsize=8)

    # (b) Cumulative Fisher
    axes[0, 1].plot(layers_t, fisher_cumul, 'o-', color='#8e44ad', markersize=3, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    # Mark 50%, 90% thresholds
    total_F = fisher_cumul[-1]
    for pct, col in [(0.5, '#3498db'), (0.9, '#e74c3c')]:
        idx = np.searchsorted(fisher_cumul, pct * total_F)
        if idx < len(layers_t):
            axes[0, 1].axvline(x=layers_t[idx], color=col, linestyle=':', linewidth=1.5,
                               label=f'{pct:.0%} at L={idx}')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Cumulative $F$')
    axes[0, 1].set_title('(b) Cumulative Fisher Information')
    axes[0, 1].legend(fontsize=8)

    # (c) KL divergence profile
    axes[0, 2].plot(layers_t, kl_mean, 's-', color='#3498db', markersize=4, linewidth=2, label='KL')
    axes[0, 2].plot(layers_t, js_mean, '^-', color='#2ecc71', markersize=4, linewidth=2, label='JS')
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Divergence')
    axes[0, 2].set_title('(c) KL and JS Divergence')
    axes[0, 2].legend(fontsize=8)

    # (d) Fisher vs KL scatter
    axes[1, 0].scatter(kl_mean, fisher_mean, c=layers_t, cmap='coolwarm', s=60, edgecolors='black')
    # Theoretical line: F = 2*KL
    kl_range = np.linspace(min(kl_mean), max(kl_mean), 100)
    axes[1, 0].plot(kl_range, 2 * kl_range, 'k--', alpha=0.3, label='$F = 2 \\cdot KL$ (theory)')
    axes[1, 0].set_xlabel('KL Divergence')
    axes[1, 0].set_ylabel('Fisher Information')
    axes[1, 0].set_title('(d) Fisher-KL Relationship')
    axes[1, 0].legend(fontsize=8)

    # (e) Per-prompt heatmap
    fisher_arr = np.array(all_fisher[:10])
    im = axes[1, 1].imshow(fisher_arr, aspect='auto', cmap='hot',
                            extent=[0, n_layers-1, 10, 0])
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    plt.colorbar(im, ax=axes[1, 1], label='Fisher $F$')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Prompt')
    axes[1, 1].set_title('(e) Fisher Landscape per Prompt')

    # (f) Summary
    summary = (
        f"Fisher Information Landscape\n\n"
        f"Critical layer: {critical_layer}\n"
        f"Peak Fisher: {fisher_mean[critical_layer]:.2f}\n\n"
        f"Pre-L0 mean F: {pre_fisher:.2f}\n"
        f"Post-L0 mean F: {post_fisher:.2f}\n"
        f"Ratio: {pre_fisher/(post_fisher+1e-10):.2f}x\n\n"
        f"Total Fisher: {total_F:.2f}\n"
        f"50%% at layer: {np.searchsorted(fisher_cumul, 0.5*total_F)}\n"
        f"90%% at layer: {np.searchsorted(fisher_cumul, 0.9*total_F)}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 192: Fisher Information Landscape', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase192_fisher')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Critical layer: {critical_layer} (F={fisher_mean[critical_layer]:.2f})")
    print(f"Pre-L0: F={pre_fisher:.2f}, Post-L0: F={post_fisher:.2f}")
    print(f"Ratio: {pre_fisher/(post_fisher+1e-10):.2f}x")
    print(f"Total Fisher: {total_F:.2f}")
    print(f"{'=' * 70}")

    save_results('phase192_fisher', {
        'experiment': 'Fisher Information Landscape',
        'fisher_mean': [float(x) for x in fisher_mean],
        'kl_mean': [float(x) for x in kl_mean],
        'js_mean': [float(x) for x in js_mean],
        'summary': {
            'critical_layer': int(critical_layer),
            'peak_fisher': float(fisher_mean[critical_layer]),
            'pre_L0_mean': float(pre_fisher),
            'post_L0_mean': float(post_fisher),
            'total_fisher': float(total_F),
        }
    })


if __name__ == '__main__':
    main()
