# -*- coding: utf-8 -*-
"""
Phase 194: Lyapunov Spectrum & Kolmogorov-Sinai Entropy
=========================================================
Inject tiny perturbation at each layer, measure how it amplifies.
Lambda(l) = (1/(L-l)) * log(divergence/perturbation)
KS entropy = sum of positive Lyapunov exponents.

Maps the "chaos landscape" of the transformer.
Positive lambda = chaotic amplification, negative = stability.
Phase transition at L0 should appear as a change in chaos regime.
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
]

EPSILON = 1e-4  # Perturbation magnitude


def main():
    print("=" * 70)
    print("Phase 194: Lyapunov Spectrum & KS Entropy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_transformer_layers = len(model.model.layers)
    n_layers = n_transformer_layers + 1
    L0 = 21

    all_lyapunov = []
    all_divergence = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Baseline
        with torch.no_grad():
            out_base = model(**inp, output_hidden_states=True)
        base_logits = out_base.logits[0, -1, :].float()
        base_probs = torch.softmax(base_logits, dim=-1)

        lyapunov_vals = []
        div_vals = []

        for inject_layer in range(n_transformer_layers):
            # Hook to inject perturbation at this layer
            perturbation = None

            def make_hook(layer_idx):
                def hook(module, input, output):
                    nonlocal perturbation
                    if isinstance(output, tuple):
                        h = output[0]
                        # Random unit perturbation scaled by epsilon
                        if perturbation is None:
                            perturbation = torch.randn_like(h) * EPSILON
                            perturbation = perturbation / (perturbation.norm() + 1e-10) * EPSILON * h.norm()
                        h_new = h + perturbation
                        return (h_new,) + output[1:]
                    return output
                return hook

            handle = model.model.layers[inject_layer].register_forward_hook(make_hook(inject_layer))

            with torch.no_grad():
                out_pert = model(**inp, output_hidden_states=True)

            handle.remove()
            perturbation = None

            # Measure divergence at output
            pert_logits = out_pert.logits[0, -1, :].float()
            pert_probs = torch.softmax(pert_logits, dim=-1)

            # KL divergence as measure of output divergence
            kl = (base_probs * torch.log((base_probs + 1e-10) / (pert_probs + 1e-10))).sum().item()
            kl = abs(kl) if not np.isnan(kl) else 0

            # L2 divergence in hidden state space
            h_base = out_base.hidden_states[-1][0, -1, :].float()
            h_pert = out_pert.hidden_states[-1][0, -1, :].float()
            l2_div = (h_pert - h_base).norm().item()

            # Lyapunov exponent
            remaining = n_transformer_layers - inject_layer
            if remaining > 0 and l2_div > 1e-15:
                lam = np.log(l2_div / (EPSILON * h_base.norm().item() + 1e-10)) / remaining
            else:
                lam = 0

            lyapunov_vals.append(lam if not np.isnan(lam) else 0)
            div_vals.append(kl)

        all_lyapunov.append(lyapunov_vals)
        all_divergence.append(div_vals)

    lyap_mean = np.mean(all_lyapunov, axis=0)
    lyap_std = np.std(all_lyapunov, axis=0)
    div_mean = np.mean(all_divergence, axis=0)

    layers = np.arange(n_transformer_layers)

    # KS entropy = sum of positive Lyapunov exponents
    positive_lyap = [l for l in lyap_mean if l > 0]
    KS_entropy = sum(positive_lyap)
    n_positive = len(positive_lyap)
    n_negative = len([l for l in lyap_mean if l < 0])

    # Max Lyapunov exponent
    max_lyap = max(lyap_mean) if len(lyap_mean) > 0 else 0
    max_lyap_layer = np.argmax(lyap_mean) if len(lyap_mean) > 0 else 0

    # Pre/post L0
    pre_lyap = np.mean(lyap_mean[:L0])
    post_lyap = np.mean(lyap_mean[L0:])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Lyapunov spectrum
    colors_a = ['#e74c3c' if l > 0 else '#3498db' for l in lyap_mean]
    axes[0, 0].bar(layers, lyap_mean, color=colors_a, edgecolor='black', alpha=0.7, width=0.8)
    axes[0, 0].axhline(y=0, color='black', linewidth=1)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].set_xlabel('Injection Layer')
    axes[0, 0].set_ylabel('Lyapunov Exponent $\\lambda$')
    axes[0, 0].set_title('(a) Lyapunov Spectrum (red=chaotic)')
    axes[0, 0].legend(fontsize=8)

    # (b) Lyapunov with error bars
    axes[0, 1].fill_between(layers, lyap_mean - lyap_std, lyap_mean + lyap_std,
                            alpha=0.3, color='#8e44ad')
    axes[0, 1].plot(layers, lyap_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linewidth=1)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Injection Layer')
    axes[0, 1].set_ylabel('$\\lambda$ (mean +/- std)')
    axes[0, 1].set_title('(b) Lyapunov Profile with Variance')

    # (c) Output KL divergence
    axes[0, 2].plot(layers, div_mean, 's-', color='#2ecc71', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Injection Layer')
    axes[0, 2].set_ylabel('KL Divergence at Output')
    axes[0, 2].set_title('(c) Perturbation Sensitivity')

    # (d) Lyapunov vs remaining layers
    remaining = n_transformer_layers - layers
    axes[1, 0].scatter(remaining, lyap_mean, c=layers, cmap='coolwarm', s=60, edgecolors='black')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Remaining Layers (L - l)')
    axes[1, 0].set_ylabel('$\\lambda$')
    axes[1, 0].set_title('(d) Exponent vs Distance to Output')

    # (e) Cumulative KS entropy
    cum_ks = np.cumsum([max(0, l) for l in lyap_mean])
    axes[1, 1].plot(layers, cum_ks, 'o-', color='#c0392b', markersize=3, linewidth=2)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Injection Layer')
    axes[1, 1].set_ylabel('Cumulative KS Entropy')
    axes[1, 1].set_title(f'(e) KS Entropy = {KS_entropy:.4f}')

    # (f) Summary
    summary = (
        f"Lyapunov Spectrum & KS Entropy\n\n"
        f"KS entropy: {KS_entropy:.4f}\n"
        f"Max Lyapunov: {max_lyap:.4f} (L={max_lyap_layer})\n\n"
        f"Positive exponents: {n_positive}/{n_transformer_layers}\n"
        f"Negative exponents: {n_negative}/{n_transformer_layers}\n\n"
        f"Pre-L0 mean: {pre_lyap:.4f}\n"
        f"Post-L0 mean: {post_lyap:.4f}\n\n"
        f"System is {'CHAOTIC' if KS_entropy > 0.1 else 'STABLE'}\n"
        f"{'EDGE OF CHAOS' if 0.01 < KS_entropy < 1 else ''}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 194: Lyapunov Spectrum & KS Entropy', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase194_lyapunov')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"KS entropy: {KS_entropy:.4f}")
    print(f"Max Lyapunov: {max_lyap:.4f} at layer {max_lyap_layer}")
    print(f"Positive: {n_positive}, Negative: {n_negative}")
    print(f"Pre-L0: {pre_lyap:.4f}, Post-L0: {post_lyap:.4f}")
    print(f"{'=' * 70}")

    save_results('phase194_lyapunov', {
        'experiment': 'Lyapunov Spectrum & KS Entropy',
        'lyapunov_mean': [float(x) for x in lyap_mean],
        'divergence_mean': [float(x) for x in div_mean],
        'summary': {
            'KS_entropy': float(KS_entropy),
            'max_lyapunov': float(max_lyap),
            'max_lyapunov_layer': int(max_lyap_layer),
            'n_positive': n_positive, 'n_negative': n_negative,
            'pre_L0_mean': float(pre_lyap), 'post_L0_mean': float(post_lyap),
        }
    })


if __name__ == '__main__':
    main()
