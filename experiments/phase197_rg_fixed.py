# -*- coding: utf-8 -*-
"""
Phase 197: Renormalization Group Fixed Points
===============================================
RG flow connects microscale (early layers) to macroscale (late layers).
At a fixed point, the system looks the same at all scales.

Does the transformer have RG fixed points? Layers where the
representation is scale-invariant (the distribution doesn't change)?

If yes: these are the "critical layers" where universality emerges.
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
    print("Phase 197: Renormalization Group Fixed Points")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_js = []  # Jensen-Shannon between consecutive layers
    all_cosine = []  # Cosine similarity of hidden states
    all_norm_ratio = []  # Norm ratio between consecutive layers

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        js_vals = []
        cos_vals = []
        nr_vals = []

        probs_prev = None
        h_prev = None

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)

            if probs_prev is not None:
                # Jensen-Shannon divergence
                m = 0.5 * (probs + probs_prev)
                js = 0.5 * (probs * torch.log(probs / (m + 1e-10) + 1e-10)).sum() + \
                     0.5 * (probs_prev * torch.log(probs_prev / (m + 1e-10) + 1e-10)).sum()
                js_vals.append(js.item() if not torch.isnan(js) else 0)

                # Cosine similarity of hidden states
                cos = torch.nn.functional.cosine_similarity(h.unsqueeze(0), h_prev.unsqueeze(0)).item()
                cos_vals.append(cos if not np.isnan(cos) else 0)

                # Norm ratio
                nr = h.norm().item() / (h_prev.norm().item() + 1e-10)
                nr_vals.append(nr if not np.isnan(nr) else 1)

            probs_prev = probs
            h_prev = h

        all_js.append(js_vals)
        all_cosine.append(cos_vals)
        all_norm_ratio.append(nr_vals)

    js_mean = np.mean(all_js, axis=0)
    js_std = np.std(all_js, axis=0)
    cos_mean = np.mean(all_cosine, axis=0)
    nr_mean = np.mean(all_norm_ratio, axis=0)

    layers_t = np.arange(n_layers - 1) + 0.5

    # Fixed point detection: layers where JS is minimal (near zero change)
    # A fixed point means the RG transformation doesn't change the system
    js_threshold = np.percentile(js_mean, 15)  # Bottom 15%
    fixed_layers = [int(layers_t[i]) for i in range(len(js_mean)) if js_mean[i] < js_threshold]

    # Beta function: rate of change of coupling constant
    # beta(l) = dJS/dl (second derivative ~ how fast the flow is changing)
    beta = np.gradient(js_mean)

    # Zero crossings of beta = fixed points
    beta_zeros = []
    for i in range(len(beta) - 1):
        if beta[i] * beta[i+1] < 0:  # Sign change
            beta_zeros.append(i + 0.5)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) JS divergence (flow magnitude)
    axes[0, 0].fill_between(layers_t, js_mean - js_std, js_mean + js_std,
                            alpha=0.3, color='#e74c3c')
    axes[0, 0].plot(layers_t, js_mean, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].axhline(y=js_threshold, color='#3498db', linewidth=1, linestyle=':',
                        label=f'Fixed pt threshold')
    for fp in fixed_layers:
        axes[0, 0].axvline(x=fp, color='#2ecc71', linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('JS Divergence')
    axes[0, 0].set_title('(a) RG Flow Magnitude')
    axes[0, 0].legend(fontsize=7)

    # (b) Beta function
    axes[0, 1].plot(layers_t, beta, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    for bz in beta_zeros:
        axes[0, 1].axvline(x=layers_t[int(bz)], color='#e74c3c', linewidth=1.5,
                            linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$\\beta$ function ($dJS/dl$)')
    axes[0, 1].set_title(f'(b) Beta Function ({len(beta_zeros)} zero crossings)')

    # (c) Cosine similarity (representation stability)
    axes[0, 2].plot(layers_t, cos_mean, 'o-', color='#3498db', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Cosine Similarity')
    axes[0, 2].set_title('(c) Representation Stability')

    # (d) Norm ratio (scale factor)
    axes[1, 0].plot(layers_t, nr_mean, 'o-', color='#2ecc71', markersize=4, linewidth=2)
    axes[1, 0].axhline(y=1, color='black', linewidth=0.5, linestyle='--', label='Scale-invariant')
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('||h_{l+1}|| / ||h_l||')
    axes[1, 0].set_title('(d) Scale Factor (RG rescaling)')
    axes[1, 0].legend(fontsize=8)

    # (e) Phase space: JS vs beta
    axes[1, 1].scatter(js_mean, beta, c=layers_t, cmap='viridis', s=60, edgecolors='black')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
    axes[1, 1].set_xlabel('JS Divergence (flow magnitude)')
    axes[1, 1].set_ylabel('$\\beta$ (flow acceleration)')
    axes[1, 1].set_title('(e) RG Phase Space')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Layer')

    # (f) Summary
    summary = (
        f"RG Fixed Point Analysis\n\n"
        f"Fixed point layers: {fixed_layers[:8]}\n"
        f"  (JS < {js_threshold:.4f})\n\n"
        f"Beta function zeros: {len(beta_zeros)}\n"
        f"  at layers: {[f'{layers_t[int(bz)]:.0f}' for bz in beta_zeros[:5]]}\n\n"
        f"Mean JS (all): {np.mean(js_mean):.4f}\n"
        f"Mean JS (pre-L0): {np.mean(js_mean[:L0]):.4f}\n"
        f"Mean JS (post-L0): {np.mean(js_mean[L0:]):.4f}\n\n"
        f"Max cosine: {max(cos_mean):.4f}\n"
        f"  at layer {np.argmax(cos_mean)+1}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 197: Renormalization Group Fixed Points', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase197_rg_fixed')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Fixed layers: {fixed_layers}")
    print(f"Beta zeros: {len(beta_zeros)}")
    print(f"Mean JS: pre-L0={np.mean(js_mean[:L0]):.4f}, post-L0={np.mean(js_mean[L0:]):.4f}")
    print(f"{'=' * 70}")

    save_results('phase197_rg_fixed', {
        'experiment': 'RG Fixed Points',
        'js_mean': [float(x) for x in js_mean],
        'beta': [float(x) for x in beta],
        'cos_mean': [float(x) for x in cos_mean],
        'fixed_layers': fixed_layers,
        'beta_zeros': [float(x) for x in beta_zeros],
        'summary': {
            'n_fixed': len(fixed_layers),
            'n_beta_zeros': len(beta_zeros),
            'mean_js_pre': float(np.mean(js_mean[:L0])),
            'mean_js_post': float(np.mean(js_mean[L0:])),
        }
    })


if __name__ == '__main__':
    main()
