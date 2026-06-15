# -*- coding: utf-8 -*-
"""
Phase 195: Onsager Reciprocal Relations
=========================================
Near equilibrium, the transport coefficients obey L_ij = L_ji
(Onsager's theorem, Nobel Prize 1968).

Test: perturb at layer i, measure response at layer j.
      Then perturb at layer j, measure response at layer i.
      If L_ij ~ L_ji, the transformer respects microscopic reversibility.

This tests the deepest symmetry principle of non-equilibrium thermodynamics.
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
    "Machine learning discovers hidden patterns",
    "Protein folding determines biological function",
]

EPSILON = 1e-3
TEST_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]  # Sample layers to test


def main():
    print("=" * 70)
    print("Phase 195: Onsager Reciprocal Relations")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_transformer_layers = len(model.model.layers)
    L0 = 21

    # Filter test layers to valid range
    test_layers = [l for l in TEST_LAYERS if l < n_transformer_layers]
    n_test = len(test_layers)

    all_L_ij = np.zeros((n_test, n_test))
    all_counts = np.zeros((n_test, n_test))

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Baseline hidden states
        with torch.no_grad():
            out_base = model(**inp, output_hidden_states=True)

        base_hidden = {}
        for idx, li in enumerate(test_layers):
            base_hidden[li] = out_base.hidden_states[li + 1][0, -1, :].float().clone()

        # For each pair (i, j) where i < j:
        # Perturb at layer i, measure response at layer j
        for idx_i, layer_i in enumerate(test_layers):
            # Create perturbation hook for layer_i
            pert_vector = torch.randn(1, 1, base_hidden[layer_i].shape[0]).to(device)
            pert_vector = pert_vector / (pert_vector.norm() + 1e-10) * EPSILON

            def make_hook(pert):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (h + pert.to(h.dtype),) + output[1:]
                    return output
                return hook

            handle = model.model.layers[layer_i].register_forward_hook(make_hook(pert_vector))
            with torch.no_grad():
                out_pert = model(**inp, output_hidden_states=True)
            handle.remove()

            # Measure response at all other test layers
            for idx_j, layer_j in enumerate(test_layers):
                if idx_j == idx_i:
                    continue
                h_pert = out_pert.hidden_states[layer_j + 1][0, -1, :].float()
                h_base = base_hidden[layer_j]
                response = (h_pert - h_base).norm().item() / (EPSILON + 1e-10)
                if not np.isnan(response):
                    all_L_ij[idx_i, idx_j] += response
                    all_counts[idx_i, idx_j] += 1

    # Average
    all_L_ij = all_L_ij / (all_counts + 1e-10)

    # Test reciprocity: L_ij vs L_ji
    reciprocity_ratios = []
    pairs = []
    for i in range(n_test):
        for j in range(i + 1, n_test):
            L_ij = all_L_ij[i, j]
            L_ji = all_L_ij[j, i]
            if L_ij > 1e-10 and L_ji > 1e-10:
                ratio = min(L_ij, L_ji) / max(L_ij, L_ji)
                reciprocity_ratios.append(ratio)
                pairs.append((test_layers[i], test_layers[j], L_ij, L_ji, ratio))

    mean_reciprocity = np.mean(reciprocity_ratios) if reciprocity_ratios else 0
    std_reciprocity = np.std(reciprocity_ratios) if reciprocity_ratios else 0

    # Separate pre-L0, post-L0, and cross pairs
    pre_pairs = [(l, r) for li, lj, lij, lji, r in pairs if li < L0 and lj < L0]
    post_pairs = [(l, r) for li, lj, lij, lji, r in pairs if li >= L0 and lj >= L0]
    cross_pairs = [(l, r) for li, lj, lij, lji, r in pairs
                   if (li < L0) != (lj < L0)]

    pre_mean = np.mean([r for _, r in pre_pairs]) if pre_pairs else 0
    post_mean = np.mean([r for _, r in post_pairs]) if post_pairs else 0
    cross_mean = np.mean([r for _, r in cross_pairs]) if cross_pairs else 0

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) L_ij matrix
    im1 = axes[0, 0].imshow(all_L_ij, aspect='auto', cmap='hot')
    axes[0, 0].set_xticks(range(n_test))
    axes[0, 0].set_xticklabels(test_layers, fontsize=7)
    axes[0, 0].set_yticks(range(n_test))
    axes[0, 0].set_yticklabels(test_layers, fontsize=7)
    axes[0, 0].set_xlabel('Response Layer $j$')
    axes[0, 0].set_ylabel('Perturbation Layer $i$')
    plt.colorbar(im1, ax=axes[0, 0], label='$L_{ij}$')
    axes[0, 0].set_title('(a) Transport Coefficient Matrix')

    # (b) Symmetry check: L_ij vs L_ji
    if pairs:
        L_ij_vals = [p[2] for p in pairs]
        L_ji_vals = [p[3] for p in pairs]
        axes[0, 1].scatter(L_ij_vals, L_ji_vals, c='#3498db', s=60, edgecolors='black', alpha=0.7)
        lim = max(max(L_ij_vals), max(L_ji_vals)) * 1.1
        axes[0, 1].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Perfect reciprocity')
        axes[0, 1].set_xlabel('$L_{ij}$ (perturb $i$, measure $j$)')
        axes[0, 1].set_ylabel('$L_{ji}$ (perturb $j$, measure $i$)')
        axes[0, 1].set_title(f'(b) Onsager Test (r={mean_reciprocity:.3f})')
        axes[0, 1].legend(fontsize=8)

    # (c) Reciprocity ratio by pair distance
    if pairs:
        distances = [abs(p[1] - p[0]) for p in pairs]
        ratios = [p[4] for p in pairs]
        axes[0, 2].scatter(distances, ratios, c='#2ecc71', s=60, edgecolors='black')
        axes[0, 2].axhline(y=1, color='black', linestyle='--', alpha=0.3, label='Perfect symmetry')
        axes[0, 2].set_xlabel('Layer Distance $|i - j|$')
        axes[0, 2].set_ylabel('min($L_{ij}$,$L_{ji}$) / max')
        axes[0, 2].set_title('(c) Reciprocity vs Distance')
        axes[0, 2].legend(fontsize=8)

    # (d) L_ij - L_ji (asymmetry)
    asym = all_L_ij - all_L_ij.T
    im2 = axes[1, 0].imshow(asym, aspect='auto', cmap='RdBu_r',
                             vmin=-np.abs(asym).max(), vmax=np.abs(asym).max())
    axes[1, 0].set_xticks(range(n_test))
    axes[1, 0].set_xticklabels(test_layers, fontsize=7)
    axes[1, 0].set_yticks(range(n_test))
    axes[1, 0].set_yticklabels(test_layers, fontsize=7)
    plt.colorbar(im2, ax=axes[1, 0], label='$L_{ij} - L_{ji}$')
    axes[1, 0].set_title('(d) Asymmetry Matrix')

    # (e) Reciprocity distribution
    if reciprocity_ratios:
        axes[1, 1].hist(reciprocity_ratios, bins=15, color='#8e44ad', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=mean_reciprocity, color='#e74c3c', linewidth=2, linestyle='--',
                            label=f'Mean={mean_reciprocity:.3f}')
        axes[1, 1].set_xlabel('Reciprocity Ratio')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('(e) Reciprocity Distribution')
        axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Onsager Reciprocal Relations\n\n"
        f"Mean reciprocity: {mean_reciprocity:.3f}\n"
        f"  +/- {std_reciprocity:.3f}\n"
        f"  (1.0 = perfect Onsager)\n\n"
        f"By region:\n"
        f"  Pre-L0:  {pre_mean:.3f}\n"
        f"  Post-L0: {post_mean:.3f}\n"
        f"  Cross:   {cross_mean:.3f}\n\n"
        f"Onsager: {'SATISFIED' if mean_reciprocity > 0.5 else 'VIOLATED'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 195: Onsager Reciprocal Relations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase195_onsager')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Mean reciprocity: {mean_reciprocity:.3f} +/- {std_reciprocity:.3f}")
    print(f"Pre-L0: {pre_mean:.3f}, Post-L0: {post_mean:.3f}, Cross: {cross_mean:.3f}")
    print(f"Onsager: {'SATISFIED' if mean_reciprocity > 0.5 else 'VIOLATED'}")
    print(f"{'=' * 70}")

    save_results('phase195_onsager', {
        'experiment': 'Onsager Reciprocal Relations',
        'L_ij_matrix': all_L_ij.tolist(),
        'test_layers': test_layers,
        'summary': {
            'mean_reciprocity': float(mean_reciprocity),
            'std_reciprocity': float(std_reciprocity),
            'pre_L0': float(pre_mean), 'post_L0': float(post_mean),
            'cross': float(cross_mean),
        }
    })


if __name__ == '__main__':
    main()
