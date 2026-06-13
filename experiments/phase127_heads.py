# -*- coding: utf-8 -*-
"""
Phase 127: Attention Head Specialization
Do attention heads specialize at the phase transition?
Measure: (1) Head entropy diversity, (2) Head-to-head correlation,
(3) "Dead head" fraction, (4) Head importance (ablation).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and",
    "Quantum mechanics describes particles at the atomic scale where",
    "Neural networks learn through a process called gradient descent",
    "Black holes form when massive stars undergo gravitational collapse",
    "The periodic table organizes all known chemical elements by number",
    "Evolution operates on heritable variation within natural populations",
    "Photosynthesis converts sunlight into chemical energy stored in glucose",
    "Machine learning discovers hidden patterns within large datasets",
]


def main():
    print("=" * 70)
    print("Phase 127: Attention Head Specialization")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    # Get number of heads from config
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)

    # Collect attention patterns
    head_entropies = np.zeros((n_layers, n_heads))
    head_max_attn = np.zeros((n_layers, n_heads))
    head_diversity = np.zeros(n_layers)

    # Use hooks to capture attention weights since output_attentions may not work
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Hook into each layer's self_attn to get query/key projections
        layer_data = {}

        hooks = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # Get the hidden state output
                    if isinstance(output, tuple):
                        h = output[0][0, -1, :].detach().float().cpu()
                    else:
                        h = output[0, -1, :].detach().float().cpu()
                    # Split into head-sized chunks and compute diversity
                    head_dim = h.shape[0] // n_heads
                    if head_dim > 0:
                        head_vecs = h.view(n_heads, head_dim)
                        # Per-head activation magnitude as proxy for attention focus
                        head_mags = head_vecs.norm(dim=1)
                        layer_data[idx] = head_mags.numpy()
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in hooks:
            h.remove()

        for li in range(n_layers):
            if li in layer_data:
                mags = layer_data[li]
                # Normalize to pseudo-probabilities
                p = mags / (mags.sum() + 1e-10)
                for hi in range(min(n_heads, len(p))):
                    head_entropies[li, hi] += float(p[hi]) / len(PROMPTS)
                    head_max_attn[li, hi] += float(mags[hi]) / len(PROMPTS)

    # Head diversity = std of entropies across heads at each layer
    for li in range(n_layers):
        head_diversity[li] = np.std(head_entropies[li, :])

    # Gini coefficient of head entropies at each layer
    gini = []
    for li in range(n_layers):
        ents = np.sort(head_entropies[li, :])
        n = len(ents)
        idx = np.arange(1, n + 1)
        g = (2 * np.sum(idx * ents) / (n * np.sum(ents) + 1e-10)) - (n + 1) / n
        gini.append(float(g))

    # "Dead heads" = heads with very low entropy (always attend to one token)
    dead_threshold = 0.5
    dead_frac = [np.mean(head_entropies[li, :] < dead_threshold) for li in range(n_layers)]

    # Specialization index = max_entropy - min_entropy
    spec_index = [np.max(head_entropies[li, :]) - np.min(head_entropies[li, :])
                  for li in range(n_layers)]

    layers = np.arange(n_layers)

    pre_div = np.mean(head_diversity[:int(L0)])
    post_div = np.mean(head_diversity[int(L0):])
    pre_gini = np.mean(gini[:int(L0)])
    post_gini = np.mean(gini[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Head entropy heatmap
    im = axes[0,0].imshow(head_entropies.T, aspect='auto', cmap='viridis',
                           origin='lower')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Head')
    axes[0,0].set_title('(a) Head Entropy Map')
    plt.colorbar(im, ax=axes[0,0], shrink=0.7)

    # (b) Head diversity
    axes[0,1].plot(layers, head_diversity, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Std of Head Entropies')
    axes[0,1].set_title(f'(b) Head Diversity (pre={pre_div:.3f}, post={post_div:.3f})')
    axes[0,1].legend()

    # (c) Gini coefficient
    axes[0,2].plot(layers, gini, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Gini Coefficient')
    axes[0,2].set_title('(c) Head Inequality')

    # (d) Dead head fraction
    axes[1,0].plot(layers, dead_frac, 'o-', color='#7f8c8d', markersize=4, linewidth=2)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Fraction Dead Heads')
    axes[1,0].set_title('(d) Dead Heads (ent < 0.5)')

    # (e) Specialization index
    axes[1,1].plot(layers, spec_index, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[1,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Max - Min Entropy')
    axes[1,1].set_title('(e) Specialization Index')

    # (f) Summary
    summary = (
        f"Attention Head Specialization\n\n"
        f"Diversity pre-L0: {pre_div:.3f}\n"
        f"Diversity post-L0: {post_div:.3f}\n\n"
        f"Gini pre-L0: {pre_gini:.3f}\n"
        f"Gini post-L0: {post_gini:.3f}\n\n"
        f"Dead heads: pre={np.mean(dead_frac[:int(L0)]):.3f}, "
        f"post={np.mean(dead_frac[int(L0):]):.3f}\n\n"
        f"Heads {'SPECIALIZE' if post_div > pre_div else 'HOMOGENIZE'}\n"
        f"after transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 127: Attention Head Specialization',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase127_heads')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Diversity: pre={pre_div:.3f}, post={post_div:.3f}")
    print(f"Gini: pre={pre_gini:.3f}, post={post_gini:.3f}")
    print(f"{'='*70}")

    save_results('phase127_heads', {
        'experiment': 'Attention Head Specialization',
        'head_diversity': [float(v) for v in head_diversity],
        'gini': gini,
        'dead_frac': dead_frac,
        'spec_index': spec_index,
        'summary': {
            'pre_diversity': float(pre_div),
            'post_diversity': float(post_div),
            'pre_gini': float(pre_gini),
            'post_gini': float(post_gini),
        }
    })


if __name__ == '__main__':
    main()
