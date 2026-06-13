# -*- coding: utf-8 -*-
"""
Phase 109: Phase Transition Predicts Optimal Pruning
If eta transitions at L0~22, then removing layers near L0 should cause
the largest quality drop. Test: remove 1 layer at each position and
measure PPL change to find the "most important" layers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

TEST_TEXTS = [
    "The theory of general relativity predicts that massive objects warp the fabric of spacetime around them",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy stored in glucose",
    "The human brain contains approximately eighty six billion neurons connected by trillions of synapses",
    "Machine learning algorithms can identify complex patterns in large datasets without explicit programming",
    "The periodic table organizes all known chemical elements according to their atomic number and properties",
    "Quantum entanglement allows particles to be correlated regardless of the distance separating them",
    "The discovery of antibiotics revolutionized medicine by providing effective treatments for bacterial infections",
    "Climate models predict significant changes in global temperature patterns over the coming decades",
]


def measure_ppl_skip_layer(model, tok, device, skip_layer):
    """Measure PPL with one layer completely skipped (output = input)."""
    hook = None
    if skip_layer is not None and skip_layer < len(model.model.layers):
        def hook_fn(module, input, output):
            # Return the input unchanged (skip this layer)
            if isinstance(output, tuple):
                return (input[0],) + output[1:]
            return input[0]
        hook = model.model.layers[skip_layer].register_forward_hook(hook_fn)

    ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        ppl = torch.exp(out.loss).item()
        if not np.isnan(ppl) and ppl < 1e6:
            ppls.append(ppl)

    if hook:
        hook.remove()

    return float(np.mean(ppls)) if ppls else 1e6


def main():
    print("=" * 70)
    print("Phase 109: Phase Transition Predicts Pruning")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    # Baseline PPL
    baseline_ppl = measure_ppl_skip_layer(model, tok, device, None)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Skip each layer one at a time
    results = []
    for li in range(n_layers):
        ppl = measure_ppl_skip_layer(model, tok, device, li)
        ppl_ratio = ppl / baseline_ppl
        importance = ppl_ratio - 1.0  # how much worse
        results.append({
            'layer': li,
            'ppl': float(ppl),
            'ppl_ratio': float(ppl_ratio),
            'importance': float(importance),
        })
        if li % 5 == 0 or li == n_layers - 1:
            print(f"  Skip L{li:2d}: PPL={ppl:.2f} ({ppl_ratio:.2f}x)")

    # Analysis
    importances = np.array([r['importance'] for r in results])
    layers = np.arange(n_layers)

    # Most important layers
    sorted_by_imp = sorted(results, key=lambda r: r['importance'], reverse=True)
    most_imp = sorted_by_imp[0]['layer']

    # Correlation: does importance peak near L0?
    # Pre/post comparison
    pre = [r['importance'] for r in results if r['layer'] < L0]
    post = [r['importance'] for r in results if r['layer'] >= L0]
    mean_pre = np.mean(pre)
    mean_post = np.mean(post)

    # Least important layers (best to prune)
    least_imp = sorted_by_imp[-1]['layer']
    safe_to_prune = [r['layer'] for r in sorted_by_imp[-5:]]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Importance profile
    colors_bar = ['#c0392b' if imp > 0.5 else '#f39c12' if imp > 0.1 else '#27ae60'
                  for imp in importances]
    axes[0,0].bar(layers, importances, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=L0, color='#2980b9', linewidth=2, linestyle='--', label=f'$L_0={L0:.0f}$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Importance (PPL ratio - 1)')
    axes[0,0].set_title('(a) Layer Importance')
    axes[0,0].legend(fontsize=8)

    # (b) PPL ratio
    ppl_ratios = [r['ppl_ratio'] for r in results]
    axes[0,1].plot(layers, ppl_ratios, 'o-', color='#8e44ad', markersize=4, linewidth=1.5)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].axhline(y=1, color='gray', linewidth=0.5)
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('PPL / Baseline')
    axes[0,1].set_title('(b) PPL Degradation')

    # (c) Cumulative importance
    cum_imp = np.cumsum(importances)
    axes[0,2].plot(layers, cum_imp, 'o-', color='#27ae60', markersize=3)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    half_total = cum_imp[-1] / 2
    half_layer = np.argmin(np.abs(cum_imp - half_total))
    axes[0,2].axhline(y=half_total, color='gray', linestyle=':', alpha=0.5)
    axes[0,2].axvline(x=half_layer, color='gray', linestyle=':', alpha=0.5,
                      label=f'50% at L{half_layer}')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Cumulative Importance')
    axes[0,2].set_title('(c) Cumulative Importance')
    axes[0,2].legend(fontsize=8)

    # (d) Pre vs Post transition
    axes[1,0].bar([0, 1], [mean_pre, mean_post],
                  color=['#3498db', '#c0392b'], alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks([0, 1])
    axes[1,0].set_xticklabels([f'Pre (L<{int(L0)})', f'Post (L>={int(L0)})'])
    axes[1,0].set_ylabel('Mean Importance')
    axes[1,0].set_title('(d) Pre vs Post Transition')

    # (e) Top 5 most and least important
    top5 = sorted_by_imp[:5]
    bot5 = sorted_by_imp[-5:]
    combined = top5 + bot5
    c_names = [f'L{r["layer"]}' for r in combined]
    c_vals = [r['importance'] for r in combined]
    c_colors = ['#c0392b']*5 + ['#27ae60']*5
    axes[1,1].barh(range(10), c_vals, color=c_colors, alpha=0.8, edgecolor='black')
    axes[1,1].set_yticks(range(10))
    axes[1,1].set_yticklabels(c_names)
    axes[1,1].set_xlabel('Importance')
    axes[1,1].set_title('(e) Most/Least Critical Layers')

    # (f) Summary
    summary = (
        f"Layer Pruning Prediction\n\n"
        f"Most important: L{most_imp}\n"
        f"Least important: L{least_imp}\n\n"
        f"Safe to prune: {safe_to_prune}\n\n"
        f"Pre-transition importance: {mean_pre:.3f}\n"
        f"Post-transition importance: {mean_post:.3f}\n\n"
        f"50% importance at: L{half_layer}\n"
        f"Matches L0: {'YES' if abs(half_layer - L0) <= 3 else 'NO'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 109: Pruning Prediction (most critical: L{most_imp})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase109_pruning')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Most important: L{most_imp}")
    print(f"Least important: L{least_imp}")
    print(f"Safe to prune: {safe_to_prune}")
    print(f"Pre importance: {mean_pre:.3f}, Post: {mean_post:.3f}")
    print(f"50% importance at: L{half_layer}")
    print(f"{'='*70}")

    save_results('phase109_pruning', {
        'experiment': 'Phase Transition Predicts Pruning',
        'baseline_ppl': float(baseline_ppl),
        'results': results,
        'summary': {
            'most_important': int(most_imp),
            'least_important': int(least_imp),
            'safe_to_prune': safe_to_prune,
            'mean_pre': float(mean_pre),
            'mean_post': float(mean_post),
            'half_importance_layer': int(half_layer),
        }
    })


if __name__ == '__main__':
    main()
