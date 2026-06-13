# -*- coding: utf-8 -*-
"""
Phase 114: Layer Teleportation (Residual Skip Connections)
If the cooling valley layers barely change the representation,
can we "teleport" (skip multiple layers at once) without quality loss?
Test skipping 1, 2, 3, 4, 5 consecutive layers starting from the valley.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

TEST_TEXTS = [
    "The theory of general relativity predicts that massive objects warp the fabric of spacetime",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy",
    "The human brain contains approximately eighty six billion neurons connected by synapses",
    "Machine learning algorithms can identify complex patterns in large datasets",
    "The periodic table organizes all known chemical elements according to atomic number",
    "Quantum entanglement allows particles to be correlated regardless of distance",
    "Climate models predict significant changes in global temperature patterns",
    "The discovery of antibiotics revolutionized medicine for bacterial infections",
]


def measure_ppl_with_skips(model, tok, device, skip_layers):
    """Skip multiple layers at once."""
    hooks = []
    for li in skip_layers:
        if li < len(model.model.layers):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        return (input[0],) + output[1:]
                    return input[0]
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        ppl = torch.exp(out.loss).item()
        if not np.isnan(ppl) and ppl < 1e6:
            ppls.append(ppl)

    for h in hooks:
        h.remove()

    return float(np.mean(ppls)) if ppls else 1e6


def main():
    print("=" * 70)
    print("Phase 114: Layer Teleportation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    baseline = measure_ppl_with_skips(model, tok, device, [])
    print(f"  Baseline PPL: {baseline:.2f}")

    # Test 1: Skip N consecutive layers starting from center of valley (L13)
    valley_center = 13
    consecutive_results = []
    for n_skip in range(1, 8):
        skip_layers = list(range(valley_center, min(valley_center + n_skip, n_layers)))
        ppl = measure_ppl_with_skips(model, tok, device, skip_layers)
        ratio = ppl / baseline
        consecutive_results.append({
            'n_skip': n_skip,
            'layers': skip_layers,
            'ppl': float(ppl),
            'ratio': float(ratio),
        })
        print(f"  Skip {n_skip} from L{valley_center}: PPL={ppl:.2f} ({ratio:.2f}x)")

    # Test 2: Skip N layers from different starting points
    start_points = [3, 7, 11, 15, 19, 23]
    window_results = []
    for start in start_points:
        for n_skip in [3, 5]:
            skip_layers = list(range(start, min(start + n_skip, n_layers)))
            ppl = measure_ppl_with_skips(model, tok, device, skip_layers)
            ratio = ppl / baseline
            window_results.append({
                'start': start,
                'n_skip': n_skip,
                'ppl': float(ppl),
                'ratio': float(ratio),
            })

    # Test 3: Optimal skip - find the best set of N layers to skip
    # Use greedy: skip the layer with lowest individual PPL impact
    layer_ppls = []
    for li in range(n_layers):
        ppl = measure_ppl_with_skips(model, tok, device, [li])
        layer_ppls.append((li, ppl / baseline))

    sorted_layers = sorted(layer_ppls, key=lambda x: x[1])
    greedy_results = []
    for k in range(1, 8):
        best_k = [l for l, _ in sorted_layers[:k]]
        ppl = measure_ppl_with_skips(model, tok, device, best_k)
        ratio = ppl / baseline
        greedy_results.append({
            'k': k,
            'layers': sorted(best_k),
            'ppl': float(ppl),
            'ratio': float(ratio),
        })
        print(f"  Greedy-{k}: skip {sorted(best_k)}, PPL={ppl:.2f} ({ratio:.2f}x)")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Consecutive skip from valley
    ns = [r['n_skip'] for r in consecutive_results]
    ratios = [r['ratio'] for r in consecutive_results]
    colors_a = ['#27ae60' if r < 1.5 else '#f39c12' if r < 2.0 else '#c0392b' for r in ratios]
    axes[0,0].bar(ns, ratios, color=colors_a, alpha=0.8, edgecolor='black')
    axes[0,0].axhline(y=1.0, color='black', linewidth=0.5)
    axes[0,0].axhline(y=1.5, color='#f39c12', linestyle='--', alpha=0.5, label='1.5x')
    axes[0,0].set_xlabel(f'N layers skipped (from L{valley_center})')
    axes[0,0].set_ylabel('PPL / Baseline')
    axes[0,0].set_title('(a) Consecutive Skip from Valley')
    axes[0,0].legend()

    # (b) Window comparison (3-layer skip)
    skip3 = [r for r in window_results if r['n_skip'] == 3]
    starts_3 = [r['start'] for r in skip3]
    ratios_3 = [r['ratio'] for r in skip3]
    colors_3 = ['#27ae60' if r < 1.3 else '#f39c12' if r < 2 else '#c0392b' for r in ratios_3]
    axes[0,1].bar(range(len(starts_3)), ratios_3, color=colors_3, alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(range(len(starts_3)))
    axes[0,1].set_xticklabels([f'L{s}-{s+2}' for s in starts_3], fontsize=8)
    axes[0,1].axhline(y=1.0, color='black', linewidth=0.5)
    axes[0,1].set_ylabel('PPL / Baseline')
    axes[0,1].set_title('(b) 3-Layer Skip at Different Positions')

    # (c) Greedy optimal skip
    ks = [r['k'] for r in greedy_results]
    g_ratios = [r['ratio'] for r in greedy_results]
    axes[0,2].plot(ks, g_ratios, 'o-', color='#8e44ad', markersize=6, linewidth=2)
    axes[0,2].axhline(y=1.5, color='#f39c12', linestyle='--', alpha=0.5, label='1.5x threshold')
    axes[0,2].set_xlabel('N layers skipped (greedy)')
    axes[0,2].set_ylabel('PPL / Baseline')
    axes[0,2].set_title('(c) Greedy Optimal Skip')
    axes[0,2].legend()

    # Find max layers skipable within 1.5x threshold
    max_skip_15 = 0
    for r in greedy_results:
        if r['ratio'] < 1.5:
            max_skip_15 = r['k']

    # (d) 5-layer skip comparison
    skip5 = [r for r in window_results if r['n_skip'] == 5]
    starts_5 = [r['start'] for r in skip5]
    ratios_5 = [r['ratio'] for r in skip5]
    colors_5 = ['#27ae60' if r < 1.5 else '#f39c12' if r < 3 else '#c0392b' for r in ratios_5]
    axes[1,0].bar(range(len(starts_5)), ratios_5, color=colors_5, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(starts_5)))
    axes[1,0].set_xticklabels([f'L{s}-{s+4}' for s in starts_5], fontsize=8)
    axes[1,0].axhline(y=1.0, color='black', linewidth=0.5)
    axes[1,0].set_ylabel('PPL / Baseline')
    axes[1,0].set_title('(b) 5-Layer Skip at Different Positions')

    # (e) Layer importance heatmap
    layer_imps = sorted(layer_ppls, key=lambda x: x[0])
    imp_vals = [x[1] for x in layer_imps]
    colors_e = ['#27ae60' if v < 1.15 else '#f39c12' if v < 1.3 else '#c0392b' for v in imp_vals]
    axes[1,1].bar(range(n_layers), imp_vals, color=colors_e, alpha=0.7, edgecolor='black')
    axes[1,1].axhline(y=1.0, color='black', linewidth=0.5)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('PPL / Baseline (single skip)')
    axes[1,1].set_title('(e) Single-Layer Importance Map')

    # (f) Summary
    best_greedy = greedy_results[max_skip_15 - 1] if max_skip_15 > 0 else greedy_results[0]
    summary = (
        f"Layer Teleportation Summary\n\n"
        f"Max layers skippable (<1.5x PPL):\n"
        f"  Greedy: {max_skip_15} layers\n"
        f"  Layers: {best_greedy['layers']}\n"
        f"  PPL: {best_greedy['ppl']:.2f} ({best_greedy['ratio']:.2f}x)\n\n"
        f"Valley (L13-17) consecutive:\n"
        f"  3-skip: {consecutive_results[2]['ratio']:.2f}x\n"
        f"  5-skip: {consecutive_results[4]['ratio']:.2f}x\n\n"
        f"Compression: {max_skip_15}/{n_layers} = {max_skip_15/n_layers*100:.0f}%"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 114: Layer Teleportation (max {max_skip_15} layers, '
                 f'{max_skip_15/n_layers*100:.0f}% compression)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase114_teleportation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Max skip <1.5x: {max_skip_15} layers ({max_skip_15/n_layers*100:.0f}%)")
    print(f"Best greedy: {best_greedy['layers']} ({best_greedy['ratio']:.2f}x)")
    print(f"{'='*70}")

    save_results('phase114_teleportation', {
        'experiment': 'Layer Teleportation',
        'baseline_ppl': float(baseline),
        'consecutive': consecutive_results,
        'window': window_results,
        'greedy': greedy_results,
        'summary': {
            'max_skip_15x': max_skip_15,
            'best_layers': best_greedy['layers'],
            'compression_pct': float(max_skip_15 / n_layers * 100),
        }
    })


if __name__ == '__main__':
    main()
