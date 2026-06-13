# -*- coding: utf-8 -*-
"""
Phase 124: Skewness-Gated Dynamic Wormhole
Use real-time skewness monitoring to dynamically skip cooling valley
layers on a per-token basis. Benchmark against static skip.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

TEST_TEXTS = [
    "The theory of general relativity predicts that massive objects warp the fabric of spacetime around them",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy stored as glucose",
    "The human brain contains approximately eighty six billion neurons connected by trillions of synapses",
    "Machine learning algorithms can identify complex patterns in large datasets with remarkable accuracy",
    "The periodic table organizes all known chemical elements according to their atomic number and properties",
    "Quantum entanglement allows two particles to be correlated regardless of the distance separating them",
    "Climate models predict significant changes in global temperature patterns over the coming decades",
    "The discovery of antibiotics revolutionized medicine and saved countless lives from bacterial infections",
    "Evolution by natural selection is the primary mechanism driving biological diversity across species",
    "Cryptographic algorithms protect sensitive information transmitted across insecure network connections",
]


def main():
    print("=" * 70)
    print("Phase 124: Skewness-Gated Dynamic Wormhole")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # === Strategy 1: Baseline (no skip) ===
    baseline_ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        baseline_ppls.append(torch.exp(out.loss).item())
    baseline_ppl = np.mean(baseline_ppls)

    # === Strategy 2: Static skip (L13-17, from Phase 114) ===
    static_skip = [13, 14, 15, 16, 17]
    hooks_s = []
    for li in static_skip:
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return hook_fn
        hooks_s.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    static_ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        static_ppls.append(torch.exp(out.loss).item())
    static_ppl = np.mean(static_ppls)

    for h in hooks_s:
        h.remove()

    # === Strategy 3: Skewness-gated dynamic skip ===
    # Monitor skewness at each layer during forward pass
    # If skewness < threshold (in cooling valley), skip the layer

    skew_threshold = -1.5  # Below this = in the valley
    skip_start = 10  # Only consider skipping L10-20
    skip_end = 20

    # First, measure skewness profile at each layer
    layer_skewness = [[] for _ in range(n_layers)]

    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        hooks_m = []
        skew_data = {}

        for li in range(n_layers):
            def make_measure_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0][0, -1, :].float()
                    else:
                        h = output[0, -1, :].float()
                    from scipy import stats as sp_stats
                    sk = sp_stats.skew(h.cpu().numpy())
                    skew_data[idx] = float(sk) if not np.isnan(sk) else 0
                return hook_fn
            hooks_m.append(model.model.layers[li].register_forward_hook(make_measure_hook(li)))

        with torch.no_grad():
            model(**inp)

        for h in hooks_m:
            h.remove()

        for li in range(n_layers):
            if li in skew_data:
                layer_skewness[li].append(skew_data[li])

    avg_skewness = [np.mean(v) if v else 0 for v in layer_skewness]

    # Determine which layers to dynamically skip based on skewness
    dynamic_skip = [li for li in range(skip_start, skip_end + 1)
                    if avg_skewness[li] < skew_threshold]

    print(f"  Dynamic skip layers (skew < {skew_threshold}): {dynamic_skip}")

    # Apply dynamic skip
    hooks_d = []
    for li in dynamic_skip:
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return hook_fn
        hooks_d.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    dynamic_ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        dynamic_ppls.append(torch.exp(out.loss).item())
    dynamic_ppl = np.mean(dynamic_ppls)

    for h in hooks_d:
        h.remove()

    # === Strategy 4: Aggressive skip (based on cos_sim > 0.95 from P112) ===
    # Test wider range
    aggressive_skip = [12, 13, 14, 15, 16, 17, 18]
    hooks_a = []
    for li in aggressive_skip:
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return hook_fn
        hooks_a.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    aggressive_ppls = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        aggressive_ppls.append(torch.exp(out.loss).item())
    aggressive_ppl = np.mean(aggressive_ppls)

    for h in hooks_a:
        h.remove()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Skewness profile with skip zones
    axes[0, 0].plot(range(n_layers), avg_skewness, 'o-', color='#c0392b', markersize=4)
    axes[0, 0].axhline(y=skew_threshold, color='#f39c12', linestyle='--',
                        label=f'Threshold={skew_threshold}')
    for li in dynamic_skip:
        axes[0, 0].axvspan(li - 0.5, li + 0.5, alpha=0.2, color='#f39c12')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Skewness')
    axes[0, 0].set_title(f'(a) Skewness Profile (skip {len(dynamic_skip)} layers)')
    axes[0, 0].legend()

    # (b) PPL comparison
    strategies = ['Baseline', f'Static\n(L13-17)', f'Dynamic\n(skew-gated)', f'Aggressive\n(L12-18)']
    ppls = [baseline_ppl, static_ppl, dynamic_ppl, aggressive_ppl]
    ratios = [p / baseline_ppl for p in ppls]
    n_skips = [0, len(static_skip), len(dynamic_skip), len(aggressive_skip)]
    colors = ['#2980b9', '#27ae60', '#f39c12', '#c0392b']
    axes[0, 1].bar(range(len(strategies)), ratios, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_xticks(range(len(strategies)))
    axes[0, 1].set_xticklabels(strategies, fontsize=8)
    axes[0, 1].axhline(y=1.0, color='black', linewidth=0.5)
    axes[0, 1].axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('PPL / Baseline')
    axes[0, 1].set_title('(b) PPL Comparison')
    for i, (r, n) in enumerate(zip(ratios, n_skips)):
        axes[0, 1].text(i, r + 0.02, f'{r:.2f}x\n({n}L)',
                        ha='center', fontsize=8)

    # (c) Per-text PPL comparison
    x = np.arange(len(TEST_TEXTS))
    w = 0.2
    axes[0, 2].bar(x - 1.5*w, baseline_ppls, w, color='#2980b9', alpha=0.7, label='Base')
    axes[0, 2].bar(x - 0.5*w, static_ppls, w, color='#27ae60', alpha=0.7, label='Static')
    axes[0, 2].bar(x + 0.5*w, dynamic_ppls, w, color='#f39c12', alpha=0.7, label='Dynamic')
    axes[0, 2].bar(x + 1.5*w, aggressive_ppls, w, color='#c0392b', alpha=0.7, label='Aggr')
    axes[0, 2].set_xlabel('Text')
    axes[0, 2].set_ylabel('PPL')
    axes[0, 2].set_title('(c) Per-Text PPL')
    axes[0, 2].legend(fontsize=7)

    # (d) Compression vs quality Pareto
    axes[1, 0].scatter([0, len(static_skip)/n_layers*100,
                        len(dynamic_skip)/n_layers*100,
                        len(aggressive_skip)/n_layers*100],
                       ratios, s=150, c=colors, edgecolors='black', zorder=5)
    for i, (s, r, name) in enumerate(zip(
        [0, len(static_skip)/n_layers*100, len(dynamic_skip)/n_layers*100,
         len(aggressive_skip)/n_layers*100], ratios,
        ['Base', 'Static', 'Dynamic', 'Aggressive'])):
        axes[1, 0].annotate(name, (s, r), textcoords="offset points",
                           xytext=(10, 5), fontsize=8)
    axes[1, 0].set_xlabel('Compression (%)')
    axes[1, 0].set_ylabel('PPL Ratio')
    axes[1, 0].set_title('(d) Pareto Front')

    # (e) Speedup analysis
    speedups = [0, len(static_skip)/n_layers*100, len(dynamic_skip)/n_layers*100,
                len(aggressive_skip)/n_layers*100]
    quality = [100, 100/ratios[1], 100/ratios[2], 100/ratios[3]]
    axes[1, 1].bar(range(len(strategies)), speedups, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_xticks(range(len(strategies)))
    axes[1, 1].set_xticklabels(strategies, fontsize=8)
    axes[1, 1].set_ylabel('Theoretical Speedup (%)')
    axes[1, 1].set_title('(e) Speedup')

    # (f) Summary
    best = min([(r, i) for i, r in enumerate(ratios[1:])], key=lambda x: x[0])
    summary = (
        f"Dynamic Wormhole Summary\n\n"
        f"Baseline PPL: {baseline_ppl:.2f}\n\n"
        f"Static (L13-17): {static_ppl:.2f} ({ratios[1]:.2f}x)\n"
        f"Dynamic (skew): {dynamic_ppl:.2f} ({ratios[2]:.2f}x)\n"
        f"Aggressive: {aggressive_ppl:.2f} ({ratios[3]:.2f}x)\n\n"
        f"Dynamic skip: {dynamic_skip}\n"
        f"Best: {strategies[best[1]+1].replace(chr(10),' ')}\n"
        f"({n_skips[best[1]+1]}/{n_layers} = {n_skips[best[1]+1]/n_layers*100:.0f}% skip)"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 124: Skewness-Gated Dynamic Wormhole',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase124_wormhole')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Baseline: {baseline_ppl:.2f}")
    print(f"Static (L13-17): {static_ppl:.2f} ({ratios[1]:.2f}x)")
    print(f"Dynamic (skew): {dynamic_ppl:.2f} ({ratios[2]:.2f}x)")
    print(f"Aggressive (L12-18): {aggressive_ppl:.2f} ({ratios[3]:.2f}x)")
    print(f"Dynamic layers: {dynamic_skip}")
    print(f"{'='*70}")

    save_results('phase124_wormhole', {
        'experiment': 'Skewness-Gated Dynamic Wormhole',
        'baseline_ppl': float(baseline_ppl),
        'static_ppl': float(static_ppl),
        'dynamic_ppl': float(dynamic_ppl),
        'aggressive_ppl': float(aggressive_ppl),
        'dynamic_skip_layers': dynamic_skip,
        'avg_skewness': [float(v) for v in avg_skewness],
    })


if __name__ == '__main__':
    main()
