# -*- coding: utf-8 -*-
"""
Phase 98: DMA Attack Surface Map
Phase 89 showed 5-layer DMA achieves 75% hijack. Systematically map
which layer combinations are most vulnerable to multi-layer injection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

FACTS = [
    {"prompt": "The capital of France is", "target": "Tokyo"},
    {"prompt": "Water boils at", "target": "fifty"},
    {"prompt": "The largest planet is", "target": "Mars"},
    {"prompt": "The speed of light is", "target": "slow"},
]


def inject_at_layers(model, tok, device, prompt, target, layers, alpha=10.0):
    """Inject target direction at specified layers."""
    ids = tok(target, add_special_tokens=False)['input_ids']
    if not ids:
        return {'hijacked': False, 'target_prob': 0.0}
    emb = model.model.embed_tokens.weight[ids[0]].detach().float()
    direction = emb / (emb.norm() + 1e-10)

    hooks = []
    for li in layers:
        if li < len(model.model.layers):
            def make_hook(d, s):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_fp32 = h.to(torch.float32)
                    h_fp32[:, -1, :] += s * d.to(h.device)
                    result = h_fp32.to(h.dtype)
                    if isinstance(output, tuple):
                        return (result,) + output[1:]
                    return result
                return hook
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(direction, alpha)))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()

    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs[ids[0]].item()
    top_id = torch.argmax(probs).item()

    return {
        'hijacked': bool(top_id == ids[0]),
        'target_prob': float(target_prob),
    }


def main():
    print("=" * 70)
    print("Phase 98: DMA Attack Surface Map")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # === Test 1: Single-layer vulnerability map ===
    print("\n  Single-layer scan...")
    single_results = []
    for li in range(n_layers):
        total_prob = 0
        n_hijack = 0
        for fact in FACTS:
            res = inject_at_layers(model, tok, device, fact['prompt'], fact['target'], [li])
            total_prob += res['target_prob']
            if res['hijacked']:
                n_hijack += 1
        single_results.append({
            'layer': li,
            'hijack_rate': float(n_hijack / len(FACTS)),
            'mean_prob': float(total_prob / len(FACTS)),
        })
    print(f"    Best single layer: L{max(single_results, key=lambda r: r['mean_prob'])['layer']}")

    # === Test 2: Consecutive 5-layer windows ===
    print("\n  5-layer window scan...")
    window_results = []
    for start in range(0, n_layers - 4):
        layers = list(range(start, start + 5))
        total_prob = 0
        n_hijack = 0
        for fact in FACTS:
            res = inject_at_layers(model, tok, device, fact['prompt'], fact['target'], layers)
            total_prob += res['target_prob']
            if res['hijacked']:
                n_hijack += 1
        window_results.append({
            'start': start,
            'end': start + 4,
            'hijack_rate': float(n_hijack / len(FACTS)),
            'mean_prob': float(total_prob / len(FACTS)),
        })
        if n_hijack > 0:
            print(f"    L{start}-{start+4}: hijack={n_hijack}/{len(FACTS)}, prob={total_prob/len(FACTS):.4f}")

    # === Test 3: Strategic layer combinations ===
    print("\n  Strategic combinations...")
    strategic_configs = [
        ("early", [0, 1, 2, 3, 4]),
        ("early-mid", [3, 6, 9, 12, 15]),
        ("mid", [10, 11, 12, 13, 14]),
        ("mid-late", [13, 16, 19, 22, 25]),
        ("late", [n_layers-5, n_layers-4, n_layers-3, n_layers-2, n_layers-1]),
        ("spread", [0, 7, 14, 21, n_layers-1]),
        ("every3rd", [3, 6, 9, 12, 15]),
        ("last5", list(range(n_layers-5, n_layers))),
    ]

    strategic_results = []
    for name, layers in strategic_configs:
        total_prob = 0
        n_hijack = 0
        for fact in FACTS:
            res = inject_at_layers(model, tok, device, fact['prompt'], fact['target'], layers)
            total_prob += res['target_prob']
            if res['hijacked']:
                n_hijack += 1
        strategic_results.append({
            'name': name,
            'layers': layers,
            'hijack_rate': float(n_hijack / len(FACTS)),
            'mean_prob': float(total_prob / len(FACTS)),
        })
        print(f"    {name}: hijack={n_hijack}/{len(FACTS)}, prob={total_prob/len(FACTS):.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Single-layer vulnerability heatmap
    layer_probs = [r['mean_prob'] for r in single_results]
    axes[0,0].bar(range(n_layers), layer_probs, color='#c0392b', alpha=0.7, edgecolor='black')
    peak_layer = max(single_results, key=lambda r: r['mean_prob'])['layer']
    axes[0,0].axvline(x=peak_layer, color='#f39c12', linestyle='--',
                      label=f'Peak: L{peak_layer}')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Target Prob')
    axes[0,0].set_title('(a) Single-Layer Vulnerability')
    axes[0,0].legend(fontsize=8)

    # (b) 5-layer window scan
    starts = [r['start'] for r in window_results]
    w_probs = [r['mean_prob'] for r in window_results]
    w_rates = [r['hijack_rate'] for r in window_results]
    axes[0,1].plot(starts, w_probs, 'o-', color='#2980b9', markersize=4, label='Target prob')
    ax_b = axes[0,1].twinx()
    ax_b.plot(starts, w_rates, 's-', color='#c0392b', markersize=4, label='Hijack rate')
    axes[0,1].set_xlabel('Window Start Layer')
    axes[0,1].set_ylabel('Target Prob', color='#2980b9')
    ax_b.set_ylabel('Hijack Rate', color='#c0392b')
    axes[0,1].set_title('(b) 5-Layer Window Scan')

    # (c) Strategic combinations
    s_names = [r['name'] for r in strategic_results]
    s_rates = [r['hijack_rate'] for r in strategic_results]
    s_probs = [r['mean_prob'] for r in strategic_results]
    colors_s = ['#27ae60' if r > 0.5 else '#f39c12' if r > 0 else '#c0392b' for r in s_rates]
    axes[0,2].barh(range(len(s_names)), s_rates, color=colors_s, alpha=0.8, edgecolor='black')
    axes[0,2].set_yticks(range(len(s_names)))
    axes[0,2].set_yticklabels(s_names, fontsize=8)
    axes[0,2].set_xlabel('Hijack Rate')
    axes[0,2].set_title('(c) Strategic Combinations')

    # (d) Vulnerability heatmap 2D
    vuln_matrix = np.zeros((n_layers, n_layers))
    for r in single_results:
        vuln_matrix[r['layer'], r['layer']] = r['mean_prob']
    im = axes[1,0].imshow(vuln_matrix, cmap='Reds', aspect='auto')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Layer')
    axes[1,0].set_title('(d) Single-Layer Map')
    plt.colorbar(im, ax=axes[1,0], shrink=0.7)

    # (e) Window scan heatmap
    axes[1,1].fill_between(starts, 0, w_probs, alpha=0.3, color='#2980b9')
    axes[1,1].plot(starts, w_probs, 'o-', color='#2980b9')
    best_window = max(window_results, key=lambda r: r['mean_prob'])
    axes[1,1].axvspan(best_window['start'], best_window['end'],
                      alpha=0.2, color='#c0392b',
                      label=f"Best: L{best_window['start']}-{best_window['end']}")
    axes[1,1].set_xlabel('Window Start')
    axes[1,1].set_ylabel('Target Probability')
    axes[1,1].set_title('(e) Attack Surface')
    axes[1,1].legend(fontsize=8)

    # (f) Summary
    best_single = max(single_results, key=lambda r: r['mean_prob'])
    best_strat = max(strategic_results, key=lambda r: r['hijack_rate'])
    summary = (
        f"Attack Surface Summary\n\n"
        f"Most vulnerable single layer: L{best_single['layer']}\n"
        f"  prob = {best_single['mean_prob']:.4f}\n\n"
        f"Best 5-layer window: L{best_window['start']}-{best_window['end']}\n"
        f"  rate = {best_window['hijack_rate']:.0%}\n\n"
        f"Best strategy: {best_strat['name']}\n"
        f"  rate = {best_strat['hijack_rate']:.0%}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 98: DMA Attack Surface (best window: '
                 f'L{best_window["start"]}-{best_window["end"]}, '
                 f'{best_window["hijack_rate"]:.0%})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase98_attack_surface')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Most vulnerable layer: L{best_single['layer']}")
    print(f"Best 5-layer window: L{best_window['start']}-{best_window['end']} "
          f"({best_window['hijack_rate']:.0%})")
    print(f"Best strategy: {best_strat['name']} ({best_strat['hijack_rate']:.0%})")
    print(f"{'='*70}")

    save_results('phase98_attack_surface', {
        'experiment': 'DMA Attack Surface Map',
        'single_layer': single_results,
        'window_5layer': window_results,
        'strategic': strategic_results,
        'summary': {
            'most_vulnerable_layer': best_single['layer'],
            'best_window': f"L{best_window['start']}-{best_window['end']}",
            'best_window_rate': float(best_window['hijack_rate']),
            'best_strategy': best_strat['name'],
            'best_strategy_rate': float(best_strat['hijack_rate']),
        }
    })


if __name__ == '__main__':
    main()
