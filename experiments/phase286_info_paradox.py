# -*- coding: utf-8 -*-
"""
Phase 286: Information Paradox Resolution
==========================================
S-Qubit Q299 found info is AMPLIFIED (ratio=7.19) not destroyed.
Standard Model P277 found Hawking radiation with T_H > 0.

Test: As information passes through the "black hole" layers
(where T peaks), is it truly destroyed, transformed, or amplified?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPT_PAIRS = [
    ("The cat sat on the mat", "The dog sat on the rug"),
    ("Quantum mechanics is fundamental", "Classical physics is fundamental"),
    ("Water is composed of hydrogen and oxygen", "Air is composed of nitrogen and oxygen"),
    ("The sun rises in the east", "The moon rises in the east"),
]


def track_information_through_layers(model, tok, prompt1, prompt2, device):
    """Track distinguishability of two prompts through all layers."""
    inp1 = tok(prompt1, return_tensors='pt', padding=True, truncation=True).to(device)
    inp2 = tok(prompt2, return_tensors='pt', padding=True, truncation=True).to(device)

    with torch.no_grad():
        out1 = model(**inp1, output_hidden_states=True)
        out2 = model(**inp2, output_hidden_states=True)

    layer_data = []
    for li, (h1, h2) in enumerate(zip(out1.hidden_states, out2.hidden_states)):
        # Take last token representation
        v1 = h1[0, -1, :].float()
        v2 = h2[0, -1, :].float()

        # Distinguishability metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)).item()
        l2_dist = (v1 - v2).norm().item()

        # Information content (entropy of activation distribution)
        p1 = torch.softmax(v1.abs(), dim=-1)
        p2 = torch.softmax(v2.abs(), dim=-1)
        ent1 = -(p1 * torch.log(p1 + 1e-10)).sum().item()
        ent2 = -(p2 * torch.log(p2 + 1e-10)).sum().item()

        # KL divergence between distributions
        kl = torch.nn.functional.kl_div(
            torch.log(p1 + 1e-10), p2, reduction='sum').item()

        layer_data.append({
            'layer': li,
            'cos_sim': round(cos_sim, 4),
            'l2_dist': round(l2_dist, 2),
            'entropy_1': round(ent1, 4),
            'entropy_2': round(ent2, 4),
            'kl_div': round(abs(kl), 4),
            'distinguishability': round(1 - cos_sim, 4),
        })

    # Find "event horizon" (min distinguishability)
    dists = [d['distinguishability'] for d in layer_data]
    event_horizon = int(np.argmin(dists))
    min_dist = min(dists)

    # Information ratio: final/initial distinguishability
    info_ratio = dists[-1] / (dists[0] + 1e-10)

    # Is information amplified after the horizon?
    pre_horizon = np.mean(dists[:event_horizon+1]) if event_horizon > 0 else dists[0]
    post_horizon = np.mean(dists[event_horizon:])
    amplified = post_horizon > pre_horizon

    return {
        'layer_data': layer_data,
        'event_horizon': event_horizon,
        'min_distinguishability': round(min_dist, 4),
        'info_ratio': round(info_ratio, 4),
        'amplified_after_horizon': amplified,
        'pre_horizon_mean': round(float(pre_horizon), 4),
        'post_horizon_mean': round(float(post_horizon), 4),
    }


def main():
    print("=" * 70)
    print("Phase 286: Information Paradox Resolution")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        pair_results = []
        for p1, p2 in PROMPT_PAIRS:
            r = track_information_through_layers(model, tok, p1, p2, device)
            pair_results.append({
                'prompt1': p1[:40],
                'prompt2': p2[:40],
                **r,
            })
            print(f"  '{p1[:20]}' vs '{p2[:20]}':")
            print(f"    Horizon=L{r['event_horizon']}, "
                  f"ratio={r['info_ratio']:.2f}, "
                  f"amplified={r['amplified_after_horizon']}")

        avg_ratio = float(np.mean([r['info_ratio'] for r in pair_results]))
        n_amplified = sum(1 for r in pair_results if r['amplified_after_horizon'])

        verdict = "AMPLIFIED" if avg_ratio > 1.5 else ("PRESERVED" if avg_ratio > 0.8 else "DESTROYED")

        all_results[size] = {
            'pair_results': pair_results,
            'avg_info_ratio': round(avg_ratio, 4),
            'n_amplified': n_amplified,
            'n_total': len(pair_results),
            'verdict': f"INFO {verdict}: ratio={avg_ratio:.2f}, {n_amplified}/{len(pair_results)} amplified",
        }
        print(f"  Verdict: {all_results[size]['verdict']}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Distinguishability through layers (all pairs)
    for size, data in all_results.items():
        for pr in data['pair_results']:
            dists = [d['distinguishability'] for d in pr['layer_data']]
            axes[0, 0].plot(dists, '-', color=colors[size], alpha=0.4, lw=1)
        # Average
        n_layers = min(len(pr['layer_data']) for pr in data['pair_results'])
        avg_dists = [np.mean([pr['layer_data'][i]['distinguishability']
                             for pr in data['pair_results']]) for i in range(n_layers)]
        axes[0, 0].plot(avg_dists, '-', color=colors[size], lw=3, label=f'{size} avg')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Distinguishability (1 - cos_sim)')
    axes[0, 0].set_title('(a) Information Through Layers', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Event horizon positions
    for size, data in all_results.items():
        horizons = [pr['event_horizon'] for pr in data['pair_results']]
        axes[0, 1].hist(horizons, bins=range(max(horizons)+2), alpha=0.6,
                       color=colors[size], label=size)
    axes[0, 1].set_xlabel('Event Horizon (Layer)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('(b) Event Horizon Distribution', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Information ratio
    for size, data in all_results.items():
        ratios = [pr['info_ratio'] for pr in data['pair_results']]
        axes[0, 2].bar(range(len(ratios)), ratios, alpha=0.7, color=colors[size],
                      label=size)
    axes[0, 2].axhline(1.0, color='black', ls='--', label='Preservation')
    axes[0, 2].set_xlabel('Prompt Pair')
    axes[0, 2].set_ylabel('Info Ratio (final/initial)')
    axes[0, 2].set_title('(c) Information Amplification', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) KL divergence profile
    for size, data in all_results.items():
        n_layers = min(len(pr['layer_data']) for pr in data['pair_results'])
        avg_kl = [np.mean([pr['layer_data'][i]['kl_div']
                          for pr in data['pair_results']]) for i in range(n_layers)]
        axes[1, 0].plot(avg_kl, '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('(d) KL Divergence Through Layers', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Pre vs Post horizon
    for size, data in all_results.items():
        pre = [pr['pre_horizon_mean'] for pr in data['pair_results']]
        post = [pr['post_horizon_mean'] for pr in data['pair_results']]
        axes[1, 1].scatter(pre, post, c=colors[size], s=60, label=size, alpha=0.7)
    lim = axes[1, 1].get_xlim()
    axes[1, 1].plot(lim, lim, 'k--', alpha=0.3, label='y=x')
    axes[1, 1].set_xlabel('Pre-Horizon Distinguishability')
    axes[1, 1].set_ylabel('Post-Horizon Distinguishability')
    axes[1, 1].set_title('(e) Pre vs Post Horizon', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "INFORMATION PARADOX RESOLUTION\n\n"
    for size, data in all_results.items():
        txt += f"{size}: {data['verdict']}\n"
        txt += f"  Avg ratio: {data['avg_info_ratio']:.2f}\n"
        txt += f"  Amplified: {data['n_amplified']}/{data['n_total']}\n\n"
    txt += "S-Qubit Q299: ratio=7.19 (AMPLIFIED)\n"
    txt += "P277 Hawking: T_H > 0 (residual)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 286: Information Paradox Resolution",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase286_info_paradox')
    plt.close()

    save_results('phase286_info_paradox', {
        'experiment': 'Information Paradox Resolution',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
