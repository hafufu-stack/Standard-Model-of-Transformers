# -*- coding: utf-8 -*-
"""
Phase 372: Season 33 Synthesis - Predictive Applications
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import save_results, save_figure, RESULTS_DIR


def main():
    print("=" * 70)
    print("Phase 372: Season 33 Synthesis - Applications")
    print("=" * 70)

    s33_files = ['phase367_hallucination', 'phase368_pruning',
                 'phase369_difficulty', 'phase370_ood', 'phase371_tqi']

    all_data = {}
    for name in s33_files:
        path = os.path.join(RESULTS_DIR, f'{name}.json')
        if os.path.exists(path):
            all_data[name] = json.load(open(path))
            print(f'  Loaded: {name}.json')
        else:
            print(f'  MISSING: {name}.json')

    total_files = glob.glob(os.path.join(RESULTS_DIR, 'phase*.json'))
    total_experiments = len(total_files)
    print(f'\n  Total experiments: {total_experiments}')

    # Summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 372: Season 33 Synthesis - Predictive Applications",
                fontsize=14, fontweight='bold')

    # (a) Hallucination AUROC
    ax = axes[0, 0]
    if 'phase367_hallucination' in all_data:
        d = all_data['phase367_hallucination']['results']
        sizes = list(d.keys())
        aurocs = [d[s]['auroc'] for s in sizes]
        ax.bar(sizes, aurocs, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.axhline(0.5, color='gray', ls='--', label='Random')
        ax.set_ylabel('AUROC')
        ax.legend()
    ax.set_title('(a) Hallucination Detection', fontweight='bold')
    ax.grid(alpha=0.3)

    # (b) Pruning KL
    ax = axes[0, 1]
    if 'phase368_pruning' in all_data:
        d = all_data['phase368_pruning']['results']
        for size in d:
            kl = d[size]['layer_kl']
            L0 = d[size]['L0']
            ax.plot(range(len(kl)), kl, 'o-', label=size, markersize=3)
            ax.axvline(L0, ls='--', alpha=0.5)
        ax.legend()
        ax.set_xlabel('Layer')
        ax.set_ylabel('KL Divergence')
    ax.set_title('(b) Layer Pruning Cost', fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Difficulty prediction
    ax = axes[0, 2]
    if 'phase369_difficulty' in all_data:
        d = all_data['phase369_difficulty']['results']
        sizes = list(d.keys())
        r2s = [d[s]['multi_feature_r2'] for s in sizes]
        ax.bar(sizes, r2s, color=['#2ecc71', '#f39c12'], alpha=0.8)
        ax.set_ylabel('R2 (Prediction)')
    ax.set_title('(c) Difficulty Prediction', fontweight='bold')
    ax.grid(alpha=0.3)

    # (d) OOD Detection
    ax = axes[1, 0]
    if 'phase370_ood' in all_data:
        d = all_data['phase370_ood']['results']
        sizes = list(d.keys())
        aurocs = [d[s]['multi_feature_auroc'] for s in sizes]
        ax.bar(sizes, aurocs, color=['#9b59b6', '#e67e22'], alpha=0.8)
        ax.axhline(0.5, color='gray', ls='--')
        ax.set_ylabel('AUROC')
    ax.set_title('(d) OOD Detection', fontweight='bold')
    ax.grid(alpha=0.3)

    # (e) TQI
    ax = axes[1, 1]
    if 'phase371_tqi' in all_data:
        d = all_data['phase371_tqi']['results']
        models = list(d.keys())
        tqis = [d[m]['tqi'] for m in models]
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(models)]
        ax.bar(models, tqis, color=colors, alpha=0.8)
        ax.set_ylabel('TQI')
    ax.set_title('(e) Quality Index', fontweight='bold')
    ax.grid(alpha=0.3)

    # (f) Summary
    axes[1, 2].axis('off')
    txt = f"SEASON 33 SYNTHESIS\n"
    txt += f"Total: {total_experiments} experiments\n\n"
    txt += "APPLICATIONS:\n"
    if 'phase367_hallucination' in all_data:
        d = all_data['phase367_hallucination']['results']
        for s in d:
            txt += f"Halluc AUROC ({s}): {d[s]['auroc']:.2f}\n"
    if 'phase370_ood' in all_data:
        d = all_data['phase370_ood']['results']
        for s in d:
            txt += f"OOD AUROC ({s}): {d[s]['multi_feature_auroc']:.2f}\n"
    if 'phase371_tqi' in all_data:
        d = all_data['phase371_tqi']['results']
        for m in d:
            txt += f"TQI ({m}): {d[m]['tqi']:.3f}\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')

    plt.tight_layout()
    save_figure(fig, 'phase372_synthesis')
    plt.close()

    save_results('phase372_synthesis', {
        'experiment': 'Season 33 Synthesis - Predictive Applications',
        'total_experiments': total_experiments,
        'results': {k: v.get('results', v) for k, v in all_data.items()},
    })


if __name__ == '__main__':
    main()
