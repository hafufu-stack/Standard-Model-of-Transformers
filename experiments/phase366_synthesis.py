# -*- coding: utf-8 -*-
"""
Phase 366: Season 32 Synthesis - Non-Equilibrium Thermodynamics
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
    print("Phase 366: Season 32 Synthesis")
    print("=" * 70)

    s32_files = ['phase361_fdt', 'phase362_jarzynski', 'phase363_crooks',
                 'phase364_onsager', 'phase365_entropy_production']

    all_data = {}
    for name in s32_files:
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
    fig.suptitle("Phase 366: Season 32 Synthesis - Non-Equilibrium Thermodynamics",
                fontsize=14, fontweight='bold')

    # (a) FDT
    ax = axes[0, 0]
    if 'phase361_fdt' in all_data:
        d = all_data['phase361_fdt']['results']
        sizes = list(d.keys())
        fdt_r = [d[s]['fdt_correlation'] for s in sizes]
        ax.bar(sizes, fdt_r, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_ylabel('FDT Correlation')
    ax.set_title('(a) FDT', fontweight='bold')
    ax.grid(alpha=0.3)

    # (b) Jarzynski
    ax = axes[0, 1]
    if 'phase362_jarzynski' in all_data:
        d = all_data['phase362_jarzynski']['results']
        for si, size in enumerate(d.keys()):
            jt = d[size]['jarzynski_test']
            sigmas = [j['sigma'] for j in jt]
            ratios = [j['ratio'] for j in jt]
            ax.plot(sigmas, ratios, 'o-', label=size, lw=2)
        ax.axhline(1.0, color='gray', ls='--')
        ax.set_xscale('log')
        ax.legend()
    ax.set_title('(b) Jarzynski', fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Crooks
    ax = axes[0, 2]
    if 'phase363_crooks' in all_data:
        d = all_data['phase363_crooks']['results']
        sizes = list(d.keys())
        ep = [d[s]['entropy_production_mean'] for s in sizes]
        ax.bar(sizes, ep, color=['#2ecc71', '#f39c12'], alpha=0.8)
        ax.set_ylabel('Entropy Production')
    ax.set_title('(c) Crooks', fontweight='bold')
    ax.grid(alpha=0.3)

    # (d) Onsager
    ax = axes[1, 0]
    if 'phase364_onsager' in all_data:
        d = all_data['phase364_onsager']['results']
        sizes = list(d.keys())
        sym = [d[s]['symmetry_ratio'] for s in sizes]
        corr = [d[s]['onsager_correlation'] for s in sizes]
        x = np.arange(len(sizes))
        ax.bar(x - 0.15, sym, 0.3, label='Symmetry', color='#3498db', alpha=0.8)
        ax.bar(x + 0.15, corr, 0.3, label='Correlation', color='#e74c3c', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sizes)
        ax.legend()
    ax.set_title('(d) Onsager', fontweight='bold')
    ax.grid(alpha=0.3)

    # (e) Entropy production rate
    ax = axes[1, 1]
    if 'phase365_entropy_production' in all_data:
        d = all_data['phase365_entropy_production']['results']
        for size in d:
            rate = d[size]['mean_rate_profile']
            ax.plot(range(len(rate)), rate, 'o-', label=size, markersize=3, lw=1.5)
        ax.axhline(0, color='gray', ls='--')
        ax.legend()
        ax.set_xlabel('Layer')
        ax.set_ylabel('dS/dl')
    ax.set_title('(e) Entropy Production Rate', fontweight='bold')
    ax.grid(alpha=0.3)

    # (f) Summary
    axes[1, 2].axis('off')
    txt = f"SEASON 32 SYNTHESIS\n"
    txt += f"Total: {total_experiments} experiments\n\n"
    txt += "NON-EQUILIBRIUM THERMO:\n"
    if 'phase361_fdt' in all_data:
        d = all_data['phase361_fdt']['results']
        r05 = d.get('0.5B', {}).get('fdt_correlation', 0)
        r15 = d.get('1.5B', {}).get('fdt_correlation', 0)
        txt += f"FDT: r={r05:.2f}/{r15:.2f}\n"
    if 'phase364_onsager' in all_data:
        d = all_data['phase364_onsager']['results']
        s05 = d.get('0.5B', {}).get('symmetry_ratio', 0)
        s15 = d.get('1.5B', {}).get('symmetry_ratio', 0)
        txt += f"Onsager sym: {s05:.2f}/{s15:.2f}\n"
    if 'phase365_entropy_production' in all_data:
        d = all_data['phase365_entropy_production']['results']
        p05 = d.get('0.5B', {}).get('prigogine_ratio', 0)
        p15 = d.get('1.5B', {}).get('prigogine_ratio', 0)
        txt += f"Prigogine: {p05:.2f}/{p15:.2f}\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')

    plt.tight_layout()
    save_figure(fig, 'phase366_synthesis')
    plt.close()

    save_results('phase366_synthesis', {
        'experiment': 'Season 32 Synthesis - Non-Equilibrium Thermodynamics',
        'total_experiments': total_experiments,
        'results': {k: v.get('results', v) for k, v in all_data.items()},
    })


if __name__ == '__main__':
    main()
