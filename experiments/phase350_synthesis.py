# -*- coding: utf-8 -*-
"""
Phase 350: Season 29 Grand Synthesis
=====================================================
Synthesize Season 29 results and update the running total.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import save_results, save_figure

def main():
    print("=" * 70)
    print("Phase 350: Season 29 Grand Synthesis")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

    # Load season 29 results
    s29_phases = {}
    for fname in sorted(glob.glob(os.path.join(results_dir, 'phase34*.json'))):
        bn = os.path.basename(fname)
        phase_num = int(bn.split('_')[0].replace('phase', ''))
        if 344 <= phase_num <= 349:
            with open(fname, 'r') as f:
                data = json.load(f)
            s29_phases[phase_num] = data
            print(f"  Loaded: {bn}")

    all_results = sorted(glob.glob(os.path.join(results_dir, 'phase*.json')))
    total = len(all_results)
    print(f"\n  Total experiments: {total}")

    findings = {}

    if 344 in s29_phases:
        r = s29_phases[344].get('results', {})
        findings['chern_number'] = {s: r[s].get('chern_number', 0) for s in r}
        findings['chern_quantized'] = {s: r[s].get('quantized', False) for s in r}

    if 345 in s29_phases:
        r = s29_phases[345].get('results', {})
        findings['edge_distinct'] = {s: r[s].get('edge_distinct', False) for s in r}
        findings['edge_conductance'] = {s: r[s].get('edge_conductance', 0) for s in r}

    if 346 in s29_phases:
        r = s29_phases[346].get('results', {})
        findings['spectral_dim'] = {s: r[s].get('spectral_dim', 0) for s in r}

    if 347 in s29_phases:
        r = s29_phases[347].get('results', {})
        findings['chiral_present'] = {s: r[s].get('chiral_present', False) for s in r}
        findings['Q_total'] = {s: r[s].get('Q_total', 0) for s in r}

    if 348 in s29_phases:
        r = s29_phases[348].get('results', {})
        findings['susy_unbroken'] = {s: r[s].get('susy_unbroken', False) for s in r}
        findings['avg_pairing'] = {s: r[s].get('avg_pairing', 0) for s in r}

    print(f"\n  Key findings: {len(findings)} categories")

    # Dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"Season 29: Topology & Symmetry ({total} total experiments)",
                fontsize=16, fontweight='bold')

    phase_names = ['Chern', 'Edge', 'Spectral', 'Anomaly', 'SUSY', 'SSB']
    phase_nums = [344, 345, 346, 347, 348, 349]
    available = [p in s29_phases for p in phase_nums]
    bar_colors = ['#2ecc71' if a else '#e74c3c' for a in available]
    axes[0, 0].barh(phase_names, [1 if a else 0 for a in available], color=bar_colors)
    axes[0, 0].set_xlim(0, 1.2)
    axes[0, 0].set_title('(a) Phase Completion', fontweight='bold')

    if 'chern_number' in findings:
        sizes = list(findings['chern_number'].keys())
        axes[0, 1].bar(sizes, [findings['chern_number'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 1].set_title('(b) Chern Number', fontweight='bold')

    if 'spectral_dim' in findings:
        sizes = list(findings['spectral_dim'].keys())
        axes[0, 2].bar(sizes, [findings['spectral_dim'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 2].set_title('(c) Spectral Dimension', fontweight='bold')

    if 'avg_pairing' in findings:
        sizes = list(findings['avg_pairing'].keys())
        axes[1, 0].bar(sizes, [findings['avg_pairing'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 0].set_title('(d) SUSY Pairing', fontweight='bold')

    if 'edge_conductance' in findings:
        sizes = list(findings['edge_conductance'].keys())
        axes[1, 1].bar(sizes, [findings['edge_conductance'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 1].set_title('(e) Edge Conductance', fontweight='bold')

    if 'Q_total' in findings:
        sizes = list(findings['Q_total'].keys())
        axes[1, 2].bar(sizes, [findings['Q_total'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 2].set_title('(f) Topological Charge', fontweight='bold')

    axes[2, 0].axis('off'); axes[2, 1].axis('off')

    txt = f"SEASON 29: {total} exps\n\n"
    for key, vals in findings.items():
        for s, v in vals.items():
            txt += f"  {key}[{s}] = {v}\n"
    axes[2, 2].text(0.5, 0.5, txt[:600], ha='center', va='center',
                   transform=axes[2, 2].transAxes, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[2, 2].axis('off'); axes[2, 2].set_title('(i) Summary')

    for ax in axes.flat:
        if ax.get_visible():
            ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'phase350_synthesis')
    plt.close()

    save_results('phase350_synthesis', {
        'experiment': 'Season 29 Synthesis',
        'total_experiments': total,
        'findings': {k: {s: str(v) for s, v in vals.items()} for k, vals in findings.items()},
    })
    print(f"\n  Synthesis complete: {total} total experiments")


if __name__ == '__main__':
    main()
