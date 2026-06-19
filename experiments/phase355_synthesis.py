# -*- coding: utf-8 -*-
"""
Phase 355: Season 30 Grand Synthesis -- String Theory
=====================================================
Synthesize all Season 30 results (Phase 351-354).
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
    print("Phase 355: Season 30 Grand Synthesis")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    s30_phases = {}
    for fname in sorted(glob.glob(os.path.join(results_dir, 'phase35*.json'))):
        bn = os.path.basename(fname)
        phase_num_str = bn.split('_')[0].replace('phase', '')
        try:
            phase_num = int(phase_num_str)
        except ValueError:
            continue
        if 351 <= phase_num <= 354:
            with open(fname, 'r') as f:
                data = json.load(f)
            s30_phases[phase_num] = data
            print(f"  Loaded: {bn}")

    all_results = sorted(glob.glob(os.path.join(results_dir, 'phase*.json')))
    total = len(all_results)
    print(f"\n  Total experiments: {total}")

    findings = {}
    if 351 in s30_phases:
        r = s30_phases[351].get('results', {})
        findings['regge_r2'] = {s: r[s].get('r2', 0) for s in r}
        findings['alpha_prime'] = {s: r[s].get('alpha_prime', 0) for s in r}
    if 352 in s30_phases:
        r = s30_phases[352].get('results', {})
        findings['mirror_corr'] = {s: r[s].get('avg_mirror', 0) for s in r}
        findings['self_dual'] = {s: r[s].get('self_dual_layer', 0) for s in r}
    if 353 in s30_phases:
        r = s30_phases[353].get('results', {})
        findings['d_compact'] = {s: r[s].get('d_compact', 0) for s in r}
        findings['euler'] = {s: r[s].get('euler', 0) for s in r}
    if 354 in s30_phases:
        r = s30_phases[354].get('results', {})
        findings['r_oc'] = {s: r[s].get('r_oc', 0) for s in r}
        findings['n_stacks'] = {s: r[s].get('n_stacks', 0) for s in r}

    print(f"  Key findings: {len(findings)} categories")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"Season 30: String Theory ({total} total experiments)",
                fontsize=16, fontweight='bold')

    names = ['Regge', 'T-dual', 'CY', 'D-brane']
    nums = [351, 352, 353, 354]
    avail = [p in s30_phases for p in nums]
    bar_c = ['#2ecc71' if a else '#e74c3c' for a in avail]
    axes[0, 0].barh(names, [1 if a else 0 for a in avail], color=bar_c)
    axes[0, 0].set_xlim(0, 1.2)
    axes[0, 0].set_title('(a) Phase Completion', fontweight='bold')

    if 'regge_r2' in findings:
        sizes = list(findings['regge_r2'].keys())
        axes[0, 1].bar(sizes, [findings['regge_r2'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 1].set_title('(b) Regge R2', fontweight='bold')

    if 'mirror_corr' in findings:
        sizes = list(findings['mirror_corr'].keys())
        axes[0, 2].bar(sizes, [findings['mirror_corr'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 2].set_title('(c) Mirror Symmetry', fontweight='bold')

    if 'd_compact' in findings:
        sizes = list(findings['d_compact'].keys())
        axes[1, 0].bar(sizes, [findings['d_compact'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 0].set_title('(d) Compact Dimensions', fontweight='bold')

    if 'r_oc' in findings:
        sizes = list(findings['r_oc'].keys())
        axes[1, 1].bar(sizes, [findings['r_oc'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 1].set_title('(e) Open-Closed Duality', fontweight='bold')

    if 'euler' in findings:
        sizes = list(findings['euler'].keys())
        axes[1, 2].bar(sizes, [findings['euler'][s] for s in sizes],
                      color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 2].set_title('(f) Euler Number', fontweight='bold')

    axes[2, 0].axis('off'); axes[2, 1].axis('off')
    txt = f"SEASON 30: {total} exps\n\n"
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
    save_figure(fig, 'phase355_synthesis')
    plt.close()

    save_results('phase355_synthesis', {
        'experiment': 'Season 30 Synthesis',
        'total_experiments': total,
        'findings': {k: {s: str(v) for s, v in vals.items()} for k, vals in findings.items()},
    })
    print(f"\n  Season 30 synthesis complete: {total} total experiments")

if __name__ == '__main__':
    main()
