# -*- coding: utf-8 -*-
"""
Phase 340: Season 27 Grand Synthesis
=====================================================
Synthesize all Season 27 results (Phase 331-339) into an updated
Standard Model with key metrics and constants.
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
    print("Phase 340: Season 27 Grand Synthesis")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

    # Load all season 27 results
    s27_phases = {}
    for fname in sorted(glob.glob(os.path.join(results_dir, 'phase33*.json'))):
        bn = os.path.basename(fname)
        phase_num = int(bn.split('_')[0].replace('phase', ''))
        if 331 <= phase_num <= 339:
            with open(fname, 'r') as f:
                data = json.load(f)
            s27_phases[phase_num] = data
            print(f"  Loaded: {bn}")

    # Also count all existing results
    all_results = sorted(glob.glob(os.path.join(results_dir, 'phase*.json')))
    total_experiments = len(all_results)
    print(f"\n  Total experiments: {total_experiments}")

    # Extract key findings
    findings = {}

    # P331: Modular Invariance
    if 331 in s27_phases:
        r = s27_phases[331].get('results', {})
        findings['modular_s_inv'] = {s: r[s].get('mean_s_inv', 0) for s in r}
        findings['modular_t_mod'] = {s: r[s].get('mean_t_mod', 0) for s in r}

    # P332: c-theorem
    if 332 in s27_phases:
        r = s27_phases[332].get('results', {})
        findings['c_theorem_mono'] = {s: r[s].get('mono_frac', 0) for s in r}
        findings['c_theorem_holds'] = {s: r[s].get('c_theorem_holds', False) for s in r}

    # P333: Bootstrap
    if 333 in s27_phases:
        r = s27_phases[333].get('results', {})
        findings['bootstrap_delta'] = {s: r[s].get('delta', 0) for s in r}
        findings['bootstrap_crossing'] = {s: r[s].get('crossing_frac', 0) for s in r}
        findings['bootstrap_unitarity'] = {s: r[s].get('unitarity_ok', False) for s in r}

    # P334: Ryu-Takayanagi
    if 334 in s27_phases:
        r = s27_phases[334].get('results', {})
        findings['rt_correlation'] = {s: r[s].get('avg_r_rt', 0) for s in r}

    # P335: ETH
    if 335 in s27_phases:
        r = s27_phases[335].get('results', {})
        findings['eth_smoothness'] = {s: r[s].get('avg_smoothness', 0) for s in r}

    # P336: QEC
    if 336 in s27_phases:
        r = s27_phases[336].get('results', {})
        findings['qec_redundancy'] = {s: r[s].get('avg_redundancy', 0) for s in r}
        findings['qec_cos10'] = {s: r[s].get('avg_cos_10pct', 0) for s in r}

    # P337: Tensor Network
    if 337 in s27_phases:
        r = s27_phases[337].get('results', {})
        findings['tn_mera_r2'] = {s: r[s].get('r2_mera', 0) for s in r}
        findings['tn_better_fit'] = {s: r[s].get('better_fit', '') for s in r}
        findings['tn_corr_length'] = {s: r[s].get('corr_length', 0) for s in r}

    # P338: Scrambling
    if 338 in s27_phases:
        r = s27_phases[338].get('results', {})
        findings['scrambling_layer'] = {s: r[s].get('scrambling_layer', 0) for s in r}
        findings['scrambling_lambda'] = {s: r[s].get('lambda_L', 0) for s in r}

    # P339: MSS
    if 339 in s27_phases:
        r = s27_phases[339].get('results', {})
        findings['mss_ratio'] = {s: r[s].get('ratio', 0) for s in r}
        findings['mss_satisfied'] = {s: r[s].get('bound_satisfied', False) for s in r}

    print(f"\n  Key findings extracted from {len(findings)} categories")

    # Visualization: Grand summary dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"Season 27 Grand Synthesis ({total_experiments} total experiments)",
                fontsize=16, fontweight='bold')

    # Plot 1: Phase summary counts
    phase_names = ['Modular', 'c-theorem', 'Bootstrap', 'RT', 'ETH', 'QEC', 'TN', 'Scramble', 'MSS']
    phase_nums = list(range(331, 340))
    available = [p in s27_phases for p in phase_nums]
    colors_status = ['#2ecc71' if a else '#e74c3c' for a in available]
    axes[0, 0].barh(phase_names, [1 if a else 0 for a in available], color=colors_status)
    axes[0, 0].set_xlim(0, 1.2)
    axes[0, 0].set_title('(a) Phase Completion', fontweight='bold')

    # Plot 2: Crossing symmetry
    if 'bootstrap_crossing' in findings:
        sizes = list(findings['bootstrap_crossing'].keys())
        vals = [findings['bootstrap_crossing'][s] for s in sizes]
        axes[0, 1].bar(sizes, vals, color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 1].axhline(0.9, color='gold', ls='--', lw=2)
        axes[0, 1].set_title('(b) Crossing Symmetry', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=axes[0, 1].transAxes)

    # Plot 3: QEC redundancy
    if 'qec_redundancy' in findings:
        sizes = list(findings['qec_redundancy'].keys())
        vals = [findings['qec_redundancy'][s] for s in sizes]
        axes[0, 2].bar(sizes, vals, color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[0, 2].set_title('(c) QEC Redundancy', fontweight='bold')
    else:
        axes[0, 2].text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=axes[0, 2].transAxes)

    # Plot 4: Tensor Network type
    if 'tn_mera_r2' in findings:
        sizes = list(findings['tn_mera_r2'].keys())
        mera_vals = [findings['tn_mera_r2'][s] for s in sizes]
        axes[1, 0].bar(sizes, mera_vals, color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 0].set_title('(d) MERA R2', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=axes[1, 0].transAxes)

    # Plot 5: Modular invariance
    if 'modular_t_mod' in findings:
        sizes = list(findings['modular_t_mod'].keys())
        vals = [findings['modular_t_mod'][s] for s in sizes]
        axes[1, 1].bar(sizes, vals, color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 1].set_title('(e) T-Modular Invariance', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=axes[1, 1].transAxes)

    # Plot 6: Scaling dimension Delta
    if 'bootstrap_delta' in findings:
        sizes = list(findings['bootstrap_delta'].keys())
        vals = [findings['bootstrap_delta'][s] for s in sizes]
        axes[1, 2].bar(sizes, vals, color=['#3498db', '#e74c3c'][:len(sizes)])
        axes[1, 2].set_title('(f) Scaling Dimension Delta', fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=axes[1, 2].transAxes)

    # Plot 7-8: empty
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')

    # Plot 9: Text summary
    txt = f"SEASON 27 SYNTHESIS\n"
    txt += f"Total: {total_experiments} experiments\n"
    txt += f"S27: Phase 331-339\n\n"
    if 'bootstrap_unitarity' in findings:
        for s in findings['bootstrap_unitarity']:
            txt += f"{s}: unitarity={'OK' if findings['bootstrap_unitarity'][s] else 'NO'}\n"
    if 'tn_better_fit' in findings:
        for s in findings['tn_better_fit']:
            txt += f"{s}: {findings['tn_better_fit'][s]} network\n"
    if 'qec_cos10' in findings:
        for s in findings['qec_cos10']:
            txt += f"{s}: cos@10%={findings['qec_cos10'][s]:.3f}\n"
    axes[2, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[2, 2].transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[2, 2].axis('off'); axes[2, 2].set_title('(i) Summary')

    for ax in axes.flat:
        if ax.get_visible():
            ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'phase340_synthesis')
    plt.close()

    synthesis = {
        'experiment': 'Season 27 Grand Synthesis',
        'total_experiments': total_experiments,
        'season_27_phases': list(s27_phases.keys()),
        'findings': {k: {s: str(v) for s, v in vals.items()} if isinstance(vals, dict) else vals
                    for k, vals in findings.items()},
    }
    save_results('phase340_synthesis', synthesis)
    print(f"\n  Synthesis complete: {total_experiments} total experiments")


if __name__ == '__main__':
    main()
