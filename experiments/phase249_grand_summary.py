# -*- coding: utf-8 -*-
"""
Phase 249: Grand Unified Thermodynamic Summary
=================================================
Compile ALL thermodynamic measurements across phases 238-248.
Create the definitive summary visualization: one figure to rule them all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import RESULTS_DIR, save_figure, save_results

# Load all results
def jload(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    return None


def main():
    print("=" * 70)
    print("Phase 249: Grand Unified Thermodynamic Summary")
    print("=" * 70)

    # Load all available results
    phases = {}
    for f in os.listdir(RESULTS_DIR):
        if f.startswith('phase') and f.endswith('.json'):
            name = f.replace('.json', '')
            try:
                phases[name] = jload(f)
            except Exception:
                pass

    print(f"  Loaded {len(phases)} phase results")

    # Extract key metrics for summary
    summary_data = {}

    # Phase 238: Instruct vs Base
    d = jload('phase238_instruct_vs_base.json')
    if d:
        summary_data['instruct'] = {}
        for name, r in d.get('results', {}).items():
            summary_data['instruct'][name] = {
                'T_final': r.get('T_final', 0),
                'P1_final': r.get('P1_final', 0),
                'rho_S': r.get('rho_S', 0),
            }

    # Phase 239: Generation
    d = jload('phase239_generation.json')
    if d:
        summary_data['generation'] = {}
        for size, r in d.get('results', {}).items():
            runs = r.get('runs', [])
            if runs:
                summary_data['generation'][size] = {
                    'T_mean': float(np.mean([run['T_mean'] for run in runs])),
                    'corr_time': float(np.mean([run['corr_time'] for run in runs])),
                    'rho_T': float(np.mean([run['rho_T'] for run in runs])),
                }

    # Phase 241: EoS
    d = jload('phase241_eos.json')
    if d:
        summary_data['eos'] = {}
        for size, r in d.get('results', {}).items():
            eos = r.get('eos_exp', {})
            summary_data['eos'][size] = {
                'a': eos.get('a', 0), 'b': eos.get('b', 0),
                'c': eos.get('c', 0), 'r': eos.get('r', 0),
            }

    # Phase 242: Work-Heat
    d = jload('phase242_work_heat.json')
    if d:
        summary_data['work_heat'] = {}
        for size, r in d.get('results', {}).items():
            summary_data['work_heat'][size] = {
                'efficiency': r.get('overall_eff', 0),
                'total_W': r.get('total_W', 0),
                'total_Q': r.get('total_Q', 0),
            }

    # Phase 244: Maxwell
    d = jload('phase244_maxwell.json')
    if d:
        summary_data['maxwell'] = {}
        for size, r in d.get('results', {}).items():
            summary_data['maxwell'][size] = {
                'maxwell1_r': r.get('maxwell1', {}).get('r', 0),
                'maxwell2_r': r.get('maxwell2', {}).get('r', 0),
                'first_law_r': r.get('first_law', {}).get('r', 0),
            }

    # Phase 245: Fluctuation
    d = jload('phase245_fluctuation.json')
    if d:
        summary_data['fluctuation'] = {}
        for size, r in d.get('results', {}).items():
            summary_data['fluctuation'][size] = {
                'alpha': r.get('alpha_T', 0),
                'r_fdt': r.get('r_fdt', 0),
                'kurtosis': r.get('kurtosis_T', 0),
            }

    # Phase 246: Linear Response
    d = jload('phase246_linear_response.json')
    if d:
        summary_data['linear_response'] = {}
        for size, r in d.get('results', {}).items():
            summary_data['linear_response'][size] = {
                'r_linear': r.get('r_linear', 0),
            }

    # === Visualization: Grand Summary ===
    fig = plt.figure(figsize=(20, 14))

    # Use a grid spec for a more complex layout
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # (1) Arrow/Phase Transition Matrix
    ax1 = fig.add_subplot(gs[0, 0:2])
    # Compile arrow strengths
    arrow_data = {}
    for f in ['phase229_grand_synthesis', 'phase230_universality', 
              'phase237_universality_classes']:
        d = jload(f'{f}.json')
        if d:
            for size, r in d.get('results', {}).items():
                if isinstance(r, dict) and 'rho_S' in r:
                    arrow_data[f'{f[-12:]}_{size}'] = r['rho_S']
    
    if arrow_data:
        names = list(arrow_data.keys())[:10]
        vals = [arrow_data[n] for n in names]
        ax1.barh(range(len(names)), vals, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels([n[:20] for n in names], fontsize=6)
    ax1.set_xlabel('Arrow Strength (rho_S)')
    ax1.set_title('Historical Arrow Strengths', fontsize=10, fontweight='bold')

    # (2) EoS P1 = f(T)
    ax2 = fig.add_subplot(gs[0, 2:4])
    d = jload('phase241_eos.json')
    if d:
        for size, r in d.get('results', {}).items():
            pts = r.get('all_points', [])
            T = [p['T'] for p in pts]
            P1 = [p['P1'] for p in pts]
            ax2.scatter(T, P1, s=5, alpha=0.3, label=size)
    ax2.set_xlabel('T'); ax2.set_ylabel('P1')
    ax2.set_title('Equation of State: P1(T)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)

    # (3) Generation dynamics
    ax3 = fig.add_subplot(gs[1, 0])
    d = jload('phase239_generation.json')
    if d:
        for size, r in d.get('results', {}).items():
            for ri, run in enumerate(r.get('runs', [])[:3]):
                T_vals = [s['T'] for s in run['step_data']]
                ax3.plot(range(len(T_vals)), T_vals, '-', alpha=0.5, lw=1)
    ax3.set_xlabel('Step'); ax3.set_ylabel('T')
    ax3.set_title('Gen. Dynamics', fontsize=9, fontweight='bold')

    # (4) Work-Heat decomposition
    ax4 = fig.add_subplot(gs[1, 1])
    d = jload('phase242_work_heat.json')
    if d:
        for size, r in d.get('results', {}).items():
            dW = r.get('mean_dW', [])
            dQ = r.get('mean_dQ', [])
            ax4.plot(range(len(dW)), dW, '-', lw=1.5, label=f'dW({size})')
            ax4.plot(range(len(dQ)), dQ, '--', lw=1, label=f'dQ({size})')
    ax4.set_xlabel('Layer'); ax4.set_ylabel('Energy')
    ax4.set_title('Work & Heat', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=5)

    # (5) Fluctuation PSD
    ax5 = fig.add_subplot(gs[1, 2])
    d = jload('phase245_fluctuation.json')
    if d:
        for size, r in d.get('results', {}).items():
            freq = np.array(r.get('freq_T', []))
            psd = np.array(r.get('psd_T', []))
            mask = freq > 0
            if mask.any():
                ax5.loglog(freq[mask], psd[mask], '-o', markersize=3, lw=1.5, label=size)
    ax5.set_xlabel('Freq'); ax5.set_ylabel('PSD')
    ax5.set_title('1/f Spectrum', fontsize=9, fontweight='bold')
    ax5.legend(fontsize=7)

    # (6) Position thermodynamics heatmap
    ax6 = fig.add_subplot(gs[1, 3])
    d = jload('phase240_position.json')
    if d:
        r = d.get('results', {})
        last_key = list(r.keys())[-1] if r else None
        if last_key:
            fields = r[last_key].get('fields', [])
            if fields:
                T_field = np.array(fields[0].get('T_field', []))
                if T_field.size > 0:
                    im = ax6.imshow(T_field, aspect='auto', cmap='hot', origin='lower')
                    fig.colorbar(im, ax=ax6, shrink=0.7)
    ax6.set_xlabel('Position'); ax6.set_ylabel('Layer')
    ax6.set_title('T-Field', fontsize=9, fontweight='bold')

    # (7) Grand Summary Table
    ax7 = fig.add_subplot(gs[2, :])
    table_data = []
    headers = ['Experiment', '0.5B', '1.5B', 'Key Finding']
    
    # Build table
    rows = [
        ('Instruct Effect', 
         f"T_f={summary_data.get('instruct',{}).get('Base-0.5B',{}).get('T_final','?'):.2f}" if summary_data.get('instruct',{}).get('Base-0.5B') else 'N/A',
         f"T_f={summary_data.get('instruct',{}).get('Base-1.5B',{}).get('T_final','?'):.2f}" if summary_data.get('instruct',{}).get('Base-1.5B') else 'N/A',
         'RLHF lowers T'),
        ('Generation',
         f"rho_T={summary_data.get('generation',{}).get('0.5B',{}).get('rho_T',0):.3f}",
         f"rho_T={summary_data.get('generation',{}).get('1.5B',{}).get('rho_T',0):.3f}",
         'T decreases during gen'),
        ('EoS fit r',
         f"r={summary_data.get('eos',{}).get('0.5B',{}).get('r',0):.3f}",
         f"r={summary_data.get('eos',{}).get('1.5B',{}).get('r',0):.3f}",
         'P1 ~ exp(-bT)'),
        ('Efficiency',
         f"{summary_data.get('work_heat',{}).get('0.5B',{}).get('efficiency',0):.3f}",
         f"{summary_data.get('work_heat',{}).get('1.5B',{}).get('efficiency',0):.3f}",
         'Work-heat balance'),
        ('1/f exponent',
         f"a={summary_data.get('fluctuation',{}).get('0.5B',{}).get('alpha',0):.2f}",
         f"a={summary_data.get('fluctuation',{}).get('1.5B',{}).get('alpha',0):.2f}",
         '1/f noise confirmed'),
        ('Linear Response',
         f"r={summary_data.get('linear_response',{}).get('0.5B',{}).get('r_linear',0):.3f}",
         f"r={summary_data.get('linear_response',{}).get('1.5B',{}).get('r_linear',0):.3f}",
         'Non-linear regime'),
    ]
    
    table = ax7.table(cellText=rows, colLabels=headers, loc='center',
                      cellLoc='center', colColours=['#2c3e50']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    # Color headers
    for j in range(4):
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax7.axis('off')
    ax7.set_title('Grand Unified Summary', fontsize=12, fontweight='bold', pad=20)

    fig.suptitle("Phase 249: Standard Model of Transformers - Grand Thermodynamic Summary",
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, 'phase249_grand_summary')
    plt.close()
    
    save_results('phase249_grand_summary', {
        'experiment': 'Grand Unified Summary',
        'summary': summary_data,
        'total_phases_loaded': len(phases),
    })

    # Print grand summary
    print("\n" + "=" * 70)
    print("  GRAND UNIFIED SUMMARY")
    print("=" * 70)
    for key, data in summary_data.items():
        print(f"\n  {key}:")
        if isinstance(data, dict):
            for sub, vals in data.items():
                if isinstance(vals, dict):
                    metrics = ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' 
                                      for k, v in vals.items())
                    print(f"    {sub}: {metrics}")


if __name__ == '__main__':
    main()
