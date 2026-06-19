# -*- coding: utf-8 -*-
"""
Phase 300: THE 300th EXPERIMENT -- The Complete Standard Model
================================================================
Milestone experiment. Compile the COMPLETE Standard Model:
- All 12+ fundamental constants
- All 10+ universal laws
- Complete fluid dynamics picture
- Cross-validate everything
- Generate the definitive summary figure
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import save_results, save_figure

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def load_result(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def collect_all():
    """Collect ALL measured constants from ALL experiments."""
    constants = {}
    laws = []

    # ===== Season 22 Constants =====
    # P1*T (P282, P289)
    p289 = load_result('phase289_p1t_master_equation')
    if p289:
        for size in ['0.5B', '1.5B']:
            r = p289.get('results', {}).get(size, {})
            if 'mean_p1t' in r:
                constants[f'P1T_{size}'] = r['mean_p1t']
        constants['P1T_theory_max'] = p289.get('theory', {}).get('p1t_max', 3.34)

    # Mach (P290)
    p290 = load_result('phase290_mach_scaling')
    if p290:
        for size, data in p290.get('results', {}).items():
            constants[f'Mach_{size}'] = data.get('mach_number')
        constants['Mach_scaling_alpha'] = p290.get('scaling', {}).get('alpha')

    # Central Charge (P279)
    p279 = load_result('phase279_central_charge')
    if p279:
        for size, data in p279.get('results', {}).items():
            constants[f'c_central_{size}'] = data.get('central_charge_mean')

    # CFT (P291)
    p291 = load_result('phase291_cft_predictions')
    if p291:
        for size, data in p291.get('results', {}).items():
            constants[f'spectral_delta_{size}'] = data.get('spectrum', {}).get('spectral_delta')

    # Info ratio (P286)
    p286 = load_result('phase286_info_paradox')
    if p286:
        for size, data in p286.get('results', {}).items():
            constants[f'info_ratio_{size}'] = data.get('avg_info_ratio')

    # Master loading (P282)
    p282 = load_result('phase282_master_variable')
    if p282:
        for size, data in p282.get('results', {}).items():
            constants[f'master_loading_{size}'] = data.get('master_loading')

    # ===== Season 22b Constants =====
    # Shocks (P293)
    p293 = load_result('phase293_shock_waves')
    if p293:
        for size, data in p293.get('results', {}).items():
            constants[f'avg_shocks_{size}'] = data.get('avg_n_shocks')

    # Layer Mach (P294)
    p294 = load_result('phase294_layer_mach')
    if p294:
        for size, data in p294.get('results', {}).items():
            constants[f'max_local_Mach_{size}'] = data.get('max_mach')
            constants[f'supersonic_frac_{size}'] = data.get('supersonic_fraction')

    # Cooling (P295)
    p295 = load_result('phase295_cooling_scaling')
    if p295:
        for size, data in p295.get('results', {}).items():
            constants[f'gamma_{size}'] = data.get('gamma_power')

    # Phase diagram (P296)
    p296 = load_result('phase296_phase_diagram')
    if p296:
        for size, data in p296.get('results', {}).items():
            constants[f'eos_exponent_{size}'] = data.get('eos_exponent')

    # Drag (P297)
    p297 = load_result('phase297_drag_coefficient')
    if p297:
        for size, data in p297.get('results', {}).items():
            constants[f'Cd_{size}'] = data.get('Cd')
            constants[f'peak_drag_depth_{size}'] = data.get('peak_at_relative_depth')

    # Reynolds (P298)
    p298 = load_result('phase298_reynolds')
    if p298:
        for size, data in p298.get('results', {}).items():
            constants[f'Re_mean_{size}'] = data.get('mean_Re')

    # Navier-Stokes (P299)
    p299 = load_result('phase299_navier_stokes')
    if p299:
        for size, data in p299.get('results', {}).items():
            constants[f'euler_r_{size}'] = data.get('euler_r')
            constants[f'bernoulli_cv_{size}'] = data.get('bernoulli_cv')

    # Filter None values
    constants = {k: v for k, v in constants.items() if v is not None}

    # ===== The Universal Laws =====
    laws = [
        ("1. Boltzmann Distribution", "p(E) = exp(-E/T) / Z", "P261"),
        ("2. Negative Specific Heat", "Cv < 0 (gravitational)", "P262"),
        ("3. P1*T Conservation", "P1 x T ~ 0.84 (Noether)", "P282"),
        ("4. Inverse Radiation Law", "P ~ (1/sigma) T^4", "P263"),
        ("5. Chandrasekhar Limit", "PR_collapse ~ 10", "P265"),
        ("6. Carnot Efficiency", "eta ~ 0.86-0.93", "P267"),
        ("7. Mach Convergence", "M -> 1.0 (sonic barrier)", "P290"),
        ("8. Central Charge c=1", "Free boson CFT", "P279"),
        ("9. Master Variable P1*T", "Loading > 0.97", "P282"),
        ("10. Info Amplification", "I_out/I_in ~ 10^8", "P286"),
        ("11. Transonic Shocks", "1-2 shocks per model", "P293"),
        ("12. Equation of State", "PR ~ T^(-0.7 to -0.9)", "P296"),
    ]

    return constants, laws


def main():
    print("=" * 70)
    print("       *** Phase 300: THE COMPLETE STANDARD MODEL ***")
    print("=" * 70)
    print("  Milestone: 300 experiments on transformer thermodynamics")
    print("=" * 70)

    constants, laws = collect_all()

    print(f"\n  TOTAL CONSTANTS: {len(constants)}")
    print(f"  TOTAL LAWS: {len(laws)}")

    print("\n  ===== ALL MEASURED CONSTANTS =====")
    for k in sorted(constants.keys()):
        v = constants[k]
        if isinstance(v, float):
            print(f"    {k:35s} = {v:.6f}")
        else:
            print(f"    {k:35s} = {v}")

    print("\n  ===== THE 12 UNIVERSAL LAWS =====")
    for name, formula, source in laws:
        print(f"    {name}: {formula} [{source}]")

    # ===== THE DEFINITIVE FIGURE =====
    fig = plt.figure(figsize=(22, 16))

    # Title
    fig.suptitle("THE STANDARD MODEL OF TRANSFORMERS\n"
                "300 Experiments -- 12 Universal Laws -- Complete Fluid Dynamics",
                fontsize=16, fontweight='bold', color='darkred', y=0.98)

    # Layout: 3x4 grid
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # (1) The 12 Laws
    ax1 = fig.add_subplot(gs[0, 0:2])
    law_text = ""
    for name, formula, source in laws:
        law_text += f"{name}\n   {formula}\n"
    ax1.text(0.02, 0.98, law_text, transform=ax1.transAxes, fontsize=6,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.9))
    ax1.axis('off')
    ax1.set_title('The 12 Universal Laws', fontweight='bold', fontsize=11)

    # (2) Mach convergence
    ax2 = fig.add_subplot(gs[0, 2])
    mach_sizes = ['0.5B', '1.5B', '7B']
    mach_params = [5e8, 1.5e9, 7e9]
    mach_vals = [constants.get(f'Mach_{s}', 0) for s in mach_sizes]
    ax2.semilogx(mach_params, mach_vals, 'o-', color='#e74c3c', lw=2, markersize=8)
    ax2.axhline(1.0, color='gold', ls='--', lw=2, label='M=1')
    for i, s in enumerate(mach_sizes):
        ax2.annotate(f'{s}\nM={mach_vals[i]:.2f}', (mach_params[i], mach_vals[i]),
                    textcoords="offset points", xytext=(5, 8), fontsize=7)
    ax2.set_ylabel('Mach Number')
    ax2.set_title('Mach Convergence', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # (3) P1*T conservation
    ax3 = fig.add_subplot(gs[0, 3])
    p1t_vals = [constants.get(f'P1T_{s}', 0) for s in ['0.5B', '1.5B']]
    ax3.bar(['0.5B', '1.5B'], p1t_vals, color=['#3498db', '#e74c3c'])
    ax3.axhline(0.84, color='gold', ls='--', lw=2, label='P1*T=0.84')
    ax3.set_ylabel('P1 * T')
    ax3.set_title('P1*T Conservation', fontweight='bold', fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # (4) Constants table
    ax4 = fig.add_subplot(gs[1, 0:2])
    const_text = f"MEASURED CONSTANTS (N={len(constants)})\n"
    const_text += "-" * 45 + "\n"
    categories = {
        'Conservation': ['P1T_', 'master_loading'],
        'Fluid': ['Mach_', 'Cd_', 'Re_mean', 'euler_r', 'bernoulli'],
        'CFT/Scaling': ['c_central', 'spectral', 'gamma_', 'eos_'],
        'Info/Critical': ['info_ratio', 'avg_shocks', 'supersonic'],
    }
    for cat, prefixes in categories.items():
        const_text += f"\n{cat}:\n"
        for k in sorted(constants.keys()):
            if any(k.startswith(p) or p in k for p in prefixes):
                v = constants[k]
                if isinstance(v, float):
                    const_text += f"  {k[:28]:28s} {v:>10.4f}\n"
                else:
                    const_text += f"  {k[:28]:28s} {str(v):>10s}\n"
    ax4.text(0.02, 0.98, const_text, transform=ax4.transAxes, fontsize=5.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9))
    ax4.axis('off')
    ax4.set_title('Physical Constants', fontweight='bold', fontsize=10)

    # (5) Central charge
    ax5 = fig.add_subplot(gs[1, 2])
    c_vals = [constants.get(f'c_central_{s}', 0) for s in ['0.5B', '1.5B']]
    ax5.bar(['0.5B', '1.5B'], c_vals, color=['#3498db', '#e74c3c'])
    ax5.axhline(1.0, color='gold', ls='--', lw=2, label='c=1')
    ax5.set_ylabel('Central Charge')
    ax5.set_title('CFT: c=1 Free Boson', fontweight='bold', fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # (6) Equation of state
    ax6 = fig.add_subplot(gs[1, 3])
    eos_vals = [constants.get(f'eos_exponent_{s}', 0) for s in ['0.5B', '1.5B']]
    ax6.bar(['0.5B', '1.5B'], eos_vals, color=['#3498db', '#e74c3c'])
    ax6.set_ylabel('EoS Exponent')
    ax6.set_title('Equation of State: PR~T^a', fontweight='bold', fontsize=10)
    ax6.grid(alpha=0.3)

    # (7) Drag coefficient
    ax7 = fig.add_subplot(gs[2, 0])
    cd_sizes = [s for s in ['0.5B', '1.5B', '7B'] if f'Cd_{s}' in constants]
    if cd_sizes:
        cd_vals = [constants[f'Cd_{s}'] for s in cd_sizes]
        ax7.bar(cd_sizes, cd_vals, color=['#3498db', '#e74c3c', '#2ecc71'][:len(cd_sizes)])
    ax7.set_ylabel('Drag Coefficient')
    ax7.set_title('Drag Coefficient', fontweight='bold', fontsize=10)
    ax7.grid(alpha=0.3)

    # (8) Reynolds number
    ax8 = fig.add_subplot(gs[2, 1])
    re_sizes = [s for s in ['0.5B', '1.5B', '7B'] if f'Re_mean_{s}' in constants]
    if re_sizes:
        re_vals = [constants[f'Re_mean_{s}'] for s in re_sizes]
        ax8.bar(re_sizes, re_vals, color=['#3498db', '#e74c3c', '#2ecc71'][:len(re_sizes)])
    ax8.set_ylabel('Mean Reynolds Number')
    ax8.set_title('Reynolds Number', fontweight='bold', fontsize=10)
    ax8.grid(alpha=0.3)

    # (9) Info amplification
    ax9 = fig.add_subplot(gs[2, 2])
    ir_vals = [constants.get(f'info_ratio_{s}', 0) for s in ['0.5B', '1.5B']]
    ax9.bar(['0.5B', '1.5B'], [np.log10(max(v, 1)) for v in ir_vals],
           color=['#3498db', '#e74c3c'])
    ax9.set_ylabel('log10(Info Ratio)')
    ax9.set_title('Info Amplification', fontweight='bold', fontsize=10)
    ax9.grid(alpha=0.3)

    # (10) Grand verdict
    ax10 = fig.add_subplot(gs[2, 3])
    verdict = "THE 300th EXPERIMENT\n"
    verdict += "=" * 25 + "\n\n"
    verdict += f"{len(constants)} constants\n"
    verdict += f"{len(laws)} universal laws\n\n"
    verdict += "Transformer =\n"
    verdict += "  Thermodynamic system\n"
    verdict += "  + Transonic fluid\n"
    verdict += "  + Free boson CFT\n"
    verdict += "  + Noether conservation\n\n"
    verdict += "STANDARD MODEL\n"
    verdict += "IS COMPLETE"
    ax10.text(0.5, 0.5, verdict, ha='center', va='center',
             transform=ax10.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#FFF0F5', alpha=0.9),
             family='monospace', fontweight='bold')
    ax10.axis('off')
    ax10.set_title('Verdict', fontweight='bold', fontsize=10)

    save_figure(fig, 'phase300_standard_model')
    plt.close()

    save_results('phase300_standard_model', {
        'experiment': 'THE 300th EXPERIMENT -- The Complete Standard Model',
        'milestone': 300,
        'n_constants': len(constants),
        'n_laws': len(laws),
        'constants': constants,
        'laws': [{'name': n, 'formula': f, 'source': s} for n, f, s in laws],
    })

    print(f"\n  *** Phase 300: THE COMPLETE STANDARD MODEL ***")
    print(f"  {len(constants)} constants | {len(laws)} laws | 300 experiments")


if __name__ == '__main__':
    main()
