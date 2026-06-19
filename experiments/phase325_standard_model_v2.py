# -*- coding: utf-8 -*-
"""
Phase 325: Grand Unified Summary v2 -- 325 Experiments
========================================================
Update the Extended Standard Model with Season 24 discoveries.
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

def main():
    print("=" * 70)
    print("Phase 325: Grand Unified Summary v2")
    print("=" * 70)

    # Count all results
    all_phases = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith('phase') and fname.endswith('.json'):
            all_phases.append(fname.replace('.json', ''))
    all_phases.sort()
    print(f"  Total result files: {len(all_phases)}")

    # Compile ALL Laws
    laws = [
        # Thermodynamics
        ("1. Boltzmann Distribution", "p(E) = exp(-E/T)/Z"),
        ("2. Negative Specific Heat", "Cv < 0 (gravitational)"),
        ("3. Stefan-Boltzmann Variant", "P ~ (1/sigma) T^4"),
        ("4. Chandrasekhar Limit", "PR_collapse ~ 10"),
        ("5. Carnot Efficiency", "eta ~ 0.86-0.93"),
        ("6. P1*T Conservation", "P1 x T ~ 0.84"),
        ("7. Master Variable", "Loading > 0.97"),
        ("8. Info Amplification", "I_out/I_in ~ 10^8"),
        # Fluid
        ("9. Mach Convergence", "M -> 1.0 (sonic barrier)"),
        ("10. Transonic Shocks", "1-2 shocks per model"),
        ("11. Euler Equation", "r = 0.57"),
        ("12. Equation of State", "PR ~ T^(-0.7)"),
        # QFT / CFT
        ("13. Central Charge c=1", "Free boson CFT"),
        ("14. RG IR Fixed Point", "beta(IR) ~ 0"),
        ("15. Confinement", "sigma > 0 (area law)"),
        ("16. OPE Structure", "R2 = 0.86"),
        ("17. Volume Law Entropy", "S ~ k (critical)"),
        # SSB
        ("18. Spontaneous Symmetry Breaking", "Gini increase > 0.1"),
        ("19. Goldstone Modes", "3-5 massless modes"),
        # Information Geometry
        ("20. Positive Curvature", "R > 0 (spherical)"),
        ("21. Berry Phase", "phi_B ~ 11.3 (constant)"),
        # Quantum Effects
        ("22. Unruh Effect", "T = c_U * a (R2=0.81)"),
        ("23. Schwinger Pairs", "21% sign-flip rate"),
        ("24. Vacuum Energy", "E_vac >> E_real"),
    ]

    # Count constants from recent results
    n_constants = 86  # from Phase 312
    recent_phases = [
        'phase313_fisher', 'phase314_topology', 'phase315_tunneling',
        'phase316_vacuum', 'phase317_casimir', 'phase318_lamb',
        'phase319_schwinger', 'phase320_unruh', 'phase321_bekenstein',
        'phase322_regge', 'phase323_anomaly', 'phase324_deconfinement',
    ]
    for pname in recent_phases:
        data = load_result(pname)
        if data and 'results' in data:
            for size, size_data in data['results'].items():
                if isinstance(size_data, dict):
                    n_constants += len([k for k, v in size_data.items()
                                       if isinstance(v, (int, float)) and not isinstance(v, bool)])

    print(f"  Total laws: {len(laws)}")
    print(f"  Total constants (est): {n_constants}")

    # The Grand Figure v2
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("THE STANDARD MODEL OF TRANSFORMERS v2\n"
                f"325 Experiments | {len(laws)} Universal Laws | ~{n_constants} Constants",
                fontsize=16, fontweight='bold', color='darkred', y=0.99)

    gs = fig.add_gridspec(5, 4, hspace=0.45, wspace=0.35)

    # Row 0: Laws 1-12
    ax = fig.add_subplot(gs[0, :2])
    txt = ""
    for name, formula in laws[:12]:
        txt += f"{name}: {formula}\n"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=5.5,
           va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.9))
    ax.axis('off'); ax.set_title('Laws 1-12: Thermo & Fluid', fontweight='bold', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2:])
    txt2 = ""
    for name, formula in laws[12:]:
        txt2 += f"{name}: {formula}\n"
    ax2.text(0.02, 0.98, txt2, transform=ax2.transAxes, fontsize=5.5,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9))
    ax2.axis('off'); ax2.set_title('Laws 13-24: QFT & Quantum Gravity', fontweight='bold', fontsize=9)

    # Row 1: Key bar charts
    colors_2 = ['#3498db', '#e74c3c']

    # Mach
    ax3 = fig.add_subplot(gs[1, 0])
    p300 = load_result('phase300_standard_model')
    if p300 and 'constants' in p300:
        mach_vals = [p300['constants'].get(f'Mach_{s}', 0) for s in ['0.5B', '1.5B']]
        ax3.bar(['0.5B', '1.5B'], mach_vals, color=colors_2)
    ax3.axhline(1.0, color='gold', ls='--', lw=2)
    ax3.set_ylabel('M'); ax3.set_title('Mach', fontweight='bold', fontsize=8); ax3.grid(alpha=0.3)

    # Berry phase
    ax4 = fig.add_subplot(gs[1, 1])
    p314 = load_result('phase314_topology')
    if p314 and 'results' in p314:
        bp = [p314['results'][s].get('avg_berry', 0) for s in ['0.5B', '1.5B'] if s in p314['results']]
        ax4.bar(['0.5B', '1.5B'][:len(bp)], bp, color=colors_2[:len(bp)])
    ax4.set_ylabel('Berry'); ax4.set_title('Berry Phase', fontweight='bold', fontsize=8); ax4.grid(alpha=0.3)

    # Unruh R2
    ax5 = fig.add_subplot(gs[1, 2])
    p320 = load_result('phase320_unruh')
    if p320 and 'results' in p320:
        ur = [p320['results'][s].get('unruh_r2', 0) for s in ['0.5B', '1.5B'] if s in p320['results']]
        ax5.bar(['0.5B', '1.5B'][:len(ur)], ur, color=colors_2[:len(ur)])
    ax5.set_ylabel('R2'); ax5.set_title('Unruh R2', fontweight='bold', fontsize=8); ax5.grid(alpha=0.3)

    # Curvature
    ax6 = fig.add_subplot(gs[1, 3])
    p313 = load_result('phase313_fisher')
    if p313 and 'results' in p313:
        rv = [p313['results'][s].get('R_scalar', 0) for s in ['0.5B', '1.5B'] if s in p313['results']]
        ax6.bar(['0.5B', '1.5B'][:len(rv)], rv, color=colors_2[:len(rv)])
    ax6.set_ylabel('R'); ax6.set_title('Scalar Curvature', fontweight='bold', fontsize=8); ax6.grid(alpha=0.3)

    # Row 2: Profiles
    # RG flow beta
    ax7 = fig.add_subplot(gs[2, 0])
    p301 = load_result('phase301_rg_flow')
    if p301 and 'results' in p301:
        for s, d in p301['results'].items():
            if 'avg_beta1' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax7.plot(d['avg_beta1'], '-', color=c, lw=1.5, label=s)
    ax7.axhline(0, color='gold', ls='--', lw=1)
    ax7.set_xlabel('L'); ax7.set_ylabel('beta'); ax7.set_title('RG Flow', fontweight='bold', fontsize=8)
    ax7.legend(fontsize=6); ax7.grid(alpha=0.3)

    # Fisher curvature
    ax8 = fig.add_subplot(gs[2, 1])
    if p313 and 'results' in p313:
        for s, d in p313['results'].items():
            if 'avg_curvature' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax8.plot(d['avg_curvature'], '-', color=c, lw=1.5, label=s)
    ax8.set_xlabel('L'); ax8.set_ylabel('K'); ax8.set_title('Curvature', fontweight='bold', fontsize=8)
    ax8.legend(fontsize=6); ax8.grid(alpha=0.3)

    # OPE
    ax9 = fig.add_subplot(gs[2, 2])
    p305 = load_result('phase305_ope')
    if p305 and 'results' in p305:
        for s, d in p305['results'].items():
            if 'avg_ope_r2' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax9.plot(d['avg_ope_r2'], '-', color=c, lw=1.5, label=s)
    ax9.set_xlabel('L'); ax9.set_ylabel('R2'); ax9.set_title('OPE R2', fontweight='bold', fontsize=8)
    ax9.legend(fontsize=6); ax9.grid(alpha=0.3)

    # Free energy
    ax10 = fig.add_subplot(gs[2, 3])
    p307 = load_result('phase307_partition')
    if p307 and 'results' in p307:
        for s, d in p307['results'].items():
            if 'avg_F' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax10.plot(d['avg_F'], '-', color=c, lw=1.5, label=s)
    ax10.set_xlabel('L'); ax10.set_ylabel('F'); ax10.set_title('Free Energy', fontweight='bold', fontsize=8)
    ax10.legend(fontsize=6); ax10.grid(alpha=0.3)

    # Row 3: More profiles
    ax11 = fig.add_subplot(gs[3, 0])
    p310 = load_result('phase310_ssb')
    if p310 and 'results' in p310:
        for s, d in p310['results'].items():
            if 'avg_gini' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax11.plot(d['avg_gini'], '-', color=c, lw=1.5, label=s)
    ax11.set_xlabel('L'); ax11.set_ylabel('Gini'); ax11.set_title('SSB (Gini)', fontweight='bold', fontsize=8)
    ax11.legend(fontsize=6); ax11.grid(alpha=0.3)

    ax12 = fig.add_subplot(gs[3, 1])
    p309 = load_result('phase309_hawking')
    if p309 and 'results' in p309:
        for s, d in p309['results'].items():
            if 'avg_S_hidden' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax12.plot(d['avg_S_hidden'], '-', color=c, lw=1.5, label=s)
    ax12.set_xlabel('L'); ax12.set_ylabel('S'); ax12.set_title('Hidden Entropy', fontweight='bold', fontsize=8)
    ax12.legend(fontsize=6); ax12.grid(alpha=0.3)

    # Geodesic distance
    ax13 = fig.add_subplot(gs[3, 2])
    if p313 and 'results' in p313:
        for s, d in p313['results'].items():
            if 'avg_geodesic_dist' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax13.plot(d['avg_geodesic_dist'], '-', color=c, lw=1.5, label=s)
    ax13.set_xlabel('L'); ax13.set_ylabel('d_FR'); ax13.set_title('Geodesic Dist', fontweight='bold', fontsize=8)
    ax13.legend(fontsize=6); ax13.grid(alpha=0.3)

    # Deconfinement (if available)
    ax14 = fig.add_subplot(gs[3, 3])
    p324 = load_result('phase324_deconfinement')
    if p324 and 'results' in p324:
        for s, d in p324['results'].items():
            if 'avg_polyakov' in d:
                c = '#3498db' if '0.5' in s else '#e74c3c'
                ax14.plot(d['avg_polyakov'], '-', color=c, lw=1.5, label=s)
    ax14.set_xlabel('L'); ax14.set_ylabel('P'); ax14.set_title('Polyakov Loop', fontweight='bold', fontsize=8)
    ax14.legend(fontsize=6); ax14.grid(alpha=0.3)

    # Row 4: Grand verdict
    ax_v = fig.add_subplot(gs[4, :])
    verdict = (
        "THE STANDARD MODEL OF TRANSFORMERS v2\n"
        "=" * 65 + "\n\n"
        "THERMODYNAMICS: Boltzmann | Cv<0 | Carnot | P1*T=0.84 | Info x10^8\n"
        "FLUID DYNAMICS: Transonic (M->1) | Shocks | Euler eq | EOS\n"
        "QUANTUM FIELD THEORY: c~1 CFT | RG fixed pts | Confinement | OPE (R2=0.86)\n"
        "SYMMETRY: SSB (Gini +0.15) | 3-5 Goldstone modes | Anomaly cancellation\n"
        "INFORMATION GEOMETRY: R>0 (spherical) | Berry ~ 11.3 | Geodesic structure\n"
        "QUANTUM GRAVITY: Unruh T=ca (R2=0.81) | Vacuum E >> real | Bekenstein bound\n\n"
        f"TOTAL: ~{n_constants} constants | {len(laws)} laws | {len(all_phases)} experiments\n"
        "Transformer = Thermodynamic + Transonic + CFT + Confining + Curved Info Geometry"
    )
    ax_v.text(0.5, 0.5, verdict, ha='center', va='center',
             transform=ax_v.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#FFF0F5', alpha=0.9),
             family='monospace', fontweight='bold')
    ax_v.axis('off')

    save_figure(fig, 'phase325_standard_model_v2')
    plt.close()

    save_results('phase325_standard_model_v2', {
        'experiment': 'Standard Model v2',
        'n_experiments': len(all_phases),
        'n_laws': len(laws),
        'n_constants_estimate': n_constants,
        'laws': [{'name': n, 'formula': f} for n, f in laws],
    })
    print(f"\n  Standard Model v2: ~{n_constants} constants | {len(laws)} laws | {len(all_phases)} experiments")

if __name__ == '__main__':
    main()
