# -*- coding: utf-8 -*-
"""
Phase 104: Universality Class Identification
Combine critical exponents from Phase 96 (beta=0.161) and Phase 102
(variance peak) to compute gamma, nu, and check scaling relations.
Compare with known universality classes (2D Ising, mean-field, etc.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import save_results, save_figure


def power_law(x, A, gamma):
    return A * np.abs(x + 0.01) ** (-gamma)


def main():
    print("=" * 70)
    print("Phase 104: Universality Class Identification")
    print("=" * 70)

    # Load Phase 102 data
    p102 = json.load(open("results/phase102_critical_fluctuations.json"))
    results_102 = p102['results']

    # Load Phase 96 data
    p96 = json.load(open("results/phase96_critical_exponent.json"))
    beta = p96['summary']['beta']  # 0.161

    # Load Phase 97 data
    p97 = json.load(open("results/phase97_eta_transition.json"))
    L0 = p97['summary']['L0_sigmoid']  # 21.7
    k_sigmoid = p97['summary']['k_sigmoid']  # 0.662

    print(f"  From Phase 96: beta = {beta:.3f}")
    print(f"  From Phase 97: L0 = {L0:.1f}, k = {k_sigmoid:.3f}")

    # Extract variance profile near transition
    Ls = np.array([r['L'] for r in results_102])
    vars_ = np.array([r['eta_var'] for r in results_102])
    means = np.array([r['eta_mean'] for r in results_102])

    # === Compute gamma from variance scaling ===
    # chi ~ |L - L0|^{-gamma} near L0
    # Use post-transition side (L > L0)
    t_values = []  # reduced "temperature" = (L - L0) / L0
    chi_values = []
    for i, L in enumerate(Ls):
        t = (L - L0) / L0
        if 0.02 < abs(t) < 0.3:
            t_values.append(abs(t))
            chi_values.append(vars_[i])

    gamma = 0
    gamma_r2 = 0
    if len(t_values) >= 3:
        t_arr = np.array(t_values)
        chi_arr = np.array(chi_values)
        try:
            popt, _ = curve_fit(power_law, t_arr, chi_arr, p0=[0.05, 1.0],
                                maxfev=5000, bounds=([0, 0], [10, 5]))
            gamma = popt[1]
            pred = power_law(t_arr, *popt)
            ss_res = np.sum((chi_arr - pred)**2)
            ss_tot = np.sum((chi_arr - np.mean(chi_arr))**2)
            gamma_r2 = 1 - ss_res / (ss_tot + 1e-10)
            print(f"  Fitted gamma = {gamma:.3f} (R2={gamma_r2:.3f})")
        except Exception as e:
            print(f"  Gamma fit failed: {e}")

    # === Compute nu from sigmoid width ===
    # The sigmoid width ~ L0 / k gives the correlation length exponent
    # xi ~ |t|^{-nu}, and sigmoid width = 1/k
    # nu ~ 1 / (k * L0) ... this is a rough estimate
    nu = 1.0 / (k_sigmoid * L0 ** 0.5) if k_sigmoid > 0 else 0
    print(f"  Estimated nu = {nu:.3f}")

    # === Check scaling relations ===
    # Rushbrooke: alpha + 2*beta + gamma = 2
    alpha_rush = 2 - 2*beta - gamma if gamma > 0 else 0

    # Widom: gamma = beta * (delta - 1) => delta = 1 + gamma/beta
    delta = 1 + gamma / (beta + 1e-10) if gamma > 0 else 0

    # Fisher: gamma = nu * (2 - eta_crit)
    eta_crit = 2 - gamma / (nu + 1e-10) if nu > 0 and gamma > 0 else 0

    # Hyperscaling (d=1): 2 - alpha = d * nu => alpha = 2 - nu
    alpha_hyper = 2 - nu if nu > 0 else 0

    print(f"\n  === Critical Exponents ===")
    print(f"  beta  = {beta:.3f}")
    print(f"  gamma = {gamma:.3f}")
    print(f"  nu    = {nu:.3f}")
    print(f"  delta = {delta:.3f}")
    print(f"  alpha (Rushbrooke) = {alpha_rush:.3f}")
    print(f"  eta_crit (Fisher) = {eta_crit:.3f}")

    # === Compare with known universality classes ===
    classes = {
        '2D Ising': {'beta': 0.125, 'gamma': 1.75, 'nu': 1.0, 'delta': 15.0, 'alpha': 0.0},
        '3D Ising': {'beta': 0.326, 'gamma': 1.237, 'nu': 0.630, 'delta': 4.789, 'alpha': 0.110},
        'Mean-field': {'beta': 0.500, 'gamma': 1.000, 'nu': 0.500, 'delta': 3.000, 'alpha': 0.0},
        '2D XY': {'beta': 0.231, 'gamma': 1.0, 'nu': 0.5, 'delta': 5.0, 'alpha': 0.0},
        '3D Heisenberg': {'beta': 0.366, 'gamma': 1.396, 'nu': 0.711, 'delta': 4.815, 'alpha': -0.133},
        'Percolation 2D': {'beta': 0.139, 'gamma': 2.389, 'nu': 1.333, 'delta': 18.2, 'alpha': -0.667},
    }

    measured = {'beta': beta, 'gamma': gamma, 'nu': nu}

    # Distance to each class
    distances = {}
    for name, exps in classes.items():
        # Weighted distance (beta is most reliable)
        d = 0
        n = 0
        if beta > 0:
            d += 3.0 * (beta - exps['beta'])**2  # weight beta 3x
            n += 3
        if gamma > 0:
            d += (gamma - exps['gamma'])**2
            n += 1
        if nu > 0:
            d += (nu - exps['nu'])**2
            n += 1
        distances[name] = float(np.sqrt(d / (n + 1e-10)))

    best_class = min(distances, key=distances.get)
    print(f"\n  Closest universality class: {best_class} (d={distances[best_class]:.4f})")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Variance scaling near L0
    if t_values:
        axes[0,0].scatter(t_values, chi_values, s=80, c='#c0392b', edgecolors='black', zorder=5)
        if gamma > 0 and gamma_r2 > 0:
            t_fit = np.linspace(min(t_values), max(t_values), 100)
            axes[0,0].plot(t_fit, power_law(t_fit, *popt), '--', color='#2980b9', linewidth=2,
                          label=f'$\\gamma = {gamma:.2f}$')
        axes[0,0].set_xlabel('$|t| = |L - L_0|/L_0$')
        axes[0,0].set_ylabel('$\\mathrm{Var}(\\eta)$')
        axes[0,0].set_title(f'(a) Variance Scaling ($\\gamma={gamma:.2f}$)')
        axes[0,0].legend()
    else:
        axes[0,0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                      transform=axes[0,0].transAxes)

    # (b) Order parameter (eta vs t)
    t_all = (Ls - L0) / L0
    axes[0,1].plot(t_all, means, 'o-', color='#8e44ad', markersize=4)
    axes[0,1].axvline(x=0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('$t = (L - L_0)/L_0$')
    axes[0,1].set_ylabel('$\\eta$')
    axes[0,1].set_title(f'(b) Order Parameter ($\\beta={beta:.3f}$)')

    # (c) Distance to universality classes
    class_names = list(distances.keys())
    class_dists = [distances[n] for n in class_names]
    bar_colors = ['#27ae60' if n == best_class else '#bdc3c7' for n in class_names]
    axes[0,2].barh(range(len(class_names)), class_dists, color=bar_colors,
                   alpha=0.8, edgecolor='black')
    axes[0,2].set_yticks(range(len(class_names)))
    axes[0,2].set_yticklabels(class_names, fontsize=9)
    axes[0,2].set_xlabel('Distance')
    axes[0,2].set_title(f'(c) Closest: {best_class}')

    # (d) Critical exponents comparison table
    table_data = []
    for name in [best_class, '2D Ising', 'Mean-field']:
        row = [name]
        for exp in ['beta', 'gamma', 'nu']:
            row.append(f"{classes[name][exp]:.3f}")
        table_data.append(row)
    table_data.append(['LLM (ours)', f'{beta:.3f}',
                       f'{gamma:.3f}' if gamma > 0 else '?',
                       f'{nu:.3f}' if nu > 0 else '?'])
    table = axes[1,0].table(cellText=table_data,
                            colLabels=['Class', '$\\beta$', '$\\gamma$', '$\\nu$'],
                            loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1,0].axis('off')
    axes[1,0].set_title('(d) Critical Exponents')

    # (e) Scaling relation check
    relations = {
        'Rushbrooke\n$\\alpha+2\\beta+\\gamma=2$': abs(alpha_rush + 2*beta + gamma - 2),
        'Widom\n$\\gamma=\\beta(\\delta-1)$': abs(gamma - beta*(delta-1)) if delta > 0 else 0,
    }
    r_names = list(relations.keys())
    r_vals = list(relations.values())
    r_colors = ['#27ae60' if v < 0.5 else '#c0392b' for v in r_vals]
    axes[1,1].bar(range(len(r_names)), r_vals, color=r_colors, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(r_names)))
    axes[1,1].set_xticklabels(r_names, fontsize=8)
    axes[1,1].set_ylabel('Violation')
    axes[1,1].axhline(y=0, color='black', linewidth=0.5)
    axes[1,1].set_title('(e) Scaling Relations')

    # (f) Summary
    summary = (
        f"Universality Class Analysis\n\n"
        f"Measured exponents:\n"
        f"  beta = {beta:.3f}\n"
        f"  gamma = {gamma:.3f}\n"
        f"  nu = {nu:.3f}\n\n"
        f"Closest class: {best_class}\n"
        f"  distance = {distances[best_class]:.4f}\n\n"
        f"Phase transition: {'2nd order' if beta < 1 else '1st order'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 104: Universality Class (closest: {best_class}, '
                 f'$\\beta={beta:.3f}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase104_universality_class')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Closest universality class: {best_class}")
    print(f"Critical exponents: beta={beta:.3f}, gamma={gamma:.3f}, nu={nu:.3f}")
    print(f"{'='*70}")

    save_results('phase104_universality_class', {
        'experiment': 'Universality Class Identification',
        'exponents': {
            'beta': float(beta),
            'gamma': float(gamma),
            'gamma_r2': float(gamma_r2),
            'nu': float(nu),
            'delta': float(delta),
            'alpha_rushbrooke': float(alpha_rush),
            'eta_fisher': float(eta_crit),
        },
        'distances': distances,
        'summary': {
            'closest_class': best_class,
            'distance': float(distances[best_class]),
            'is_2nd_order': bool(beta < 1),
        }
    })


if __name__ == '__main__':
    main()
