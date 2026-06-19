# -*- coding: utf-8 -*-
"""
Phase 274: Thermodynamic Compressibility
==========================================
If P1*T = const is the ideal gas law, we can define the isothermal
compressibility kappa_T = -(1/V)(dV/dP)_T.

Here "volume" ~ participation ratio PR, "pressure" ~ P1.
Measure the response function as prompt complexity varies.
Deviations from ideal behavior reveal "non-ideal gas" effects.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from utils import load_model, save_results, save_figure

# Prompts ordered by expected complexity (simple -> complex)
COMPLEXITY_PROMPTS = [
    ("trivial", "The color of the sky is"),
    ("simple", "Water freezes at zero degrees"),
    ("medium", "The theory of evolution explains how species change over time through natural selection"),
    ("complex", "The relationship between quantum entanglement and spacetime geometry suggests a deep connection"),
    ("expert", "The renormalization group flow of the coupling constants in the Standard Model of particle physics exhibits asymptotic freedom in quantum chromodynamics"),
    ("adversarial", "Colorless green ideas sleep furiously while the square root of negative one contemplates its own existence in a non-Euclidean manifold"),
]


def measure_eos_point(model, tok, prompt, device):
    """Measure P1, T, PR, U at the final layer for a single prompt."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    hs = out.hidden_states[-1][0, -1, :].float()
    U = hs.norm().item()

    # PR from hidden state
    h_sq = hs ** 2
    h_prob = h_sq / (h_sq.sum() + 1e-10)
    PR = 1.0 / (h_prob ** 2).sum().item()

    # T, P1 from logits
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    p1 = probs.max().item()
    t_val = -(probs * torch.log(probs + 1e-10)).sum().item()

    # Top-5 probabilities for distribution shape
    top5 = probs.topk(5).values.tolist()

    return {
        'P1': p1, 'T': t_val, 'PR': PR, 'U': U,
        'P1T': p1 * t_val,
        'top5': [round(x, 4) for x in top5],
    }


def main():
    print("=" * 70)
    print("Phase 274: Thermodynamic Compressibility")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        measurements = []
        for label, prompt in COMPLEXITY_PROMPTS:
            m = measure_eos_point(model, tok, prompt, device)
            m['complexity'] = label
            m['prompt'] = prompt[:60]
            measurements.append(m)
            print(f"  {label:12s}: P1={m['P1']:.3f}, T={m['T']:.3f}, "
                  f"PR={m['PR']:.0f}, P1T={m['P1T']:.3f}")

        # Compute compressibility: kappa = -d(PR)/dP1 * (1/PR)
        # Sort by P1
        sorted_m = sorted(measurements, key=lambda x: x['P1'])
        p1_vals = [m['P1'] for m in sorted_m]
        pr_vals = [m['PR'] for m in sorted_m]
        t_vals = [m['T'] for m in sorted_m]

        # Numerical derivative
        kappas = []
        for i in range(1, len(sorted_m)):
            dp1 = p1_vals[i] - p1_vals[i-1]
            dpr = pr_vals[i] - pr_vals[i-1]
            if abs(dp1) > 1e-6 and pr_vals[i] > 0:
                kappa = -(dpr / dp1) / pr_vals[i]
                kappas.append(kappa)

        mean_kappa = float(np.mean(kappas)) if kappas else 0

        # Fit van der Waals-like equation: (P1 + a/PR^2)(PR - b) = c*T
        # Or simpler: check if P1*T deviates systematically
        p1t_vals = [m['P1T'] for m in measurements]
        p1t_cv = float(np.std(p1t_vals) / (np.mean(p1t_vals) + 1e-10))

        # Virial expansion: P1*T = c0 + c1/PR + c2/PR^2
        pr_array = np.array(pr_vals)
        p1t_array = np.array([m['P1T'] for m in sorted_m])
        if len(pr_array) >= 3:
            try:
                coeffs = np.polyfit(1.0 / pr_array, p1t_array, 2)
                virial_fit = True
            except Exception:
                coeffs = [0, 0, 0]
                virial_fit = False
        else:
            coeffs = [0, 0, 0]
            virial_fit = False

        all_results[size] = {
            'measurements': measurements,
            'mean_kappa': round(mean_kappa, 6),
            'p1t_cv': round(p1t_cv, 4),
            'virial_coeffs': [round(c, 6) for c in coeffs],
            'virial_fit': virial_fit,
        }
        print(f"  Compressibility kappa = {mean_kappa:.4f}")
        print(f"  P1T CV = {p1t_cv:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) P1 vs T (equation of state)
    for size, data in all_results.items():
        m = data['measurements']
        axes[0, 0].scatter([x['T'] for x in m], [x['P1'] for x in m],
                          c=colors[size], s=80, label=size, zorder=5)
        for x in m:
            axes[0, 0].annotate(x['complexity'][:4], (x['T'], x['P1']),
                               fontsize=6, ha='center', va='bottom')
    axes[0, 0].set_xlabel('Temperature T')
    axes[0, 0].set_ylabel('P1 (Pressure)')
    axes[0, 0].set_title('(a) Equation of State: P1 vs T', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) P1*T vs complexity
    for size, data in all_results.items():
        m = data['measurements']
        labels = [x['complexity'] for x in m]
        vals = [x['P1T'] for x in m]
        axes[0, 1].plot(range(len(vals)), vals, 'o-', color=colors[size],
                       label=f'{size} (CV={data["p1t_cv"]:.3f})')
    axes[0, 1].axhline(0.84, color='red', ls='--', label='P1T=0.84')
    axes[0, 1].set_xticks(range(len(COMPLEXITY_PROMPTS)))
    axes[0, 1].set_xticklabels([c[0][:6] for c in COMPLEXITY_PROMPTS], rotation=45, fontsize=7)
    axes[0, 1].set_ylabel('P1 * T')
    axes[0, 1].set_title('(b) P1*T vs Prompt Complexity', fontweight='bold')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    # (c) PR vs P1 (compressibility curve)
    for size, data in all_results.items():
        m = sorted(data['measurements'], key=lambda x: x['P1'])
        axes[0, 2].plot([x['P1'] for x in m], [x['PR'] for x in m],
                       'o-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('P1 (Pressure)')
    axes[0, 2].set_ylabel('PR (Volume)')
    axes[0, 2].set_title('(c) PV Diagram', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Virial expansion
    for size, data in all_results.items():
        m = sorted(data['measurements'], key=lambda x: x['PR'])
        pr_arr = np.array([x['PR'] for x in m])
        p1t_arr = np.array([x['P1T'] for x in m])
        axes[1, 0].scatter(1.0/pr_arr, p1t_arr, color=colors[size], s=60, label=size)
        if data['virial_fit']:
            x_fit = np.linspace(min(1.0/pr_arr), max(1.0/pr_arr), 50)
            y_fit = np.polyval(data['virial_coeffs'], x_fit)
            axes[1, 0].plot(x_fit, y_fit, '--', color=colors[size], alpha=0.7)
    axes[1, 0].set_xlabel('1/PR')
    axes[1, 0].set_ylabel('P1 * T')
    axes[1, 0].set_title('(d) Virial Expansion', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) U (internal energy) vs T
    for size, data in all_results.items():
        m = data['measurements']
        axes[1, 1].scatter([x['T'] for x in m], [x['U'] for x in m],
                          c=colors[size], s=60, label=size)
    axes[1, 1].set_xlabel('Temperature T')
    axes[1, 1].set_ylabel('Internal Energy U')
    axes[1, 1].set_title('(e) Internal Energy vs Temperature', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "THERMODYNAMIC COMPRESSIBILITY\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  kappa = {data['mean_kappa']:.4f}\n"
        summary += f"  P1T CV = {data['p1t_cv']:.4f}\n"
        summary += f"  Virial: {data['virial_coeffs']}\n\n"
    summary += "Ideal gas: kappa=0, P1T=const"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 274: Thermodynamic Compressibility",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase274_compressibility')
    plt.close()

    save_results('phase274_compressibility', {
        'experiment': 'Thermodynamic Compressibility',
        'results': all_results,
    })


if __name__ == '__main__':
    main()
