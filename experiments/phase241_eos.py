# -*- coding: utf-8 -*-
"""
Phase 241: Thermodynamic Equation of State
=============================================
Derive the equation of state P = f(T, V) for transformers.
P = P1 (order parameter / pressure), T = entropy, V = hidden dim / layer
Map the isotherm and isobar curves.
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

# Diverse prompts spanning temperature range
PROMPTS_BY_DIFFICULTY = {
    'trivial': [
        "One plus one equals", "The sky is", "Water is",
        "The sun is a", "Cats are",
    ],
    'easy': [
        "The capital of France is", "Water freezes at",
        "The largest planet is", "Shakespeare wrote",
        "Gravity pulls objects",
    ],
    'medium': [
        "The fundamental theorem of calculus",
        "Quantum entanglement occurs when",
        "Neural networks learn through",
        "Evolution by natural selection",
        "The Riemann hypothesis states",
    ],
    'hard': [
        "The implications of Goedel incompleteness for",
        "Non-equilibrium thermodynamics predicts that",
        "The AdS CFT correspondence relates",
        "Topological quantum computing uses anyons to",
        "The renormalization group explains universality through",
    ],
    'nonsense': [
        "Purple elephants calculated the square root of",
        "Yesterday tomorrow forgot to remember",
        "Silence tasted exactly like growing",
        "Seven abstract thoughts collided creating",
        "The moon decided to become a",
    ],
}


def measure_eos(model, tok, device, model_name):
    """Measure equation of state: (T, P1, U) at every layer for many prompts."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_points = []  # (T, P1, U, layer_frac, difficulty)
    for diff, prompts in PROMPTS_BY_DIFFICULTY.items():
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            for li, hs in enumerate(out.hidden_states):
                h = hs[0, -1, :].float()
                U = h.norm().item()
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                P1 = float(probs.max().item())
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T = float(S) if not np.isnan(S) else 0
                all_points.append({
                    'T': T, 'P1': P1, 'U': U,
                    'layer': li, 'layer_frac': li / (len(out.hidden_states)-1),
                    'difficulty': diff,
                })

    # Fit equation of state: P1 = a * exp(-b * T) + c
    T_arr = np.array([p['T'] for p in all_points])
    P1_arr = np.array([p['P1'] for p in all_points])
    U_arr = np.array([p['U'] for p in all_points])

    # Exponential fit P1(T)
    try:
        def exp_model(T, a, b, c):
            return a * np.exp(-b * T) + c
        popt, pcov = optimize.curve_fit(exp_model, T_arr, P1_arr,
                                        p0=[0.5, 0.3, 0.1], maxfev=5000)
        r_exp = float(np.corrcoef(P1_arr, exp_model(T_arr, *popt))[0, 1])
    except Exception:
        popt = [0, 0, 0]
        r_exp = 0

    # Power law fit P1(T)
    try:
        def power_model(T, a, b):
            return a * (T + 0.1) ** b
        popt_pow, _ = optimize.curve_fit(power_model, T_arr, P1_arr,
                                          p0=[1.0, -0.5], maxfev=5000)
        r_pow = float(np.corrcoef(P1_arr, power_model(T_arr, *popt_pow))[0, 1])
    except Exception:
        popt_pow = [0, 0]
        r_pow = 0

    # Correlation matrix
    r_TP1, _ = stats.pearsonr(T_arr, P1_arr)
    r_TU, _ = stats.pearsonr(T_arr, U_arr)
    r_P1U, _ = stats.pearsonr(P1_arr, U_arr)

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_points': len(all_points),
        'eos_exp': {'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2]),
                    'r': r_exp},
        'eos_pow': {'a': float(popt_pow[0]), 'b': float(popt_pow[1]), 'r': r_pow},
        'correlations': {'r_TP1': float(r_TP1), 'r_TU': float(r_TU), 'r_P1U': float(r_P1U)},
        'all_points': all_points,
    }


def main():
    print("=" * 70)
    print("Phase 241: Thermodynamic Equation of State")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_eos(model, tok, device, size)
        results[size] = r
        print(f"  {r['n_points']} points")
        print(f"  EoS (exp): P1 = {r['eos_exp']['a']:.3f} * exp(-{r['eos_exp']['b']:.3f} * T) + {r['eos_exp']['c']:.3f} (r={r['eos_exp']['r']:.4f})")
        print(f"  EoS (pow): P1 = {r['eos_pow']['a']:.3f} * T^{r['eos_pow']['b']:.3f} (r={r['eos_pow']['r']:.4f})")
        print(f"  r(T,P1)={r['correlations']['r_TP1']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    diff_colors = {'trivial': '#2ecc71', 'easy': '#3498db', 'medium': '#f39c12',
                   'hard': '#e74c3c', 'nonsense': '#9b59b6'}

    for si, (size, r) in enumerate(results.items()):
        pts = r['all_points']

        # (a,b) P1 vs T colored by difficulty
        ax = axes[0, si]
        for diff in PROMPTS_BY_DIFFICULTY:
            sub = [p for p in pts if p['difficulty'] == diff]
            ax.scatter([p['T'] for p in sub], [p['P1'] for p in sub],
                      c=diff_colors[diff], s=8, alpha=0.4, label=diff)
        # Fit curve
        T_fit = np.linspace(0, max(p['T'] for p in pts), 100)
        try:
            eos = r['eos_exp']
            P1_fit = eos['a'] * np.exp(-eos['b'] * T_fit) + eos['c']
            ax.plot(T_fit, P1_fit, 'k-', lw=2, label=f"fit (r={eos['r']:.3f})")
        except Exception:
            pass
        ax.set_xlabel('T (entropy)'); ax.set_ylabel('P1')
        ax.set_title(f'({chr(97+si)}) Equation of State ({size})')
        ax.legend(fontsize=5, markerscale=2)

    # (c) Isotherms: P1 vs layer at different T bins
    r15 = results[list(results.keys())[-1]]
    pts = r15['all_points']
    T_vals = [p['T'] for p in pts]
    T_bins = np.percentile(T_vals, [10, 30, 50, 70, 90])
    ax = axes[0, 2]
    for i, (lo, hi) in enumerate(zip([0]+list(T_bins), list(T_bins)+[20])):
        sub = [p for p in pts if lo <= p['T'] < hi]
        if len(sub) < 5: continue
        # Average P1 by layer_frac
        from collections import defaultdict
        by_layer = defaultdict(list)
        for p in sub:
            by_layer[p['layer']].append(p['P1'])
        layers = sorted(by_layer.keys())
        avg_P1 = [np.mean(by_layer[l]) for l in layers]
        label = f"T={lo:.1f}-{hi:.1f}"
        ax.plot(layers, avg_P1, '-', lw=1.5, alpha=0.7, label=label)
    ax.set_xlabel('Layer'); ax.set_ylabel('P1')
    ax.set_title('(c) Isotherms')
    ax.legend(fontsize=5)

    # (d) T-U plane
    for diff in PROMPTS_BY_DIFFICULTY:
        sub = [p for p in pts if p['difficulty'] == diff]
        axes[1, 0].scatter([p['T'] for p in sub], [p['U'] for p in sub],
                          c=diff_colors[diff], s=8, alpha=0.3, label=diff)
    axes[1, 0].set_xlabel('T'); axes[1, 0].set_ylabel('U')
    axes[1, 0].set_title(f"(d) T-U Plane (r={r15['correlations']['r_TU']:.3f})")
    axes[1, 0].legend(fontsize=6, markerscale=2)

    # (e) P1-U plane
    for diff in PROMPTS_BY_DIFFICULTY:
        sub = [p for p in pts if p['difficulty'] == diff]
        axes[1, 1].scatter([p['U'] for p in sub], [p['P1'] for p in sub],
                          c=diff_colors[diff], s=8, alpha=0.3, label=diff)
    axes[1, 1].set_xlabel('U'); axes[1, 1].set_ylabel('P1')
    axes[1, 1].set_title(f"(e) U-P1 Plane (r={r15['correlations']['r_P1U']:.3f})")
    axes[1, 1].legend(fontsize=6, markerscale=2)

    # (f) Summary
    summary = "EQUATION OF STATE\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        eos = r['eos_exp']
        summary += f"  P1 = {eos['a']:.2f}*exp(-{eos['b']:.2f}*T)+{eos['c']:.2f}\n"
        summary += f"  r = {eos['r']:.4f}\n"
        c = r['correlations']
        summary += f"  r(T,P1)={c['r_TP1']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 241: Thermodynamic Equation of State",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase241_eos')
    plt.close()
    save_results('phase241_eos', {'experiment': 'Equation of State', 'results': results})


if __name__ == '__main__':
    main()
