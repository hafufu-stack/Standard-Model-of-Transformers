# -*- coding: utf-8 -*-
"""
Phase 217: Universality Class Determination
=============================================
Combine all critical exponents from Phases 212-216 and determine
which universality class the transformer phase transition belongs to.

Compare measured exponents with:
- Mean-field (beta=0.5, gamma=1.0, nu=0.5)
- 2D Ising (beta=0.125, gamma=1.75, nu=1.0)
- 3D Ising (beta=0.326, gamma=1.237, nu=0.630)
- 3D XY (beta=0.345, gamma=1.316, nu=0.671)
- 3D Heisenberg (beta=0.365, gamma=1.386, nu=0.707)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, RESULTS_DIR

MODEL_SIZES = ['0.5B', '1.5B']

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]

# Known universality classes
KNOWN_CLASSES = {
    'Mean-field': {'beta': 0.5, 'gamma': 1.0, 'nu': 0.5},
    '2D Ising': {'beta': 0.125, 'gamma': 1.75, 'nu': 1.0},
    '3D Ising': {'beta': 0.326, 'gamma': 1.237, 'nu': 0.630},
    '3D XY': {'beta': 0.345, 'gamma': 1.316, 'nu': 0.671},
    '3D Heisenberg': {'beta': 0.365, 'gamma': 1.386, 'nu': 0.707},
}


def measure_exponents(model, tok, device, model_name):
    """Measure critical exponents beta, gamma, nu for a single model."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # === Collect layer-wise data ===
    all_P1 = []  # Order parameter candidate (top-1 prob)
    all_T = []   # Temperature
    all_U = []   # Internal energy

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        P1_list, T_list, U_list = [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_list.append(probs.max().item())
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)
        all_P1.append(P1_list)
        all_T.append(T_list)
        all_U.append(U_list)

    n_hs = min(len(p) for p in all_P1)
    mean_P1 = np.array([np.mean([p[i] for p in all_P1]) for i in range(n_hs)])
    mean_T = np.array([np.mean([t[i] for t in all_T]) for i in range(n_hs)])
    mean_U = np.array([np.mean([u[i] for u in all_U]) for i in range(n_hs)])

    # L0 from dT maximum
    dT = np.abs(np.diff(mean_T))
    L0 = int(np.argmax(dT))

    # === beta: order parameter exponent ===
    # P1(l) ~ |l - L0|^beta for l > L0 (ordered phase)
    distances_after = []
    P1_after = []
    for l in range(L0 + 1, n_hs):
        d = l - L0
        if mean_P1[l] > 0:
            distances_after.append(d)
            P1_after.append(mean_P1[l])

    beta = None
    if len(distances_after) > 2:
        try:
            log_d = np.log(distances_after)
            log_P1 = np.log(np.array(P1_after))
            coeffs = np.polyfit(log_d, log_P1, 1)
            beta = float(coeffs[0])
        except Exception:
            pass

    # === gamma: susceptibility exponent ===
    # chi ~ |l - L0|^{-gamma}
    # Use variance of P1 across prompts as susceptibility
    var_P1 = np.array([np.var([p[i] for p in all_P1 if i < len(p)]) for i in range(n_hs)])
    distances_all = []
    chi_values = []
    for l in range(n_hs):
        d = abs(l - L0)
        if d > 0 and d < n_hs // 2 and var_P1[l] > 0:
            distances_all.append(d)
            chi_values.append(var_P1[l])

    gamma = None
    if len(distances_all) > 2:
        try:
            log_d = np.log(distances_all)
            log_chi = np.log(chi_values)
            coeffs = np.polyfit(log_d, log_chi, 1)
            gamma = float(-coeffs[0])
        except Exception:
            pass

    # === nu: correlation length exponent ===
    # xi ~ |l - L0|^{-nu}
    # Correlation length from autocorrelation of hidden state norms
    correlations = []
    for delta in range(1, min(10, n_hs)):
        corr_vals = []
        for l in range(n_hs - delta):
            c = np.corrcoef(
                [p[l] for p in all_P1 if l < len(p) and l+delta < len(p)],
                [p[l+delta] for p in all_P1 if l < len(p) and l+delta < len(p)]
            )[0, 1]
            if not np.isnan(c):
                corr_vals.append(abs(c))
        if corr_vals:
            correlations.append(float(np.mean(corr_vals)))
        else:
            correlations.append(0)

    # Correlation length = distance at which correlation drops to 1/e
    xi = None
    for i, c in enumerate(correlations):
        if c < 1.0 / np.e and i > 0:
            xi = float(i)
            break

    # nu from scaling: xi ~ N^nu (finite-size scaling)
    nu = None
    if xi is not None and n_layers > 0:
        nu = float(np.log(xi + 1) / np.log(n_layers))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'L0': L0,
        'L0_ratio': L0 / n_layers,
        'beta': beta,
        'gamma': gamma,
        'nu': nu,
        'xi': xi,
        'mean_P1': mean_P1.tolist(),
        'mean_T': mean_T.tolist(),
        'mean_U': mean_U.tolist(),
        'var_P1': var_P1.tolist(),
        'correlations': correlations,
    }


def classify_universality(exponents):
    """Find the closest known universality class."""
    measured = {
        'beta': exponents.get('beta'),
        'gamma': exponents.get('gamma'),
        'nu': exponents.get('nu'),
    }
    # Filter out None values
    valid = {k: v for k, v in measured.items() if v is not None}
    if not valid:
        return 'Unknown', float('inf')

    best_class = 'Unknown'
    best_dist = float('inf')
    for cls_name, cls_exp in KNOWN_CLASSES.items():
        dist = 0
        for k in valid:
            dist += (valid[k] - cls_exp[k]) ** 2
        dist = np.sqrt(dist)
        if dist < best_dist:
            best_dist = dist
            best_class = cls_name

    return best_class, best_dist


def main():
    print("=" * 70)
    print("Phase 217: Universality Class Determination")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_results = {}

    for size in MODEL_SIZES:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        exponents = measure_exponents(model, tok, device, size)
        model_results[size] = exponents

        print(f"  L0={exponents['L0']} (ratio={exponents['L0_ratio']:.3f})")
        print(f"  beta={exponents['beta']}")
        print(f"  gamma={exponents['gamma']}")
        print(f"  nu={exponents['nu']}")

        cls_name, cls_dist = classify_universality(exponents)
        exponents['class'] = cls_name
        exponents['class_distance'] = cls_dist
        print(f"  Closest class: {cls_name} (dist={cls_dist:.3f})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Load prior results if available ===
    prior_data = {}
    for fname in ['phase212_cross_scale.json', 'phase213_critical_slowing.json',
                   'phase214_fdt.json', 'phase215_order_parameter.json',
                   'phase216_susceptibility.json']:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            prior_data[fname] = json.load(open(path))

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Order parameter profiles (normalized)
    for size in MODEL_SIZES:
        r = model_results[size]
        x_norm = np.linspace(0, 1, len(r['mean_P1']))
        axes[0, 0].plot(x_norm, r['mean_P1'], '-', color=colors[size], lw=2,
                        label=f'{size}')
        L0_norm = r['L0_ratio']
        axes[0, 0].axvline(x=L0_norm, color=colors[size], ls='--', alpha=0.5)
    axes[0, 0].set_xlabel('Normalized Layer')
    axes[0, 0].set_ylabel('Top-1 Probability')
    axes[0, 0].set_title('(a) Order Parameter P1')
    axes[0, 0].legend(fontsize=8)

    # (b) Susceptibility (variance of P1)
    for size in MODEL_SIZES:
        r = model_results[size]
        x_norm = np.linspace(0, 1, len(r['var_P1']))
        axes[0, 1].plot(x_norm, r['var_P1'], '-', color=colors[size], lw=2,
                        label=f'{size}')
    axes[0, 1].set_xlabel('Normalized Layer')
    axes[0, 1].set_ylabel('Var(P1) ~ chi')
    axes[0, 1].set_title('(b) Susceptibility (P1 variance)')
    axes[0, 1].legend(fontsize=8)

    # (c) Correlations
    for size in MODEL_SIZES:
        r = model_results[size]
        axes[0, 2].plot(range(1, len(r['correlations'])+1), r['correlations'],
                        'o-', color=colors[size], lw=2, label=f'{size}', markersize=5)
    axes[0, 2].axhline(y=1/np.e, color='gray', ls='--', alpha=0.5, label='1/e')
    axes[0, 2].set_xlabel('Layer Distance delta')
    axes[0, 2].set_ylabel('Autocorrelation')
    axes[0, 2].set_title('(c) Correlation Length')
    axes[0, 2].legend(fontsize=8)

    # (d) Critical exponents comparison
    exp_names = ['beta', 'gamma', 'nu']
    x = np.arange(len(exp_names))
    w = 0.35
    for si, size in enumerate(MODEL_SIZES):
        vals = []
        for e in exp_names:
            v = model_results[size].get(e)
            vals.append(v if v is not None else 0)
        offset = (si - 0.5) * w
        axes[1, 0].bar(x + offset, vals, w * 0.9, label=size, color=colors[size], alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(exp_names)
    axes[1, 0].set_ylabel('Exponent Value')
    axes[1, 0].set_title('(d) Critical Exponents')
    axes[1, 0].legend(fontsize=8)

    # (e) Distance to known universality classes
    class_names = list(KNOWN_CLASSES.keys())
    for si, size in enumerate(MODEL_SIZES):
        r = model_results[size]
        measured = {k: r[k] for k in ['beta', 'gamma', 'nu'] if r.get(k) is not None}
        dists = []
        for cls in class_names:
            d = sum((measured.get(k, 0) - KNOWN_CLASSES[cls][k]) ** 2
                    for k in measured) ** 0.5
            dists.append(d)
        offset = (si - 0.5) * w
        axes[1, 1].bar(np.arange(len(class_names)) + offset, dists,
                       w * 0.9, label=size, color=colors[size], alpha=0.7)
    axes[1, 1].set_xticks(np.arange(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=25, ha='right', fontsize=7)
    axes[1, 1].set_ylabel('Euclidean Distance')
    axes[1, 1].set_title('(e) Distance to Known Classes')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Universality Class\n\n"
    for size in MODEL_SIZES:
        r = model_results[size]
        summary += f"{size}:\n"
        summary += f"  beta  = {r['beta']:.3f}\n" if r['beta'] else "  beta  = N/A\n"
        summary += f"  gamma = {r['gamma']:.3f}\n" if r['gamma'] else "  gamma = N/A\n"
        summary += f"  nu    = {r['nu']:.3f}\n" if r['nu'] else "  nu    = N/A\n"
        summary += f"  -> {r['class']} (d={r['class_distance']:.3f})\n\n"

    # Universality check: same class?
    c1 = model_results['0.5B'].get('class', '')
    c2 = model_results['1.5B'].get('class', '')
    summary += f"Universal: {'YES' if c1 == c2 else 'NO'} ({c1} vs {c2})"

    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 217: Universality Class Determination",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase217_universality_class')
    plt.close()

    save_results('phase217_universality_class', {
        'experiment': 'Universality Class',
        'models': model_results,
        'prior_integration': {k: 'loaded' for k in prior_data},
    })


if __name__ == '__main__':
    main()
