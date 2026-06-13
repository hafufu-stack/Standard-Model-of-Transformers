# -*- coding: utf-8 -*-
"""
Phase 110: Input-Dependent Phase Transition
Does the eta transition point L0 shift for different input types?
Test: natural text, code, math, creative writing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

INPUT_CATEGORIES = {
    'natural': [
        "The weather today is expected to be sunny with temperatures reaching",
        "After a long day at work she decided to take a walk through the park",
        "The cat sat quietly on the windowsill watching the birds outside",
        "He opened the door to find an unexpected package on the doorstep",
    ],
    'scientific': [
        "The second law of thermodynamics states that entropy in an isolated system",
        "Protein folding is determined by the amino acid sequence through molecular",
        "The cosmic microwave background radiation provides evidence for the big bang",
        "Quantum entanglement allows particles to share states across arbitrary distances",
    ],
    'code': [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1)",
        "import numpy as np\ndata = np.random.randn(100)\nmean = np.mean(data)",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value",
        "for i in range(len(matrix)):\n    for j in range(len(matrix[0])):\n        result",
    ],
    'math': [
        "The integral of x squared from zero to one equals one third because",
        "If the matrix A has eigenvalues lambda one and lambda two then the determinant",
        "The Taylor series expansion of the exponential function around zero gives",
        "By the fundamental theorem of algebra every polynomial of degree n has exactly",
    ],
}


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def measure_eta_profile(model, tok, device, prompts):
    """Measure eta at each effective depth for given prompts."""
    n_layers = len(model.model.layers) + 1
    eta_by_L = []

    for L in range(4, n_layers):
        etas = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            T_vals = []
            for li in range(min(L + 1, len(out.hidden_states))):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                if not np.isnan(T):
                    T_vals.append(T)
            if len(T_vals) >= 4:
                T_hot = max(T_vals)
                T_cold = min(T_vals[len(T_vals)//2:])
                if T_hot > 0.01:
                    etas.append(1.0 - T_cold / T_hot)
        eta_by_L.append({
            'L': L,
            'eta': float(np.mean(etas)) if etas else 0.0,
            'std': float(np.std(etas)) if etas else 0.0,
        })

    return eta_by_L


def main():
    print("=" * 70)
    print("Phase 110: Input-Dependent Phase Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    all_data = {}

    for cat_name, prompts in INPUT_CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        profile = measure_eta_profile(model, tok, device, prompts)

        Ls = np.array([r['L'] for r in profile])
        etas = np.array([r['eta'] for r in profile])

        # Fit sigmoid
        try:
            popt, _ = curve_fit(sigmoid, Ls, etas,
                                p0=[20, 0.5, np.min(etas), np.max(etas)],
                                maxfev=10000)
            L0_fit = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            ss_res = np.sum((etas - sig_pred)**2)
            ss_tot = np.sum((etas - np.mean(etas))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        except Exception:
            L0_fit = 20.0
            r2 = 0.0

        print(f"  L0 = {L0_fit:.1f}, R2 = {r2:.4f}")

        all_data[cat_name] = {
            'profile': profile,
            'L0': float(L0_fit),
            'r2': float(r2),
        }

    # === Analysis ===
    L0s = [all_data[c]['L0'] for c in INPUT_CATEGORIES]
    mean_L0 = np.mean(L0s)
    std_L0 = np.std(L0s)
    cv_L0 = std_L0 / (mean_L0 + 1e-10)
    is_input_independent = cv_L0 < 0.1

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'natural': '#27ae60', 'scientific': '#c0392b',
              'code': '#2980b9', 'math': '#8e44ad'}

    # (a) All eta profiles
    for cat_name, data in all_data.items():
        Ls = [r['L'] for r in data['profile']]
        etas = [r['eta'] for r in data['profile']]
        axes[0].plot(Ls, etas, 'o-', color=colors[cat_name], markersize=3,
                    linewidth=1.5, label=f'{cat_name} ($L_0={data["L0"]:.1f}$)')
    L_sm = np.linspace(4, 28, 200)
    axes[0].plot(L_sm, 1-1/np.sqrt(L_sm), 'k--', alpha=0.3, label='Theory')
    axes[0].set_xlabel('Effective Layer Count $L$')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) Eta Profiles by Input Type')
    axes[0].legend(fontsize=7)

    # (b) L0 comparison
    cat_names = list(INPUT_CATEGORIES.keys())
    L0_vals = [all_data[c]['L0'] for c in cat_names]
    bar_colors = [colors[c] for c in cat_names]
    axes[1].bar(range(len(cat_names)), L0_vals, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(cat_names)))
    axes[1].set_xticklabels(cat_names, fontsize=9)
    axes[1].set_ylabel('$L_0$ (transition point)')
    axes[1].axhline(y=mean_L0, color='black', linestyle='--',
                    label=f'Mean={mean_L0:.1f}, CV={cv_L0:.3f}')
    axes[1].set_title(f'(b) Transition Points ({"STABLE" if is_input_independent else "SHIFTS"})')
    axes[1].legend(fontsize=8)

    # (c) Summary
    summary = (
        f"Input Dependence Analysis\n\n"
        + "\n".join(f"{c}: L0={all_data[c]['L0']:.1f} (R2={all_data[c]['r2']:.3f})"
                    for c in cat_names)
        + f"\n\nMean L0: {mean_L0:.1f} +/- {std_L0:.1f}\n"
        f"CV: {cv_L0:.3f}\n\n"
        f"{'INPUT-INDEPENDENT' if is_input_independent else 'INPUT-DEPENDENT'}"
    )
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle(f'Phase 110: Input-Dependent L0 '
                 f'(mean={mean_L0:.1f}, {"STABLE" if is_input_independent else "SHIFTS"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase110_input_dependence')
    plt.close()

    print(f"\n{'='*70}")
    for c in cat_names:
        print(f"  {c}: L0={all_data[c]['L0']:.1f}, R2={all_data[c]['r2']:.3f}")
    print(f"  Mean L0: {mean_L0:.1f} +/- {std_L0:.1f}, CV={cv_L0:.3f}")
    print(f"  {'INPUT-INDEPENDENT' if is_input_independent else 'INPUT-DEPENDENT'}")
    print(f"{'='*70}")

    save_results('phase110_input_dependence', {
        'experiment': 'Input-Dependent Phase Transition',
        'categories': {c: {'L0': all_data[c]['L0'], 'r2': all_data[c]['r2']}
                       for c in cat_names},
        'summary': {
            'mean_L0': float(mean_L0),
            'std_L0': float(std_L0),
            'cv': float(cv_L0),
            'is_input_independent': is_input_independent,
        }
    })


if __name__ == '__main__':
    main()
