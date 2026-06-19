# -*- coding: utf-8 -*-
"""
Phase 322: Regge Trajectory -- Angular Momentum vs Mass
=========================================================
In hadron physics, Regge trajectories show J ~ alpha' * M^2.
Angular momentum grows quadratically with mass.
Test if effective angular momentum (rotation in hidden space)
scales with energy (mass-squared).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_regge(model, tok, prompt, device):
    """Measure Regge trajectory: J vs M^2."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # For each layer, compute "mass" and "angular momentum"
    masses_sq = []
    angular_momenta = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()

        # Mass^2 = total energy = |h|^2
        M2 = float((h ** 2).sum().item())
        masses_sq.append(M2)

        # Angular momentum = rotation in hidden space
        # J = |x cross p| where p ~ dh/dl
        if li < n_layers:
            h_next = out.hidden_states[li + 1][0, -1, :].float()
            dh = h_next - h  # "momentum"
            # Cross product magnitude in 2D projection
            # Use first 2 principal components
            from sklearn.decomposition import PCA
            pair = torch.stack([h, dh]).cpu().numpy()
            if pair.shape[0] >= 2:
                pca = PCA(n_components=2)
                proj = pca.fit_transform(pair)
                J = abs(proj[0, 0] * proj[1, 1] - proj[0, 1] * proj[1, 0])
            else:
                J = 0
        else:
            J = angular_momenta[-1] if angular_momenta else 0

        angular_momenta.append(float(J))

    # Regge trajectory: J = alpha' * M^2 + alpha_0
    M2_arr = np.array(masses_sq)
    J_arr = np.array(angular_momenta)

    slope, intercept, r, _, _ = stats.linregress(M2_arr, J_arr)
    regge_r2 = r**2
    alpha_prime = slope  # Regge slope

    return {
        'masses_sq': [round(m, 2) for m in masses_sq],
        'angular_momenta': [round(j, 4) for j in angular_momenta],
        'alpha_prime': round(float(alpha_prime), 8),
        'regge_r2': round(float(regge_r2), 4),
        'regge_intercept': round(float(intercept), 4),
    }


def main():
    print("=" * 70)
    print("Phase 322: Regge Trajectory")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        reg_data = []
        for prompt in PROMPTS:
            r = measure_regge(model, tok, prompt, device)
            reg_data.append(r)

        all_results[size] = {
            'alpha_prime': round(float(np.mean([r['alpha_prime'] for r in reg_data])), 8),
            'regge_r2': round(float(np.mean([r['regge_r2'] for r in reg_data])), 4),
            'regge_intercept': round(float(np.mean([r['regge_intercept'] for r in reg_data])), 4),
        }
        print(f"  alpha': {all_results[size]['alpha_prime']:.8f}")
        print(f"  Regge R2: {all_results[size]['regge_r2']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    sizes = list(all_results.keys())

    axes[0].bar(sizes, [all_results[s]['regge_r2'] for s in sizes],
               color=[colors[s] for s in sizes])
    axes[0].set_ylabel('R2'); axes[0].set_title('(a) Regge R2: J ~ M^2', fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].bar(sizes, [all_results[s]['alpha_prime'] * 1e6 for s in sizes],
               color=[colors[s] for s in sizes])
    axes[1].set_ylabel("alpha' x 10^6"); axes[1].set_title("(b) Regge Slope alpha'", fontweight='bold')
    axes[1].grid(alpha=0.3)

    txt = "REGGE TRAJECTORY\n\nJ = alpha' * M^2 + J_0\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n  R2={d['regge_r2']:.3f}\n\n"
    axes[2].text(0.5, 0.5, txt, ha='center', va='center',
                transform=axes[2].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                family='monospace')
    axes[2].axis('off'); axes[2].set_title('(c) Summary')

    fig.suptitle("Phase 322: Regge Trajectory -- J vs M^2", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase322_regge')
    plt.close()
    save_results('phase322_regge', {'experiment': 'Regge Trajectory', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
