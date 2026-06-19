# -*- coding: utf-8 -*-
"""
Phase 351: String Spectrum -- Regge Trajectory
=====================================================
In string theory, particles lie on Regge trajectories: J = alpha' * M^2 + a_0.
The angular momentum J is linearly related to mass squared.
Test whether the Transformer's excitation spectrum follows Regge behavior.
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
    """Measure Regge trajectories in hidden state spectrum."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # For each layer, compute "mass" (energy) and "spin" (angular momentum)
    masses = []
    spins = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Mass ~ norm (energy)
        M = float(torch.norm(h).item())
        masses.append(M)

        # Angular momentum ~ cross product structure
        # J = sum |h_i * h_{i+1} - h_{i+1} * h_i| (rotation in adjacent dims)
        h_np = h.numpy()
        J = 0.0
        for d in range(0, min(dim - 1, 256), 2):
            J += abs(h_np[d] * h_np[d + 1])
        spins.append(J)

    # Regge trajectory: J = alpha' * M^2 + a_0
    M2 = np.array(masses) ** 2
    J = np.array(spins)
    slope, intercept, r, p, se = stats.linregress(M2, J)
    r2 = r ** 2

    # String tension: sigma = 1/(2*pi*alpha')
    alpha_prime = slope
    sigma = 1.0 / (2 * np.pi * abs(alpha_prime) + 1e-30)

    # Hagedorn temperature: T_H = 1/(2*pi*sqrt(alpha'))
    if alpha_prime > 0:
        T_H = 1.0 / (2 * np.pi * np.sqrt(alpha_prime))
    else:
        T_H = float('inf')

    # Excited state spectrum: level spacing
    level_spacing = []
    sorted_masses = sorted(masses)
    for i in range(1, len(sorted_masses)):
        spacing = sorted_masses[i] - sorted_masses[i - 1]
        level_spacing.append(float(spacing))

    avg_spacing = float(np.mean(level_spacing)) if level_spacing else 0

    return {
        'alpha_prime': round(float(alpha_prime), 6),
        'intercept': round(float(intercept), 4),
        'r2': round(float(r2), 4),
        'sigma': round(float(sigma), 4),
        'T_H': round(float(min(T_H, 1000)), 4),
        'masses': [round(m, 4) for m in masses],
        'spins': [round(j, 4) for j in spins],
        'avg_spacing': round(avg_spacing, 4),
        'regge_holds': r2 > 0.5,
    }


def main():
    print("=" * 70)
    print("Phase 351: Regge Trajectory")
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

        n = len(reg_data[0]['masses'])
        all_results[size] = {
            'alpha_prime': round(float(np.mean([r['alpha_prime'] for r in reg_data])), 6),
            'r2': round(float(np.mean([r['r2'] for r in reg_data])), 4),
            'sigma': round(float(np.mean([r['sigma'] for r in reg_data])), 4),
            'T_H': round(float(np.mean([r['T_H'] for r in reg_data])), 4),
            'avg_spacing': round(float(np.mean([r['avg_spacing'] for r in reg_data])), 4),
            'masses': [round(float(np.mean([r['masses'][i] for r in reg_data])), 4) for i in range(n)],
            'spins': [round(float(np.mean([r['spins'][i] for r in reg_data])), 4) for i in range(n)],
            'regge_holds': sum(1 for r in reg_data if r['regge_holds']) >= 4,
        }
        holds = 'YES' if all_results[size]['regge_holds'] else 'NO'
        print(f"  alpha': {all_results[size]['alpha_prime']:.6f}")
        print(f"  R2: {all_results[size]['r2']:.4f}")
        print(f"  sigma: {all_results[size]['sigma']:.4f}")
        print(f"  Regge holds: {holds}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        M2 = np.array(data['masses'])**2
        axes[0, 0].scatter(M2, data['spins'], color=colors[size], s=30, alpha=0.7, label=size)
        # Fit line
        sl, ic = np.polyfit(M2, data['spins'], 1)
        axes[0, 0].plot(M2, sl*M2 + ic, '--', color=colors[size], alpha=0.5)
    axes[0, 0].set_xlabel('M^2'); axes[0, 0].set_ylabel('J')
    axes[0, 0].set_title('(a) Regge Trajectory', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['masses'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Mass (norm)')
    axes[0, 1].set_title('(b) Mass Spectrum', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['spins'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Spin (J)')
    axes[0, 2].set_title('(c) Spin Spectrum', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['r2'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_title('(d) Regge R2', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "REGGE TRAJECTORY\n\n"
    txt += "J = alpha' * M^2 + a_0\n\n"
    for s in sizes:
        d = all_results[s]
        h = 'YES' if d['regge_holds'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  alpha' = {d['alpha_prime']:.5f}\n"
        txt += f"  R2 = {d['r2']:.3f}\n"
        txt += f"  sigma = {d['sigma']:.2f}\n"
        txt += f"  Regge: {h}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 351: Regge Trajectory", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase351_regge')
    plt.close()
    save_results('phase351_regge', {'experiment': 'Regge Trajectory', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
