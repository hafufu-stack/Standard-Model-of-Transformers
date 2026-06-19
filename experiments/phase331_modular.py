# -*- coding: utf-8 -*-
"""
Phase 331: Modular Invariance -- Lattice Symmetry
=====================================================
In 2D CFT, the partition function Z is invariant under modular
transformations (SL(2,Z)). Test whether the Transformer's partition
function exhibits modular invariance on the layer "torus".
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


def measure_modular(model, tok, prompt, device):
    """Measure modular invariance of the partition function."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float() for li in range(n_layers + 1)]

    # Partition function: Z = sum_i exp(-beta * E_i)
    # At each layer, compute Z for various beta
    betas = np.linspace(0.01, 2.0, 20)
    Z_layers = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        energies = h.abs()
        Zs = []
        for beta in betas:
            Z = float(torch.sum(torch.exp(-beta * energies)).item())
            Zs.append(Z)
        Z_layers.append(Zs)

    Z_arr = np.array(Z_layers)  # (n_layers+1, n_betas)

    # T-duality: Z(beta) should relate to Z(1/beta) under modular S
    # Test: Z(tau) vs Z(-1/tau) where tau = i*beta
    s_invariance_scores = []
    for li in range(n_layers + 1):
        # Compare Z(beta) with Z(1/beta) via correlation
        Z_beta = Z_arr[li, :]
        Z_inv = Z_arr[li, ::-1]  # Approximate Z(1/beta) via reversal
        if np.std(Z_beta) > 1e-10 and np.std(Z_inv) > 1e-10:
            r, _ = stats.pearsonr(np.log(Z_beta + 1), np.log(Z_inv + 1))
            s_invariance_scores.append(float(r))
        else:
            s_invariance_scores.append(0)

    # T-modular: Z(beta) ~ Z(beta + 1) periodicity
    t_scores = []
    for bi in range(len(betas) - 1):
        col1 = Z_arr[:, bi]
        col2 = Z_arr[:, bi + 1]
        if np.std(col1) > 1e-10 and np.std(col2) > 1e-10:
            r, _ = stats.pearsonr(col1, col2)
            t_scores.append(float(r))
        else:
            t_scores.append(1.0)

    # Free energy as function of beta
    F_layers = []
    for li in range(n_layers + 1):
        Fs = []
        for bi, beta in enumerate(betas):
            F = -np.log(Z_arr[li, bi] + 1e-30) / (beta + 1e-10)
            Fs.append(float(F))
        F_layers.append(Fs)

    # Crossing symmetry: F(beta, layer) should have symmetry structure
    F_arr = np.array(F_layers)
    cross_sym = float(np.corrcoef(F_arr.flatten(), F_arr.T.flatten()[:len(F_arr.flatten())])[0, 1])

    return {
        's_invariance': [round(s, 4) for s in s_invariance_scores],
        'mean_s_inv': round(float(np.mean(s_invariance_scores)), 4),
        't_modular': [round(t, 4) for t in t_scores],
        'mean_t_mod': round(float(np.mean(t_scores)), 4),
        'cross_symmetry': round(cross_sym, 4),
        'Z_profile_beta1': [round(float(Z_arr[li, 10]), 4) for li in range(n_layers + 1)],
    }


def main():
    print("=" * 70)
    print("Phase 331: Modular Invariance")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        mod_data = []
        for prompt in PROMPTS:
            m = measure_modular(model, tok, prompt, device)
            mod_data.append(m)

        n = len(mod_data[0]['s_invariance'])
        all_results[size] = {
            'avg_s_invariance': [round(float(np.mean([m['s_invariance'][i] for m in mod_data])), 4) for i in range(n)],
            'mean_s_inv': round(float(np.mean([m['mean_s_inv'] for m in mod_data])), 4),
            'mean_t_mod': round(float(np.mean([m['mean_t_mod'] for m in mod_data])), 4),
            'cross_symmetry': round(float(np.mean([m['cross_symmetry'] for m in mod_data])), 4),
            'Z_profile': [round(float(np.mean([m['Z_profile_beta1'][i] for m in mod_data])), 4)
                         for i in range(n)],
        }
        print(f"  S-invariance: {all_results[size]['mean_s_inv']:.4f}")
        print(f"  T-modular: {all_results[size]['mean_t_mod']:.4f}")
        print(f"  Cross sym: {all_results[size]['cross_symmetry']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_s_invariance'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('S-invariance')
    axes[0, 0].set_title('(a) S-Modular Invariance', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['Z_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Z(beta=1)')
    axes[0, 1].set_title('(b) Partition Function', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 2].bar(x - w, [all_results[s]['mean_s_inv'] for s in sizes], w,
                  label='S-inv', color='#3498db')
    axes[0, 2].bar(x, [all_results[s]['mean_t_mod'] for s in sizes], w,
                  label='T-mod', color='#e74c3c')
    axes[0, 2].bar(x + w, [all_results[s]['cross_symmetry'] for s in sizes], w,
                  label='Cross', color='#2ecc71')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Symmetry Scores', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')

    txt = "MODULAR INVARIANCE\n\n"
    txt += "S: Z(tau) <-> Z(-1/tau)\n"
    txt += "T: Z(tau) <-> Z(tau+1)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  S-inv = {d['mean_s_inv']:.3f}\n"
        txt += f"  T-mod = {d['mean_t_mod']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 331: Modular Invariance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase331_modular')
    plt.close()
    save_results('phase331_modular', {'experiment': 'Modular Invariance', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
