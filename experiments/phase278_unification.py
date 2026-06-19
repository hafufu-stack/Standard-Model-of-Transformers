# -*- coding: utf-8 -*-
"""
Phase 278: Grand Unification Tensor
======================================
Compile all 6 Universal Laws into a single covariance matrix.
Test independence and correlations between laws across prompts.

Construct Omega_ij = corr(Law_i, Law_j) and perform PCA
to find the fundamental dimensions of the Standard Model.
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

# 20 prompts for cross-law correlation
PROMPTS = [
    "The speed of light is constant in all reference frames",
    "Water is composed of hydrogen and oxygen atoms",
    "Neural networks learn through gradient descent optimization",
    "The French Revolution began in 1789 with the storming of the Bastille",
    "Photosynthesis converts carbon dioxide and water into glucose",
    "The Pythagorean theorem relates the sides of a right triangle",
    "Antibiotics target bacterial cell processes without harming host cells",
    "The stock market fluctuates based on supply and demand dynamics",
    "Quantum computers use qubits that can exist in superposition states",
    "The Renaissance was a cultural movement that began in Italy",
    "Protein folding determines the three-dimensional structure and function",
    "Climate models predict future temperature changes based on emissions",
    "The human genome contains approximately three billion base pairs",
    "Cryptographic hash functions map arbitrary data to fixed-size values",
    "Volcanic eruptions release gases and particulate matter into the atmosphere",
    "Machine learning models can overfit when trained on limited data",
    "The immune system recognizes foreign antigens through antibody binding",
    "General relativity describes gravity as curvature of spacetime geometry",
    "Ocean acidification threatens marine ecosystems through pH reduction",
    "Transformer architectures process sequences through self-attention mechanisms",
]


def measure_all_laws(model, tok, prompt, device):
    """Measure all 6 law quantities for a single prompt."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states)

    # Collect per-layer data
    temps = []
    p1s = []
    us = []
    prs = []
    energies = []

    for li, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        U = h.norm().item()
        us.append(U)

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()
        prs.append(PR)

        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(t_val): t_val = 0

        temps.append(t_val)
        p1s.append(p1)

        # Energy distribution (for Boltzmann check)
        energies.append(U)

    temps = np.array(temps[1:])  # skip layer 0
    p1s = np.array(p1s[1:])
    us = np.array(us[1:])
    prs = np.array(prs[1:])

    # === Law 1: Boltzmann Distribution ===
    # Check if energy distribution follows exp(-E/kT)
    # Use R^2 of log-linear fit
    if len(us) > 3:
        u_sorted = np.sort(us)
        log_rank = np.log(np.arange(1, len(u_sorted) + 1))
        _, _, r_boltz, _, _ = stats.linregress(u_sorted, log_rank)
        law1_R2 = r_boltz**2
    else:
        law1_R2 = 0

    # === Law 2: Negative Specific Heat ===
    # dT/dU should be negative
    if len(us) > 2:
        slope_cv, _, _, p_cv, _ = stats.linregress(us, temps)
        law2_Cv = slope_cv  # negative = confirmed
    else:
        law2_Cv = 0
        p_cv = 1

    # === Law 3: Inverse Radiation ===
    # L ~ T^n where n < 0
    final_logits = out.logits[0, -1, :].float()
    loss = -torch.log(torch.softmax(final_logits, dim=-1).max()).item()
    law3_loss = loss
    law3_T = temps[-1]

    # === Law 4: Carnot Efficiency ===
    T_hot = max(temps)
    T_cold = min(temps)
    eta_carnot = 1 - T_cold / (T_hot + 1e-10)
    law4_eta = eta_carnot

    # === Law 5: Information Concentration ===
    F_initial = us[0] - temps[0] * np.log(prs[0] + 1)
    F_final = us[-1] - temps[-1] * np.log(prs[-1] + 1)
    law5_F_ratio = F_final / (F_initial + 1e-10)

    # === Law 6: P1*T Conservation ===
    p1t = p1s * temps
    law6_P1T_mean = float(np.mean(p1t))
    law6_P1T_cv = float(np.std(p1t) / (np.mean(p1t) + 1e-10))

    return {
        'law1_R2': round(law1_R2, 4),
        'law2_Cv': round(float(law2_Cv), 4),
        'law3_loss': round(law3_loss, 4),
        'law3_T': round(float(law3_T), 4),
        'law4_eta': round(float(law4_eta), 4),
        'law5_F_ratio': round(float(law5_F_ratio), 4),
        'law6_P1T': round(law6_P1T_mean, 4),
        'law6_cv': round(law6_P1T_cv, 4),
    }


def main():
    print("=" * 70)
    print("Phase 278: Grand Unification Tensor")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        measurements = []
        for pi, prompt in enumerate(PROMPTS):
            m = measure_all_laws(model, tok, prompt, device)
            measurements.append(m)
            if (pi + 1) % 5 == 0:
                print(f"  Processed {pi+1}/{len(PROMPTS)}")

        # Build the unification matrix
        law_keys = ['law1_R2', 'law2_Cv', 'law3_loss', 'law4_eta',
                     'law5_F_ratio', 'law6_P1T']
        law_names = ['Boltzmann', 'Neg. Cv', 'Inv. Radiation',
                      'Carnot', 'Info Conc.', 'P1*T']

        # Data matrix: [n_prompts x n_laws]
        data_matrix = np.array([[m[k] for k in law_keys] for m in measurements])

        # Correlation matrix
        corr_matrix = np.corrcoef(data_matrix.T)
        # Handle NaN in correlation
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)

        # PCA
        centered = data_matrix - data_matrix.mean(axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by eigenvalue (largest first)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            explained_variance = eigenvalues / (eigenvalues.sum() + 1e-10)
            pca_success = True
        except Exception:
            eigenvalues = np.zeros(len(law_keys))
            explained_variance = np.zeros(len(law_keys))
            eigenvectors = np.eye(len(law_keys))
            pca_success = False

        # Independence test: how many PCA components explain 95%?
        cumvar = np.cumsum(explained_variance)
        n_independent = int(np.searchsorted(cumvar, 0.95) + 1)

        all_results[size] = {
            'measurements': measurements,
            'correlation_matrix': corr_matrix.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'explained_variance': explained_variance.tolist(),
            'n_independent_dims': n_independent,
            'law_means': {k: round(float(data_matrix[:, i].mean()), 4)
                         for i, k in enumerate(law_keys)},
            'law_stds': {k: round(float(data_matrix[:, i].std()), 4)
                        for i, k in enumerate(law_keys)},
        }

        print(f"  Independent dimensions: {n_independent}/{len(law_keys)}")
        print(f"  Explained variance: {explained_variance[:3]}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    law_names_short = ['Boltz', 'Cv', 'Rad', 'Carnot', 'Info', 'P1T']

    # (a) Correlation matrix (first model)
    first_size = list(all_results.keys())[0]
    cm = np.array(all_results[first_size]['correlation_matrix'])
    im = axes[0, 0].imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_xticks(range(len(law_names_short)))
    axes[0, 0].set_xticklabels(law_names_short, rotation=45, fontsize=8)
    axes[0, 0].set_yticks(range(len(law_names_short)))
    axes[0, 0].set_yticklabels(law_names_short, fontsize=8)
    for i in range(len(law_names_short)):
        for j in range(len(law_names_short)):
            axes[0, 0].text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title(f'(a) Omega_ij ({first_size})', fontweight='bold')

    # (b) Explained variance
    colors_bar = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    for size, data in all_results.items():
        ev = data['explained_variance']
        axes[0, 1].bar(np.arange(len(ev)) + (0.15 if size == '1.5B' else -0.15),
                      ev, width=0.3, color=colors_bar[size], label=size)
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_title('(b) PCA Spectrum', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Cumulative variance
    for size, data in all_results.items():
        cumvar = np.cumsum(data['explained_variance'])
        axes[0, 2].plot(range(1, len(cumvar)+1), cumvar, 'o-',
                       color=colors_bar[size], lw=2, label=size)
    axes[0, 2].axhline(0.95, color='red', ls='--', label='95%')
    axes[0, 2].set_xlabel('Number of Components')
    axes[0, 2].set_ylabel('Cumulative Variance')
    axes[0, 2].set_title('(c) Cumulative Explained Variance', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Correlation matrix (second model)
    if '1.5B' in all_results:
        cm2 = np.array(all_results['1.5B']['correlation_matrix'])
        im2 = axes[1, 0].imshow(cm2, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(law_names_short)))
        axes[1, 0].set_xticklabels(law_names_short, rotation=45, fontsize=8)
        axes[1, 0].set_yticks(range(len(law_names_short)))
        axes[1, 0].set_yticklabels(law_names_short, fontsize=8)
        for i in range(len(law_names_short)):
            for j in range(len(law_names_short)):
                axes[1, 0].text(j, i, f'{cm2[i,j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im2, ax=axes[1, 0])
        axes[1, 0].set_title('(d) Omega_ij (1.5B)', fontweight='bold')

    # (e) Law means comparison
    law_keys_plot = ['law1_R2', 'law2_Cv', 'law3_loss', 'law4_eta', 'law5_F_ratio', 'law6_P1T']
    x_pos = np.arange(len(law_keys_plot))
    width = 0.35
    for i, (size, data) in enumerate(all_results.items()):
        means = [data['law_means'][k] for k in law_keys_plot]
        stds = [data['law_stds'][k] for k in law_keys_plot]
        # Normalize for visualization
        max_abs = max(abs(m) for m in means) + 1e-10
        norm_means = [m / max_abs for m in means]
        axes[1, 1].bar(x_pos + i*width - width/2, norm_means, width,
                      color=colors_bar[size], label=size)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(law_names_short, fontsize=8)
    axes[1, 1].set_ylabel('Normalized Mean')
    axes[1, 1].set_title('(e) Law Means Comparison', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "GRAND UNIFICATION TENSOR\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  Independent dims: {data['n_independent_dims']}/6\n"
        summary += f"  Top 3 eigenvalues:\n"
        for i, ev in enumerate(data['eigenvalues'][:3]):
            summary += f"    PC{i+1}: {data['explained_variance'][i]:.3f}\n"
        summary += "\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 278: Grand Unification Tensor -- The Standard Model Complete",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase278_unification')
    plt.close()

    save_results('phase278_unification', {
        'experiment': 'Grand Unification Tensor',
        'n_prompts': len(PROMPTS),
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'measurements'}
                   for k, v in all_results.items()},
    })


if __name__ == '__main__':
    main()
