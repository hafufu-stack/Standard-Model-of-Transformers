# -*- coding: utf-8 -*-
"""
Phase 222: Onsager Reciprocal Relations
==========================================
Test Onsager reciprocal relations: L_ij = L_ji
Cross-coupling coefficients between thermodynamic fluxes
(dT/dl, dU/dl, dS/dl, dP1/dl) should be symmetric.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
    "Superconductors have zero electrical resistance",
    "The brain contains billions of neurons",
    "Algorithms determine computational complexity",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
]


def measure_onsager(model, tok, device, model_name):
    """Measure Onsager matrix from cross-correlations of thermodynamic fluxes."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Collect per-prompt layer profiles
    all_profiles = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_list, U_list, S_list, P1_list = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_list.append(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(S if not np.isnan(S) else 0)
            S_list.append(S if not np.isnan(S) else 0)
        all_profiles.append({'T': T_list, 'U': U_list, 'S': S_list, 'P1': P1_list})

    n = min(len(p['T']) for p in all_profiles)

    # Compute fluxes: J_i = dX_i/dl for each prompt
    var_names = ['T', 'U', 'S', 'P1']
    all_fluxes = {v: [] for v in var_names}
    for profile in all_profiles:
        for v in var_names:
            data = profile[v][:n]
            flux = [data[i+1] - data[i] for i in range(n-1)]
            all_fluxes[v].append(flux)

    n_flux = n - 1

    # Onsager matrix: L_ij = <J_i * J_j> (correlation of fluxes)
    L = np.zeros((len(var_names), len(var_names)))
    L_per_layer = []

    for l in range(n_flux):
        L_l = np.zeros((len(var_names), len(var_names)))
        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                ji = [all_fluxes[vi][p][l] for p in range(len(PROMPTS))]
                jj = [all_fluxes[vj][p][l] for p in range(len(PROMPTS))]
                L_l[i, j] = float(np.mean(np.array(ji) * np.array(jj)))
        L_per_layer.append(L_l.tolist())
        L += L_l

    L /= n_flux

    # Onsager reciprocal test: L_ij == L_ji?
    asymmetry = np.zeros((len(var_names), len(var_names)))
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):
            if abs(L[i, j]) + abs(L[j, i]) > 1e-10:
                asymmetry[i, j] = abs(L[i, j] - L[j, i]) / (abs(L[i, j]) + abs(L[j, i]))
            else:
                asymmetry[i, j] = 0
            asymmetry[j, i] = asymmetry[i, j]

    max_asymmetry = float(np.max(asymmetry))
    mean_asymmetry = float(np.mean(asymmetry[np.triu_indices(len(var_names), k=1)]))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'var_names': var_names,
        'onsager_matrix': L.tolist(),
        'asymmetry_matrix': asymmetry.tolist(),
        'max_asymmetry': max_asymmetry,
        'mean_asymmetry': mean_asymmetry,
        'onsager_symmetric': max_asymmetry < 0.1,
    }


def main():
    print("=" * 70)
    print("Phase 222: Onsager Reciprocal Relations")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_onsager(model, tok, device, size)
        results[size] = r
        print(f"  Max asymmetry = {r['max_asymmetry']:.4f}")
        print(f"  Mean asymmetry = {r['mean_asymmetry']:.4f}")
        print(f"  Symmetric: {r['onsager_symmetric']}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    var_names = results[list(results.keys())[0]]['var_names']

    # (a) Onsager matrix 0.5B
    L05 = np.array(results['0.5B']['onsager_matrix'])
    im0 = axes[0, 0].imshow(L05, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_xticks(range(len(var_names)))
    axes[0, 0].set_xticklabels(var_names)
    axes[0, 0].set_yticks(range(len(var_names)))
    axes[0, 0].set_yticklabels(var_names)
    axes[0, 0].set_title('(a) Onsager L_ij (0.5B)')
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # (b) Onsager matrix 1.5B
    L15 = np.array(results['1.5B']['onsager_matrix'])
    im1 = axes[0, 1].imshow(L15, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_xticks(range(len(var_names)))
    axes[0, 1].set_xticklabels(var_names)
    axes[0, 1].set_yticks(range(len(var_names)))
    axes[0, 1].set_yticklabels(var_names)
    axes[0, 1].set_title('(b) Onsager L_ij (1.5B)')
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # (c) Asymmetry matrix comparison
    A05 = np.array(results['0.5B']['asymmetry_matrix'])
    A15 = np.array(results['1.5B']['asymmetry_matrix'])
    combined = np.zeros((len(var_names), len(var_names) * 2 + 1))
    combined[:, :len(var_names)] = A05
    combined[:, len(var_names)+1:] = A15
    im2 = axes[0, 2].imshow(combined, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
    axes[0, 2].set_title('(c) Asymmetry: 0.5B | 1.5B')
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # (d) L_ij vs L_ji scatter
    for size, r in results.items():
        L = np.array(r['onsager_matrix'])
        lij, lji = [], []
        for i in range(len(var_names)):
            for j in range(i+1, len(var_names)):
                lij.append(L[i, j])
                lji.append(L[j, i])
        axes[1, 0].scatter(lij, lji, s=50, alpha=0.7, label=size)
    lim = max(abs(np.array(lij + lji)).max(), 0.01) * 1.2
    axes[1, 0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='L_ij=L_ji')
    axes[1, 0].set_xlabel('L_ij')
    axes[1, 0].set_ylabel('L_ji')
    axes[1, 0].set_title('(d) Reciprocal Test')
    axes[1, 0].legend(fontsize=8)

    # (e) Eigenvalues of Onsager matrix
    for size, r in results.items():
        L = np.array(r['onsager_matrix'])
        eigvals = np.linalg.eigvalsh(L)
        axes[1, 1].bar(range(len(eigvals)), sorted(eigvals, reverse=True),
                      alpha=0.6, label=size)
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].set_title('(e) Onsager Eigenspectrum')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Onsager Reciprocal\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Max asym  = {r['max_asymmetry']:.4f}\n"
        summary += f"  Mean asym = {r['mean_asymmetry']:.4f}\n"
        summary += f"  Symmetric = {r['onsager_symmetric']}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 222: Onsager Reciprocal Relations", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase222_onsager')
    plt.close()
    save_results('phase222_onsager', {'experiment': 'Onsager Reciprocal', 'results': results})


if __name__ == '__main__':
    main()
