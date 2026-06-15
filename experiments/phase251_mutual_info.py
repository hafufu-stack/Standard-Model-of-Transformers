# -*- coding: utf-8 -*-
"""
Phase 251: Mutual Information Between Layers
===============================================
Measure mutual information I(l_i; l_j) between layer pairs.
This reveals the information flow structure: which layers are
informationally coupled and which are independent.
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
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Random words create unpredictable sequences when",
]


def mutual_info_matrix(model, tok, device, model_name):
    """Compute MI proxy between all layer pairs."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_hs = len(model.model.layers) + 1  # +1 for embedding layer

    # Collect top-k tokens at each layer for each prompt
    K = 50  # top-K tokens for MI estimation
    all_topk = []  # [prompt][layer] = set of top-K token IDs

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        topk_layers = []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            topk_ids = torch.topk(logits, K).indices.cpu().numpy()
            topk_layers.append(set(topk_ids.tolist()))
        all_topk.append(topk_layers)

    actual_n = min(len(tl) for tl in all_topk)
    
    # MI proxy: Jaccard similarity of top-K sets across prompts
    # Higher overlap = higher MI
    mi_matrix = np.zeros((actual_n, actual_n))
    for i in range(actual_n):
        for j in range(actual_n):
            jaccard_sum = 0
            for p in range(len(PROMPTS)):
                if i < len(all_topk[p]) and j < len(all_topk[p]):
                    si = all_topk[p][i]
                    sj = all_topk[p][j]
                    inter = len(si & sj)
                    union = len(si | sj)
                    jaccard_sum += inter / (union + 1e-10)
            mi_matrix[i, j] = jaccard_sum / len(PROMPTS)

    # Extract structure
    # Off-diagonal decay rate
    offdiag = []
    for d in range(1, actual_n):
        vals = [mi_matrix[i, i+d] for i in range(actual_n - d)]
        offdiag.append(float(np.mean(vals)))

    # Correlation length: first d where MI < 0.5 * MI(0)
    mi_diag_mean = float(np.mean(np.diag(mi_matrix)))
    corr_length = actual_n
    for d, v in enumerate(offdiag):
        if v < 0.5 * offdiag[0]:
            corr_length = d + 1
            break

    # Block structure: eigenvalue spectrum
    eigenvals = np.linalg.eigvalsh(mi_matrix)
    eigenvals = np.sort(eigenvals)[::-1]

    return {
        'model': model_name,
        'n_layers': actual_n,
        'mi_matrix': mi_matrix.tolist(),
        'offdiag_decay': offdiag,
        'corr_length': corr_length,
        'eigenvalues': eigenvals[:10].tolist(),
        'spectral_gap': float(eigenvals[0] - eigenvals[1]) if len(eigenvals) > 1 else 0,
    }


def main():
    print("=" * 70)
    print("Phase 251: Mutual Information Between Layers")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = mutual_info_matrix(model, tok, device, size)
        results[size] = r
        print(f"  MI matrix: {r['n_layers']}x{r['n_layers']}")
        print(f"  Correlation length: {r['corr_length']}")
        print(f"  Spectral gap: {r['spectral_gap']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a,b) MI heatmaps
    for si, (size, r) in enumerate(results.items()):
        mi = np.array(r['mi_matrix'])
        im = axes[0, si].imshow(mi, cmap='viridis', origin='lower', aspect='auto')
        axes[0, si].set_xlabel('Layer j')
        axes[0, si].set_ylabel('Layer i')
        axes[0, si].set_title(f'({chr(97+si)}) MI Matrix ({size})')
        fig.colorbar(im, ax=axes[0, si], shrink=0.7)

    # (c) Off-diagonal decay
    for size, r in results.items():
        axes[0, 2].plot(range(1, len(r['offdiag_decay'])+1), r['offdiag_decay'],
                       '-o', color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 2].set_xlabel('Layer Distance')
    axes[0, 2].set_ylabel('Mean MI')
    axes[0, 2].set_title('(c) MI Decay with Distance')
    axes[0, 2].legend(fontsize=8)

    # (d) Log MI decay
    for size, r in results.items():
        vals = r['offdiag_decay']
        if len(vals) > 2:
            axes[1, 0].semilogy(range(1, len(vals)+1), vals,
                               '-o', color=colors[size], lw=2, markersize=3, label=size)
    axes[1, 0].set_xlabel('Layer Distance')
    axes[1, 0].set_ylabel('MI (log scale)')
    axes[1, 0].set_title('(d) MI Decay (Log Scale)')
    axes[1, 0].legend(fontsize=8)

    # (e) Eigenvalue spectrum
    for size, r in results.items():
        axes[1, 1].plot(range(len(r['eigenvalues'])), r['eigenvalues'],
                       '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[1, 1].set_xlabel('Eigenvalue Index')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].set_title('(e) MI Spectrum')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "MUTUAL INFORMATION\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Corr length: {r['corr_length']}\n"
        summary += f"  Spectral gap: {r['spectral_gap']:.4f}\n"
        summary += f"  Top eigenvals: {r['eigenvalues'][:3]}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 251: Mutual Information Between Layers",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase251_mutual_info')
    plt.close()
    save_results('phase251_mutual_info', {'experiment': 'Mutual Information', 'results': results})


if __name__ == '__main__':
    main()
