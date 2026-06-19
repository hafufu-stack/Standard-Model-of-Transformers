# -*- coding: utf-8 -*-
"""
Phase 348: Supersymmetry -- Boson-Fermion Pairing
=====================================================
SUSY pairs bosonic and fermionic degrees of freedom. In condensed
matter, this appears as particle-hole symmetry. Test whether the
hidden state spectrum has a paired structure (SUSY-like).
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


def measure_susy(model, tok, prompt, device):
    """Measure supersymmetric pairing."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Spectral pairing: does the eigenvalue spectrum come in +/- pairs?
    pairing_scores = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        sorted_vals = torch.sort(h)[0].numpy()
        # Fold around median
        median = float(np.median(sorted_vals))
        positive = sorted_vals[sorted_vals > median] - median
        negative = median - sorted_vals[sorted_vals < median]
        negative = np.sort(negative)

        # Compare distributions of |positive| and |negative|
        n_compare = min(len(positive), len(negative))
        if n_compare > 5:
            r, p = stats.pearsonr(positive[:n_compare], negative[:n_compare])
            pairing_scores.append(round(float(r), 4))
        else:
            pairing_scores.append(0.0)

    # 2. Witten index: Tr((-1)^F) where F = fermion number
    # Nonzero Witten index -> SUSY unbroken
    witten_indices = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Assign F=0 (boson) if h_i > 0, F=1 (fermion) if h_i < 0
        n_pos = int((h > 0).sum().item())
        n_neg = int((h < 0).sum().item())
        witten = abs(n_pos - n_neg)
        witten_indices.append(witten)

    # 3. Supercharge Q: Q|boson> = |fermion>, Q|fermion> = |boson>
    # Test if there's a matrix Q such that Q^2 ~ H (Hamiltonian)
    # Proxy: correlation between positive and negative sector dynamics
    sector_corrs = []
    for li in range(n_layers):
        h1 = hiddens[li]
        h2 = hiddens[li + 1]
        # Split into sectors
        pos_mask = h1 > 0
        neg_mask = h1 <= 0

        delta_pos = (h2[pos_mask] - h1[pos_mask]) if pos_mask.sum() > 0 else torch.zeros(1)
        delta_neg = (h2[neg_mask] - h1[neg_mask]) if neg_mask.sum() > 0 else torch.zeros(1)

        # Symmetry: do positive and negative sectors evolve similarly?
        n_compare = min(len(delta_pos), len(delta_neg))
        if n_compare > 5:
            r, _ = stats.pearsonr(
                delta_pos[:n_compare].numpy(),
                delta_neg[:n_compare].numpy()
            )
            sector_corrs.append(round(float(r), 4))
        else:
            sector_corrs.append(0.0)

    # 4. SUSY breaking order parameter: <0|Q|0>
    # If SUSY is broken, there's a goldstino (massless fermion)
    # Measure as the asymmetry of the ground state
    gs = hiddens[0]
    gs_skewness = float(stats.skew(gs.numpy()))

    avg_pairing = float(np.mean(pairing_scores))
    avg_witten = float(np.mean(witten_indices))
    avg_sector_corr = float(np.mean(sector_corrs)) if sector_corrs else 0

    return {
        'pairing_profile': pairing_scores,
        'witten_profile': witten_indices,
        'sector_corr_profile': sector_corrs,
        'avg_pairing': round(avg_pairing, 4),
        'avg_witten': round(avg_witten, 2),
        'avg_sector_corr': round(avg_sector_corr, 4),
        'gs_skewness': round(float(gs_skewness), 4),
        'susy_unbroken': avg_pairing > 0.5 and avg_witten > 0,
    }


def main():
    print("=" * 70)
    print("Phase 348: Supersymmetry")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        susy_data = []
        for prompt in PROMPTS:
            s = measure_susy(model, tok, prompt, device)
            susy_data.append(s)

        n = len(susy_data[0]['pairing_profile'])
        n_sc = len(susy_data[0]['sector_corr_profile'])
        all_results[size] = {
            'pairing_profile': [round(float(np.mean([s['pairing_profile'][i] for s in susy_data])), 4)
                               for i in range(n)],
            'witten_profile': [round(float(np.mean([s['witten_profile'][i] for s in susy_data])), 2)
                              for i in range(n)],
            'sector_corr_profile': [round(float(np.mean([s['sector_corr_profile'][i] for s in susy_data])), 4)
                                   for i in range(n_sc)],
            'avg_pairing': round(float(np.mean([s['avg_pairing'] for s in susy_data])), 4),
            'avg_witten': round(float(np.mean([s['avg_witten'] for s in susy_data])), 2),
            'avg_sector_corr': round(float(np.mean([s['avg_sector_corr'] for s in susy_data])), 4),
            'gs_skewness': round(float(np.mean([s['gs_skewness'] for s in susy_data])), 4),
            'susy_unbroken': sum(1 for s in susy_data if s['susy_unbroken']) >= 4,
        }
        susy = 'UNBROKEN' if all_results[size]['susy_unbroken'] else 'BROKEN'
        print(f"  Pairing: {all_results[size]['avg_pairing']:.4f}")
        print(f"  Witten: {all_results[size]['avg_witten']:.2f}")
        print(f"  Sector corr: {all_results[size]['avg_sector_corr']:.4f}")
        print(f"  SUSY: {susy}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['pairing_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Pairing R')
    axes[0, 0].set_title('(a) Spectral Pairing', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['witten_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Witten index')
    axes[0, 1].set_title('(b) Witten Index', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['sector_corr_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Sector correlation')
    axes[0, 2].set_title('(c) Sector Correlation', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['gs_skewness'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_title('(d) Ground State Skewness', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "SUPERSYMMETRY\n\n"
    for s in sizes:
        d = all_results[s]
        susy = 'UNBROKEN' if d['susy_unbroken'] else 'BROKEN'
        txt += f"{s}:\n"
        txt += f"  pair = {d['avg_pairing']:.3f}\n"
        txt += f"  W = {d['avg_witten']:.0f}\n"
        txt += f"  sector = {d['avg_sector_corr']:.3f}\n"
        txt += f"  SUSY: {susy}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 348: Supersymmetry", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase348_susy')
    plt.close()
    save_results('phase348_susy', {'experiment': 'Supersymmetry', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
