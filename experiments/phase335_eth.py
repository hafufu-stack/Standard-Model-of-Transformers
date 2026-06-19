# -*- coding: utf-8 -*-
"""
Phase 335: Eigenstate Thermalization Hypothesis (ETH)
=====================================================
ETH: in a chaotic quantum system, energy eigenstates look thermal.
The diagonal matrix elements of observables are smooth functions
of energy, and off-diagonal elements are exponentially small.
Test whether the Transformer's hidden state eigenstates thermalize.
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


def measure_eth(model, tok, prompt, device):
    """Test Eigenstate Thermalization Hypothesis."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    results_per_layer = []

    for li in range(1, n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu()
        dim = h.shape[0]

        # Construct correlation matrix from hidden state
        # Use a small subsystem for tractability
        sub_dim = min(128, dim)
        h_sub = h[:sub_dim]

        # Construct "Hamiltonian" from outer product
        rho = torch.outer(h_sub, h_sub)
        rho = (rho + rho.T) / 2  # Symmetrize

        # Diagonalize
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(rho)
            eigenvalues = eigenvalues.numpy()
            eigenvectors = eigenvectors.numpy()
        except:
            results_per_layer.append({
                'diagonal_smoothness': 0, 'offdiag_suppression': 0,
                'eth_score': 0
            })
            continue

        # ETH test 1: Diagonal matrix elements <n|O|n> are smooth
        # Observable O = h_sub (the hidden state itself)
        O = h_sub.numpy()
        diag_elements = []
        for n in range(min(sub_dim, 64)):
            elem = float(np.dot(eigenvectors[:, n], O))
            diag_elements.append(elem)

        # Smoothness: correlation between adjacent eigenstate expectations
        if len(diag_elements) > 2:
            smoothness_r, _ = stats.pearsonr(diag_elements[:-1], diag_elements[1:])
            smoothness = abs(float(smoothness_r))
        else:
            smoothness = 0

        # ETH test 2: Off-diagonal elements |<m|O|n>| are suppressed
        offdiag_vals = []
        for m in range(min(sub_dim, 32)):
            for n in range(m + 1, min(sub_dim, 32)):
                val = abs(float(np.dot(eigenvectors[:, m], O) * np.dot(eigenvectors[:, n], O)))
                offdiag_vals.append(val)

        diag_mean = float(np.mean(np.abs(diag_elements))) if diag_elements else 1
        offdiag_mean = float(np.mean(offdiag_vals)) if offdiag_vals else 1
        suppression = float(offdiag_mean / (diag_mean**2 + 1e-30))

        # ETH score: high smoothness + low off-diagonal
        eth_score = smoothness * (1.0 / (1.0 + suppression))

        results_per_layer.append({
            'diagonal_smoothness': round(smoothness, 4),
            'offdiag_suppression': round(suppression, 4),
            'eth_score': round(float(eth_score), 4),
        })

    avg_smooth = float(np.mean([r['diagonal_smoothness'] for r in results_per_layer]))
    avg_suppression = float(np.mean([r['offdiag_suppression'] for r in results_per_layer]))
    avg_eth = float(np.mean([r['eth_score'] for r in results_per_layer]))

    return {
        'smoothness_profile': [r['diagonal_smoothness'] for r in results_per_layer],
        'suppression_profile': [r['offdiag_suppression'] for r in results_per_layer],
        'eth_profile': [r['eth_score'] for r in results_per_layer],
        'avg_smoothness': round(avg_smooth, 4),
        'avg_suppression': round(avg_suppression, 4),
        'avg_eth': round(avg_eth, 4),
        'eth_holds': avg_smooth > 0.3 and avg_suppression < 2.0,
    }


def main():
    print("=" * 70)
    print("Phase 335: Eigenstate Thermalization Hypothesis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        eth_data = []
        for prompt in PROMPTS:
            e = measure_eth(model, tok, prompt, device)
            eth_data.append(e)

        n = len(eth_data[0]['eth_profile'])
        all_results[size] = {
            'smoothness_profile': [round(float(np.mean([e['smoothness_profile'][i] for e in eth_data])), 4)
                                  for i in range(n)],
            'suppression_profile': [round(float(np.mean([e['suppression_profile'][i] for e in eth_data])), 4)
                                   for i in range(n)],
            'eth_profile': [round(float(np.mean([e['eth_profile'][i] for e in eth_data])), 4)
                           for i in range(n)],
            'avg_smoothness': round(float(np.mean([e['avg_smoothness'] for e in eth_data])), 4),
            'avg_suppression': round(float(np.mean([e['avg_suppression'] for e in eth_data])), 4),
            'avg_eth': round(float(np.mean([e['avg_eth'] for e in eth_data])), 4),
            'eth_holds': sum(1 for e in eth_data if e['eth_holds']) >= 4,
        }
        holds = 'YES' if all_results[size]['eth_holds'] else 'NO'
        print(f"  Smoothness: {all_results[size]['avg_smoothness']:.4f}")
        print(f"  Suppression: {all_results[size]['avg_suppression']:.4f}")
        print(f"  ETH holds: {holds}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['smoothness_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Smoothness')
    axes[0, 0].set_title('(a) Diagonal Smoothness', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['suppression_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Off-diag ratio')
    axes[0, 1].set_title('(b) Off-Diagonal Suppression', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['eth_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('ETH score')
    axes[0, 2].set_title('(c) ETH Score', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.25
    axes[1, 0].bar(x - w/2, [all_results[s]['avg_smoothness'] for s in sizes], w,
                  label='Smoothness', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['avg_eth'] for s in sizes], w,
                  label='ETH', color='#2ecc71')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) Summary Scores', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "EIGENSTATE THERMALIZATION\n\n"
    txt += "Diagonal: smooth fn of E\n"
    txt += "Off-diag: exponentially small\n\n"
    for s in sizes:
        d = all_results[s]
        holds = 'YES' if d['eth_holds'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  smooth = {d['avg_smoothness']:.3f}\n"
        txt += f"  suppr = {d['avg_suppression']:.3f}\n"
        txt += f"  ETH: {holds}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 335: Eigenstate Thermalization Hypothesis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase335_eth')
    plt.close()
    save_results('phase335_eth', {'experiment': 'ETH', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
