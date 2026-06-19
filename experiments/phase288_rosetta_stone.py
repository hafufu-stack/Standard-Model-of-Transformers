# -*- coding: utf-8 -*-
"""
Phase 288: Rosetta Stone -- Quantum Information <-> Thermodynamics Dictionary
===============================================================================
Build quantitative mapping between S-Qubit (quantum info) and
Standard Model (thermodynamic) descriptions of the same phenomena.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of general relativity predicts that",
    "Quantum mechanics describes particles as waves",
    "The most effective way to solve climate change",
    "Once upon a time in a kingdom far away",
    "Machine learning models learn patterns from data",
    "The chemical composition of water molecules is",
    "The speed of light is constant in all frames",
    "Artificial intelligence will transform how we live",
    "Evolution explains the diversity of life on Earth",
    "The structure of the atom includes a nucleus",
]


def measure_quantum_properties(model, tok, prompt, device):
    """Measure quantum-information properties of a prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    h_final = out.hidden_states[-1][0, -1, :].float()
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    # Superposition: how many tokens have non-negligible probability
    threshold = 0.01
    n_superposed = (probs > threshold).sum().item()

    # Von Neumann entropy (output distribution)
    vn_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    # Purity (1 - mixedness)
    purity = (probs ** 2).sum().item()

    # Coherence: off-diagonal elements of density matrix proxy
    # Use hidden state covariance across positions
    h_all = out.hidden_states[-1][0].float()  # (seq, hidden)
    if h_all.shape[0] >= 3:
        # Gram matrix normalized
        h_norm = h_all / (h_all.norm(dim=1, keepdim=True) + 1e-10)
        gram = h_norm @ h_norm.T  # (seq, seq)
        # Off-diagonal sum = coherence
        mask = 1 - torch.eye(gram.shape[0], device=gram.device)
        coherence = (gram * mask).abs().sum().item() / mask.sum().item()
    else:
        coherence = 0.0

    # Entanglement: mutual info between first/second half of hidden state
    h_dim = h_final.shape[0]
    h1 = h_final[:h_dim//2]
    h2 = h_final[h_dim//2:]
    corr = torch.nn.functional.cosine_similarity(
        h1.unsqueeze(0), h2.unsqueeze(0)).item()
    entanglement = abs(corr)

    return {
        'n_superposed': n_superposed,
        'vn_entropy': round(vn_entropy, 4),
        'purity': round(purity, 6),
        'coherence': round(coherence, 4),
        'entanglement': round(entanglement, 4),
    }


def measure_thermo_properties(model, tok, prompt, device):
    """Measure thermodynamic properties of a prompt."""
    results, out = measure_full_thermodynamics(model, tok, prompt, device)

    # Key thermodynamic quantities at final layer
    final = results[-1]
    U = final['U']
    T = final['T']
    PR = final['PR']
    P1T = final['PRT']

    # Free energy
    F = U - T

    # Specific heat from middle layers
    mid = len(results) // 2
    dT = results[mid+1]['T'] - results[mid]['T']
    dU = results[mid+1]['U'] - results[mid]['U']
    Cv = dT / (dU + 1e-10)

    # Entropy production rate
    dS = [results[i+1]['T'] - results[i]['T'] for i in range(len(results)-1)]
    entropy_production = float(np.std(dS))

    return {
        'U': round(U, 4),
        'T': round(T, 4),
        'PR': round(PR, 2),
        'P1T': round(P1T, 4),
        'F': round(F, 4),
        'Cv': round(Cv, 4),
        'entropy_production': round(entropy_production, 4),
    }


def main():
    print("=" * 70)
    print("Phase 288: Rosetta Stone")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        quantum_data = []
        thermo_data = []

        for pi, prompt in enumerate(PROMPTS):
            q = measure_quantum_properties(model, tok, prompt, device)
            t = measure_thermo_properties(model, tok, prompt, device)
            quantum_data.append(q)
            thermo_data.append(t)

            if pi % 5 == 0:
                print(f"  Processed {pi+1}/{len(PROMPTS)}")

        # Build correlation dictionary
        q_keys = ['vn_entropy', 'purity', 'coherence', 'entanglement', 'n_superposed']
        t_keys = ['U', 'T', 'PR', 'P1T', 'F', 'Cv']

        rosetta = {}
        for qk in q_keys:
            for tk in t_keys:
                q_vals = np.array([d[qk] for d in quantum_data])
                t_vals = np.array([d[tk] for d in thermo_data])
                if np.std(q_vals) < 1e-10 or np.std(t_vals) < 1e-10:
                    r, p = 0.0, 1.0
                else:
                    r, p = stats.pearsonr(q_vals, t_vals)
                rosetta[f"{qk}<->{tk}"] = {
                    'r': round(float(r), 4),
                    'p': round(float(p), 6),
                    'significant': p < 0.05,
                }

        # Find strongest correspondences
        sorted_pairs = sorted(rosetta.items(), key=lambda x: abs(x[1]['r']), reverse=True)
        top_5 = sorted_pairs[:5]

        all_results[size] = {
            'rosetta': rosetta,
            'top_correspondences': {k: v for k, v in top_5},
            'quantum_means': {k: round(float(np.mean([d[k] for d in quantum_data])), 4)
                            for k in q_keys},
            'thermo_means': {k: round(float(np.mean([d[k] for d in thermo_data])), 4)
                           for k in t_keys},
        }

        print("\n  Top 5 correspondences:")
        for pair, data in top_5:
            sig = "*" if data['significant'] else ""
            print(f"    {pair}: r={data['r']:.3f}{sig}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Correlation heatmap (first model)
    first_size = list(all_results.keys())[0]
    data = all_results[first_size]
    corr_matrix = np.zeros((len(q_keys), len(t_keys)))
    for i, qk in enumerate(q_keys):
        for j, tk in enumerate(t_keys):
            corr_matrix[i, j] = data['rosetta'][f"{qk}<->{tk}"]['r']

    im = axes[0, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0, 0].set_xticks(range(len(t_keys)))
    axes[0, 0].set_xticklabels(t_keys, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_yticks(range(len(q_keys)))
    axes[0, 0].set_yticklabels([k[:8] for k in q_keys], fontsize=8)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[0, 0].set_title(f'(a) Rosetta Stone ({first_size})', fontweight='bold')

    # (b) Correlation heatmap (second model)
    if len(all_results) > 1:
        second_size = list(all_results.keys())[1]
        data2 = all_results[second_size]
        corr_matrix2 = np.zeros((len(q_keys), len(t_keys)))
        for i, qk in enumerate(q_keys):
            for j, tk in enumerate(t_keys):
                corr_matrix2[i, j] = data2['rosetta'][f"{qk}<->{tk}"]['r']
        im2 = axes[0, 1].imshow(corr_matrix2, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0, 1].set_xticks(range(len(t_keys)))
        axes[0, 1].set_xticklabels(t_keys, rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_yticks(range(len(q_keys)))
        axes[0, 1].set_yticklabels([k[:8] for k in q_keys], fontsize=8)
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        axes[0, 1].set_title(f'(b) Rosetta Stone ({second_size})', fontweight='bold')

    # (c) Cross-model consistency
    if len(all_results) > 1:
        r1 = [data['rosetta'][k]['r'] for k in data['rosetta']]
        r2 = [data2['rosetta'][k]['r'] for k in data2['rosetta']]
        axes[0, 2].scatter(r1, r2, s=30, alpha=0.6, c='#2ecc71')
        axes[0, 2].plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
        r_cross, _ = stats.pearsonr(r1, r2)
        axes[0, 2].set_xlabel(f'{first_size} correlation')
        axes[0, 2].set_ylabel(f'{second_size} correlation')
        axes[0, 2].set_title(f'(c) Cross-Model Consistency (r={r_cross:.3f})',
                           fontweight='bold')
        axes[0, 2].grid(alpha=0.3)

    # (d) Top correspondences
    top_pairs = list(data['top_correspondences'].keys())[:5]
    top_rs = [abs(data['top_correspondences'][k]['r']) for k in top_pairs]
    axes[1, 0].barh(range(len(top_pairs)), top_rs, color='#e74c3c')
    axes[1, 0].set_yticks(range(len(top_pairs)))
    axes[1, 0].set_yticklabels([p.replace('<->', '\n<->') for p in top_pairs], fontsize=7)
    axes[1, 0].set_xlabel('|r|')
    axes[1, 0].set_title('(d) Strongest Correspondences', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Significant pairs count
    for size, sdata in all_results.items():
        n_sig = sum(1 for v in sdata['rosetta'].values() if v['significant'])
        n_total = len(sdata['rosetta'])
        axes[1, 1].bar(size, n_sig, color={'0.5B': '#3498db', '1.5B': '#e74c3c'}.get(size, '#999'))
        axes[1, 1].text(size, n_sig + 0.5, f"{n_sig}/{n_total}", ha='center')
    axes[1, 1].set_ylabel('# Significant Pairs (p<0.05)')
    axes[1, 1].set_title('(e) Significant Correspondences', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # (f) Summary / Dictionary
    txt = "ROSETTA STONE\n"
    txt += "Quantum Info <-> Thermodynamics\n\n"
    txt += "Top Universal Correspondences:\n"
    # Find pairs that are significant in both models
    if len(all_results) > 1:
        for pair in data['rosetta']:
            r1 = data['rosetta'][pair]
            r2 = data2['rosetta'][pair]
            if r1['significant'] and r2['significant'] and abs(r1['r']) > 0.5:
                txt += f"  {pair}\n"
                txt += f"    0.5B: r={r1['r']:.3f}\n"
                txt += f"    1.5B: r={r2['r']:.3f}\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Universal Dictionary')

    fig.suptitle("Phase 288: Rosetta Stone -- Quantum Info <-> Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase288_rosetta_stone')
    plt.close()

    save_results('phase288_rosetta_stone', {
        'experiment': 'Rosetta Stone - QI to Thermo Dictionary',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
