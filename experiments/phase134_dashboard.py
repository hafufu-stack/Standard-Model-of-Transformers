# -*- coding: utf-8 -*-
"""
Phase 134: The Complete Thermodynamic Dashboard
Combine ALL thermodynamic quantities into one coherent picture.
Plot the full "state diagram" of a Transformer forward pass.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
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
]


def main():
    print("=" * 70)
    print("Phase 134: Complete Thermodynamic Dashboard")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Compute ALL quantities at each layer
    all_S = [[] for _ in range(n_layers)]
    all_eta = [[] for _ in range(n_layers)]
    all_kT = [[] for _ in range(n_layers)]
    all_skew = [[] for _ in range(n_layers)]
    all_U = [[] for _ in range(n_layers)]
    all_cos = []  # cos sim between consecutive layers

    from scipy import stats as sp_stats

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        prev_h = None
        cos_sims = []

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            # Entropy
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(S): S = 0
            T_vals.append(S)
            all_S[li].append(S)

            # kT
            top_k = 50
            top_probs = torch.topk(probs, top_k).values
            log_probs = torch.log(top_probs + 1e-10).cpu().numpy()
            ranks = np.arange(1, top_k + 1, dtype=np.float64)
            if np.std(log_probs) > 0.01:
                slope = np.polyfit(ranks, log_probs, 1)[0]
                kT = -1.0 / (slope + 1e-10)
            else:
                kT = 0.1
            kT = max(0.01, min(kT, 50))
            all_kT[li].append(float(kT))

            # Skewness
            h_np = h.cpu().numpy()
            sk = sp_stats.skew(h_np)
            all_skew[li].append(float(sk) if not np.isnan(sk) else 0)

            # Internal energy
            U = (h ** 2).mean().item()
            all_U[li].append(float(U))

            # Cosine similarity
            if prev_h is not None:
                cos = torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item()
                cos_sims.append(cos)
            prev_h = h

        # Eta
        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0
            all_eta[li].append(eta)

        all_cos.append(cos_sims)

    # Averages
    avg = lambda x: [np.mean(v) if v else 0 for v in x]
    S = avg(all_S)
    eta = avg(all_eta)
    kT = avg(all_kT)
    skew = avg(all_skew)
    U = avg(all_U)
    cos_sim = np.mean(all_cos, axis=0).tolist() if all_cos else []

    # Derivatives
    dS = np.gradient(S)
    deta = np.gradient(eta)
    dkT = np.gradient(kT)

    layers = np.arange(n_layers)

    # === Grand Visualization: 3x3 Dashboard ===
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    L0_color = '#f39c12'

    # Row 1: Order Parameter
    axes[0,0].plot(layers, eta, 'o-', color='#c0392b', markersize=3, linewidth=2)
    axes[0,0].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[0,0].set_ylabel('$\\eta$ (order parameter)')
    axes[0,0].set_title('Order Parameter')

    axes[0,1].plot(layers, skew, 'o-', color='#8e44ad', markersize=3, linewidth=2)
    axes[0,1].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[0,1].axhline(y=0, color='gray', linewidth=0.5)
    axes[0,1].set_ylabel('Skewness')
    axes[0,1].set_title('Symmetry (skewness)')

    axes[0,2].plot(layers, deta, 'o-', color='#e67e22', markersize=3, linewidth=2)
    axes[0,2].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[0,2].set_ylabel('$d\\eta/dL$')
    axes[0,2].set_title('Susceptibility')

    # Row 2: Thermodynamics
    axes[1,0].plot(layers, S, 'o-', color='#2980b9', markersize=3, linewidth=2)
    axes[1,0].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[1,0].set_ylabel('$S$ (entropy)')
    axes[1,0].set_title('Output Entropy')

    axes[1,1].plot(layers, kT, 'o-', color='#c0392b', markersize=3, linewidth=2)
    axes[1,1].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[1,1].set_ylabel('$kT$')
    axes[1,1].set_title('Temperature')

    axes[1,2].plot(layers, U, 'o-', color='#27ae60', markersize=3, linewidth=2)
    axes[1,2].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[1,2].set_ylabel('$U = \\langle|h|^2\\rangle$')
    axes[1,2].set_title('Internal Energy')

    # Row 3: Transport
    dS_colors = ['#c0392b' if d > 0 else '#2980b9' for d in dS]
    axes[2,0].bar(layers, dS, color=dS_colors, alpha=0.7, edgecolor='black')
    axes[2,0].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[2,0].set_ylabel('$\\sigma = dS/dL$')
    axes[2,0].set_xlabel('Layer')
    axes[2,0].set_title('Entropy Production')

    if cos_sim:
        axes[2,1].plot(np.arange(len(cos_sim))+0.5, cos_sim, 'o-', color='#e74c3c',
                      markersize=3, linewidth=2)
        axes[2,1].axvline(x=L0, color=L0_color, linewidth=2, linestyle='--')
    axes[2,1].set_ylabel('Cosine Similarity')
    axes[2,1].set_xlabel('Layer')
    axes[2,1].set_title('Layer Redundancy')

    # Phase diagram
    sc = axes[2,2].scatter(eta[4:], kT[4:], c=S[4:], s=80, cmap='coolwarm',
                           edgecolors='black')
    for i, li in enumerate(range(4, n_layers)):
        if i % 3 == 0:
            axes[2,2].annotate(f'{li}', (eta[li], kT[li]), fontsize=6)
    plt.colorbar(sc, ax=axes[2,2], label='$S$')
    axes[2,2].set_xlabel('$\\eta$')
    axes[2,2].set_ylabel('$kT$')
    axes[2,2].set_title('Phase Diagram')

    fig.suptitle('Phase 134: Complete Thermodynamic Dashboard\n'
                 'Standard Model of Transformers',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase134_dashboard')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Dashboard complete. All quantities visualized.")
    print(f"{'='*70}")

    save_results('phase134_dashboard', {
        'experiment': 'Complete Thermodynamic Dashboard',
        'S': S, 'eta': eta, 'kT': kT, 'skew': skew, 'U': U,
        'cos_sim': cos_sim,
    })


if __name__ == '__main__':
    main()
