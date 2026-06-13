# -*- coding: utf-8 -*-
"""
Phase 171: Thermodynamic Phase Diagram
Construct a 2D phase diagram: kT (temperature) vs S (entropy).
Map each layer of each prompt into this space.
Identify phase boundaries and critical regions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    "The capital of France is Paris which is known for",
    "Two plus two equals four which everyone knows",
    "The meaning of life according to philosophy is",
    "What is consciousness and how does it arise from",
    "The best way to learn programming is through practice",
    "Climate change poses significant challenges to modern",
]


def main():
    print("=" * 70)
    print("Phase 171: Thermodynamic Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect (kT, S, eta, layer, prompt_id) tuples
    points_kT = []
    points_S = []
    points_eta = []
    points_layer = []
    points_conf = []

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S = S if not np.isnan(S) else 0
            T_vals.append(S)
            conf = probs.max().item()

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

            # eta
            T_sub = T_vals[:li+1]
            if len(T_sub) >= 4:
                T_hot = max(T_sub)
                T_cold = min(T_sub[len(T_sub)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0

            points_kT.append(float(kT))
            points_S.append(float(S))
            points_eta.append(float(eta))
            points_layer.append(li)
            points_conf.append(float(conf))

    points_kT = np.array(points_kT)
    points_S = np.array(points_S)
    points_eta = np.array(points_eta)
    points_layer = np.array(points_layer)
    points_conf = np.array(points_conf)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Phase diagram: kT vs S, colored by layer
    sc = axes[0,0].scatter(points_kT, points_S, c=points_layer, cmap='viridis',
                           s=15, alpha=0.6, edgecolors='none')
    plt.colorbar(sc, ax=axes[0,0], label='Layer')
    axes[0,0].set_xlabel('$kT$')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) Phase Diagram (kT vs S)')

    # (b) Phase diagram: kT vs S, colored by eta
    sc2 = axes[0,1].scatter(points_kT, points_S, c=points_eta, cmap='RdYlGn',
                            s=15, alpha=0.6, edgecolors='none')
    plt.colorbar(sc2, ax=axes[0,1], label='$\\eta$')
    axes[0,1].set_xlabel('$kT$')
    axes[0,1].set_ylabel('$S$')
    axes[0,1].set_title('(b) Phase Diagram (colored by eta)')

    # (c) 3D-like: S vs layer colored by kT
    sc3 = axes[0,2].scatter(points_layer, points_S, c=points_kT, cmap='hot',
                            s=15, alpha=0.5, edgecolors='none')
    plt.colorbar(sc3, ax=axes[0,2], label='$kT$')
    axes[0,2].axvline(x=21.7, color='cyan', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) Layer-S Diagram')

    # (d) Density plot: 2D histogram of kT vs S
    axes[1,0].hist2d(points_kT, points_S, bins=25, cmap='inferno')
    axes[1,0].set_xlabel('$kT$')
    axes[1,0].set_ylabel('$S$')
    axes[1,0].set_title('(d) Density in Phase Space')

    # (e) Phase boundaries: pre-L0 vs post-L0
    pre_mask = points_layer < 20
    post_mask = points_layer >= 22
    axes[1,1].scatter(points_kT[pre_mask], points_S[pre_mask],
                     c='#2980b9', s=15, alpha=0.4, label='Pre-$L_0$')
    axes[1,1].scatter(points_kT[post_mask], points_S[post_mask],
                     c='#c0392b', s=15, alpha=0.4, label='Post-$L_0$')
    axes[1,1].set_xlabel('$kT$')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) Phase Separation')
    axes[1,1].legend()

    # (f) Summary
    pre_kT = np.mean(points_kT[pre_mask])
    post_kT = np.mean(points_kT[post_mask])
    pre_S = np.mean(points_S[pre_mask])
    post_S = np.mean(points_S[post_mask])
    # Separation metric
    from scipy.spatial.distance import cdist
    pre_center = np.array([pre_kT, pre_S])
    post_center = np.array([post_kT, post_S])
    separation = np.linalg.norm(pre_center - post_center)

    summary = (
        f"Thermodynamic Phase Diagram\n\n"
        f"Pre-L0 center:\n"
        f"  kT={pre_kT:.1f}, S={pre_S:.2f}\n"
        f"Post-L0 center:\n"
        f"  kT={post_kT:.1f}, S={post_S:.2f}\n\n"
        f"Phase separation: {separation:.2f}\n\n"
        f"Two distinct phases\n"
        f"{'CONFIRMED' if separation > 2 else 'WEAK'}\n"
        f"in (kT, S) space"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 171: Thermodynamic Phase Diagram',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase171_phase_diagram')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-L0: kT={pre_kT:.1f}, S={pre_S:.2f}")
    print(f"Post-L0: kT={post_kT:.1f}, S={post_S:.2f}")
    print(f"Phase separation: {separation:.2f}")
    print(f"{'='*70}")

    save_results('phase171_phase_diagram', {
        'experiment': 'Phase Diagram',
        'summary': {
            'pre_kT': float(pre_kT),
            'post_kT': float(post_kT),
            'pre_S': float(pre_S),
            'post_S': float(post_S),
            'separation': float(separation),
        }
    })


if __name__ == '__main__':
    main()
