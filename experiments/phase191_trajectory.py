# -*- coding: utf-8 -*-
"""
Phase 191: Trajectory Geometry in Probability Space
=====================================================
Phase 187 showed 11.38x geodesic detour. What SHAPE is this detour?
Project the probability trajectory onto principal components and
measure: curvature, torsion, winding number.

Is the computation a spiral, an arc, a random walk, or a helix?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    "Cryptographic hash functions ensure data integrity",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "Superconductors carry current with zero resistance",
    "Artificial neural networks are inspired by biological neurons",
]


def compute_curvature_torsion(trajectory):
    """Compute discrete curvature and torsion of a 3D trajectory."""
    n = len(trajectory)
    curvatures = []
    torsions = []

    for i in range(1, n - 1):
        # Tangent vectors
        t1 = trajectory[i] - trajectory[i-1]
        t2 = trajectory[i+1] - trajectory[i]
        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 < 1e-10 or n2 < 1e-10:
            curvatures.append(0)
            continue
        t1_hat = t1 / n1
        t2_hat = t2 / n2

        # Curvature = |dt/ds| ~ angle between consecutive tangents
        cos_angle = np.clip(np.dot(t1_hat, t2_hat), -1, 1)
        kappa = np.arccos(cos_angle) / ((n1 + n2) / 2 + 1e-10)
        curvatures.append(kappa)

    for i in range(2, n - 1):
        t1 = trajectory[i-1] - trajectory[i-2]
        t2 = trajectory[i] - trajectory[i-1]
        t3 = trajectory[i+1] - trajectory[i]
        b1 = np.cross(t1, t2)
        b2 = np.cross(t2, t3)
        nb1, nb2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if nb1 < 1e-10 or nb2 < 1e-10:
            torsions.append(0)
            continue
        cos_tau = np.clip(np.dot(b1/nb1, b2/nb2), -1, 1)
        tau = np.arccos(cos_tau) / (np.linalg.norm(t2) + 1e-10)
        torsions.append(tau)

    return curvatures, torsions


def main():
    print("=" * 70)
    print("Phase 191: Trajectory Geometry")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    # Collect probability distributions at each layer for all prompts
    all_trajectories_3d = []
    all_curvatures = []
    all_torsions = []
    mean_sqrt_p = None  # For mean trajectory

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Collect sqrt(p) at each layer (Bhattacharyya embedding)
        sqrt_p_matrix = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            sqrt_p = torch.sqrt(probs).cpu().numpy()
            sqrt_p_matrix.append(sqrt_p)

        sqrt_p_matrix = np.array(sqrt_p_matrix)  # (n_layers, vocab)

        if mean_sqrt_p is None:
            mean_sqrt_p = sqrt_p_matrix
        else:
            mean_sqrt_p += sqrt_p_matrix

        # PCA for this prompt
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=3, random_state=42)
        trajectory_3d = svd.fit_transform(sqrt_p_matrix)
        all_trajectories_3d.append(trajectory_3d)

        # Curvature and torsion
        curv, tors = compute_curvature_torsion(trajectory_3d)
        all_curvatures.append(curv)
        all_torsions.append(tors)

    mean_sqrt_p /= len(PROMPTS)

    # Mean trajectory PCA
    from sklearn.decomposition import TruncatedSVD
    svd_mean = TruncatedSVD(n_components=3, random_state=42)
    mean_traj = svd_mean.fit_transform(mean_sqrt_p)
    variance_explained = svd_mean.explained_variance_ratio_

    mean_curv, mean_tors = compute_curvature_torsion(mean_traj)

    # Total curvature (integral of |kappa|)
    total_curv = sum(mean_curv)
    mean_kappa = np.mean(mean_curv) if mean_curv else 0
    mean_tau = np.mean(mean_tors) if mean_tors else 0

    # === Visualization ===
    fig = plt.figure(figsize=(18, 10))

    # (a) 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    for i in range(n_layers - 1):
        ax1.plot(mean_traj[i:i+2, 0], mean_traj[i:i+2, 1], mean_traj[i:i+2, 2],
                 '-', color=colors[i], linewidth=2)
    ax1.scatter(*mean_traj[0], s=150, marker='s', c='green', edgecolors='black',
                zorder=10, label='Start')
    ax1.scatter(*mean_traj[-1], s=150, marker='*', c='red', edgecolors='black',
                zorder=10, label='End')
    ax1.scatter(*mean_traj[L0], s=150, marker='D', c='#f39c12', edgecolors='black',
                zorder=10, label=f'$L_0$')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title(f'(a) Trajectory in Probability Space\n'
                  f'VE: {variance_explained[0]:.1%}, {variance_explained[1]:.1%}, {variance_explained[2]:.1%}')
    ax1.legend(fontsize=7)

    # (b) 2D projection (PC1 vs PC2)
    ax2 = fig.add_subplot(2, 3, 2)
    for pi in range(min(5, len(all_trajectories_3d))):
        t = all_trajectories_3d[pi]
        ax2.plot(t[:, 0], t[:, 1], '-', alpha=0.2, linewidth=1)
    ax2.plot(mean_traj[:, 0], mean_traj[:, 1], 'ko-', markersize=3, linewidth=2, label='Mean')
    ax2.scatter([mean_traj[0, 0]], [mean_traj[0, 1]], s=100, marker='s', c='green',
                edgecolors='black', zorder=10)
    ax2.scatter([mean_traj[-1, 0]], [mean_traj[-1, 1]], s=100, marker='*', c='red',
                edgecolors='black', zorder=10)
    ax2.scatter([mean_traj[L0, 0]], [mean_traj[L0, 1]], s=100, marker='D', c='#f39c12',
                edgecolors='black', zorder=10)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('(b) PC1-PC2 Projection')

    # (c) Curvature profile
    ax3 = fig.add_subplot(2, 3, 3)
    if mean_curv:
        layers_c = np.arange(1, len(mean_curv) + 1)
        ax3.plot(layers_c, mean_curv, 'o-', color='#e74c3c', markersize=4, linewidth=2)
        ax3.axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Curvature $\\kappa$')
    ax3.set_title(f'(c) Path Curvature (total={total_curv:.2f})')
    ax3.legend(fontsize=8)

    # (d) Torsion profile
    ax4 = fig.add_subplot(2, 3, 4)
    if mean_tors:
        layers_t = np.arange(2, len(mean_tors) + 2)
        ax4.plot(layers_t, mean_tors, 's-', color='#8e44ad', markersize=4, linewidth=2)
        ax4.axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Torsion $\\tau$')
    ax4.set_title('(d) Path Torsion (out-of-plane bending)')

    # (e) Speed profile (step size in PC space)
    ax5 = fig.add_subplot(2, 3, 5)
    speeds = [np.linalg.norm(mean_traj[i+1] - mean_traj[i]) for i in range(n_layers-1)]
    ax5.plot(np.arange(n_layers-1) + 0.5, speeds, 'o-', color='#2ecc71', markersize=4, linewidth=2)
    ax5.axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Speed $|\\Delta x|$')
    ax5.set_title('(e) Speed in Probability Space')

    # (f) Summary
    ax6 = fig.add_subplot(2, 3, 6)
    # Classify shape
    if mean_tau > 0.01 and total_curv > 1:
        shape = "HELIX (3D spiral)"
    elif total_curv > 1:
        shape = "ARC / 2D SPIRAL"
    elif total_curv < 0.5:
        shape = "NEAR-STRAIGHT"
    else:
        shape = "CURVED PATH"

    summary = (
        f"Trajectory Geometry\n\n"
        f"Shape: {shape}\n\n"
        f"Variance explained:\n"
        f"  PC1: {variance_explained[0]:.1%}\n"
        f"  PC2: {variance_explained[1]:.1%}\n"
        f"  PC3: {variance_explained[2]:.1%}\n\n"
        f"Total curvature: {total_curv:.2f}\n"
        f"Mean curvature: {mean_kappa:.4f}\n"
        f"Mean torsion: {mean_tau:.4f}\n"
        f"Mean speed: {np.mean(speeds):.4f}"
    )
    ax6.text(0.5, 0.5, summary, ha='center', va='center',
             transform=ax6.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             family='monospace')
    ax6.axis('off')
    ax6.set_title('(f) Summary')

    fig.suptitle('Phase 191: Trajectory Geometry in Probability Space', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase191_trajectory')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Shape: {shape}")
    print(f"VE: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}, PC3={variance_explained[2]:.1%}")
    print(f"Total curvature: {total_curv:.2f}, Mean torsion: {mean_tau:.4f}")
    print(f"{'=' * 70}")

    save_results('phase191_trajectory', {
        'experiment': 'Trajectory Geometry',
        'shape': shape,
        'variance_explained': [float(x) for x in variance_explained],
        'total_curvature': float(total_curv),
        'mean_curvature': float(mean_kappa),
        'mean_torsion': float(mean_tau),
        'mean_speed': float(np.mean(speeds)),
    })


if __name__ == '__main__':
    main()
