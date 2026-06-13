# -*- coding: utf-8 -*-
"""
Phase 77: Thermodynamic Phase Diagram (T-U-PR 3D mapping)
Create the complete phase diagram of the Transformer's thermodynamic state space.
Map out the attractor basin, collapse regions, and transition boundaries.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 77: Thermodynamic Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Diverse prompts to map the full phase space
    prompts = [
        # Factual
        "Water boils at one hundred degrees Celsius at standard atmospheric",
        "The speed of light is approximately three hundred million",
        # Reasoning
        "If x plus three equals seven then x equals",
        "The derivative of x squared is two times x because",
        # Creative
        "Once upon a time in a magical forest there lived a",
        "The taste of moonlight on a winter evening reminds me of",
        # Nonsense (should be hot/uncertain)
        "Colorless green ideas sleep furiously while the abstract",
        "The square root of purple tastes like seventeen divided by",
        # Technical
        "The eigenvalues of a symmetric matrix are always real because",
        "In a TCP handshake the client first sends a SYN packet to",
        # Long context
        "The history of artificial intelligence begins in ancient times with myths and stories of artificial beings endowed with intelligence",
        "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy",
    ]

    all_points = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()

            top1 = probs.max().item()
            PRT = PR * T

            all_points.append({
                'layer': li, 'U': float(U), 'T': float(T),
                'PR': float(PR), 'PRT': float(PRT),
                'top1': float(top1), 'prompt': prompt[:30],
            })

    n_layers = max(p['layer'] for p in all_points) + 1

    # === Visualization ===
    fig = plt.figure(figsize=(18, 12))

    # (a) 3D phase diagram: T-U-PR
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    layers_arr = np.array([p['layer'] for p in all_points])
    T_arr = np.array([p['T'] for p in all_points])
    U_arr = np.array([p['U'] for p in all_points])
    PR_arr = np.array([p['PR'] for p in all_points])
    sc = ax1.scatter(T_arr, U_arr, PR_arr, c=layers_arr, cmap='viridis',
                     s=5, alpha=0.3)
    ax1.set_xlabel('T (entropy)')
    ax1.set_ylabel('U (energy)')
    ax1.set_zlabel('PR')
    ax1.set_title('(a) 3D Phase Space')

    # (b) T-U phase diagram (2D projection)
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(T_arr, U_arr, c=layers_arr, cmap='viridis', s=5, alpha=0.3)
    # Draw trajectory means
    for li in range(n_layers):
        mask = layers_arr == li
        if mask.any():
            ax2.scatter(T_arr[mask].mean(), U_arr[mask].mean(),
                       s=50, c='red', edgecolors='black', zorder=5)
    ax2.set_xlabel('T (entropy)')
    ax2.set_ylabel('U (energy)')
    ax2.set_title('(b) T-U Phase Diagram')
    plt.colorbar(sc2, ax=ax2, label='Layer')

    # (c) T-PR diagram
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(T_arr, PR_arr, c=layers_arr, cmap='viridis', s=5, alpha=0.3)
    ax3.set_xlabel('T')
    ax3.set_ylabel('PR')
    ax3.set_title('(c) T-PR Phase Diagram')

    # (d) Layer-averaged phase trajectory
    ax4 = fig.add_subplot(2, 3, 4)
    mean_T = [T_arr[layers_arr == l].mean() for l in range(n_layers)]
    mean_U = [U_arr[layers_arr == l].mean() for l in range(n_layers)]
    ax4.plot(mean_T, mean_U, 'o-', color='#e74c3c', markersize=5, linewidth=2)
    ax4.scatter(mean_T[0], mean_U[0], s=100, c='green', edgecolors='black',
               zorder=5, label='L0 (start)')
    ax4.scatter(mean_T[-1], mean_U[-1], s=100, c='red', edgecolors='black',
               zorder=5, label=f'L{n_layers-1} (end)')
    ax4.set_xlabel('T')
    ax4.set_ylabel('U')
    ax4.set_title('(d) Mean Trajectory (T-U)')
    ax4.legend()
    # Annotate direction
    for i in range(0, n_layers, 7):
        ax4.annotate(f'L{i}', (mean_T[i], mean_U[i]), fontsize=7)

    # (e) PRT conservation surface
    ax5 = fig.add_subplot(2, 3, 5)
    PRT_arr = np.array([p['PRT'] for p in all_points])
    mean_PRT = [PRT_arr[layers_arr == l].mean() for l in range(n_layers)]
    std_PRT = [PRT_arr[layers_arr == l].std() for l in range(n_layers)]
    ax5.fill_between(range(n_layers),
                     [m-s for m, s in zip(mean_PRT, std_PRT)],
                     [m+s for m, s in zip(mean_PRT, std_PRT)],
                     alpha=0.3, color='#3498db')
    ax5.plot(range(n_layers), mean_PRT, 'o-', color='#3498db', linewidth=2, markersize=3)
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('PRT')
    ax5.set_title('(e) PRT Surface')

    # (f) Phase classification
    ax6 = fig.add_subplot(2, 3, 6)
    # Classify each point: hot/cold, ordered/disordered
    T_median = np.median(T_arr)
    U_median = np.median(U_arr)
    phases = []
    for p in all_points:
        if p['T'] > T_median and p['U'] > U_median:
            phases.append('Hot-Energetic')
        elif p['T'] > T_median and p['U'] <= U_median:
            phases.append('Hot-Relaxed')
        elif p['T'] <= T_median and p['U'] > U_median:
            phases.append('Cold-Energetic')
        else:
            phases.append('Cold-Relaxed')

    phase_counts = {}
    for ph in set(phases):
        phase_counts[ph] = phases.count(ph)

    phase_colors = {'Hot-Energetic': '#e74c3c', 'Hot-Relaxed': '#f39c12',
                    'Cold-Energetic': '#3498db', 'Cold-Relaxed': '#2ecc71'}
    ax6.bar(list(phase_counts.keys()), list(phase_counts.values()),
            color=[phase_colors.get(p, 'gray') for p in phase_counts.keys()],
            alpha=0.8)
    ax6.set_ylabel('Count')
    ax6.set_title('(f) Phase Classification')
    plt.xticks(rotation=20, ha='right', fontsize=8)

    fig.suptitle('Phase 77: Thermodynamic Phase Diagram of Transformer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase77_phase_diagram')
    plt.close()

    # Trajectory analysis
    T_drop = mean_T[0] - mean_T[-1]
    U_rise = mean_U[-1] - mean_U[0]
    trajectory = 'cooling+compression' if T_drop > 0 and U_rise > 0 else 'other'

    print(f"\n{'='*70}")
    print(f"VERDICT: Trajectory: T drops by {T_drop:.1f}, U rises by {U_rise:.0f}. "
          f"Pattern: {trajectory}. Dominant phase: {max(phase_counts, key=phase_counts.get)}.")
    print(f"{'='*70}")

    save_results('phase77_phase_diagram', {
        'experiment': 'Phase Diagram',
        'summary': {
            'T_drop': float(T_drop),
            'U_rise': float(U_rise),
            'trajectory': trajectory,
            'dominant_phase': max(phase_counts, key=phase_counts.get),
        }
    })


if __name__ == '__main__':
    main()
