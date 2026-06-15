# -*- coding: utf-8 -*-
"""
Phase 236: Thermodynamic Phase Diagram
========================================
Construct the complete phase diagram of the transformer in
(T, U, P1) space. Identify phase boundaries, critical exponents,
and order parameter behavior.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
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


def phase_diagram(model, tok, device, model_name):
    """Construct phase diagram in (T, U, P1) space."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_points = []  # (T, U, P1, layer, prompt_idx)
    all_trajectories = []

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        traj = []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T = float(S) if not np.isnan(S) else 0
            all_points.append((T, U, P1, li, pi))
            traj.append((T, U, P1))
        all_trajectories.append(traj)

    # Phase boundary detection: find layers where d^2T/dl^2 changes sign
    n = min(len(t) for t in all_trajectories)
    avg_T = [np.mean([all_trajectories[p][l][0] for p in range(len(PROMPTS))]) for l in range(n)]
    avg_U = [np.mean([all_trajectories[p][l][1] for p in range(len(PROMPTS))]) for l in range(n)]
    avg_P1 = [np.mean([all_trajectories[p][l][2] for p in range(len(PROMPTS))]) for l in range(n)]

    dT = [avg_T[i+1] - avg_T[i] for i in range(n-1)]
    d2T = [dT[i+1] - dT[i] for i in range(len(dT)-1)]

    # Sign changes in d2T = inflection points = phase boundaries
    phase_boundaries = []
    for i in range(len(d2T)-1):
        if d2T[i] * d2T[i+1] < 0:
            phase_boundaries.append(i+1)

    # Critical exponent: fit P1 ~ (l - l_c)^beta near phase boundary
    critical_exponents = []
    for lb in phase_boundaries:
        # Use 5 layers around boundary
        start_l = max(0, lb - 5)
        end_l = min(n-1, lb + 5)
        x_data = np.array(range(start_l, end_l + 1)) - lb
        y_data = np.array(avg_P1[start_l:end_l+1])
        # Fit log(P1) ~ beta * log(|x|)
        mask = x_data != 0
        if mask.sum() >= 3:
            try:
                log_x = np.log(np.abs(x_data[mask]))
                log_y = np.log(y_data[mask] + 1e-10)
                slope, _, r, _, _ = stats.linregress(log_x, log_y)
                critical_exponents.append({'layer': lb, 'beta': float(slope), 'r': float(r)})
            except Exception:
                pass

    # Susceptibility: chi = d<P1>/dT
    chi = []
    for l in range(n-1):
        dP1_dT = (avg_P1[l+1] - avg_P1[l]) / (avg_T[l+1] - avg_T[l] + 1e-10)
        chi.append(float(dP1_dT))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'avg_T': [float(x) for x in avg_T],
        'avg_U': [float(x) for x in avg_U],
        'avg_P1': [float(x) for x in avg_P1],
        'dT': [float(x) for x in dT],
        'd2T': [float(x) for x in d2T],
        'phase_boundaries': phase_boundaries,
        'critical_exponents': critical_exponents,
        'chi': chi,
        'trajectories': [[(float(t), float(u), float(p)) for t,u,p in traj] for traj in all_trajectories[:5]],
    }


def main():
    print("=" * 70)
    print("Phase 236: Thermodynamic Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = phase_diagram(model, tok, device, size)
        results[size] = r
        print(f"  Phase boundaries: {r['phase_boundaries']}")
        for ce in r['critical_exponents']:
            print(f"  Critical exponent at L{ce['layer']}: beta={ce['beta']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig = plt.figure(figsize=(18, 12))

    # (a) 3D phase diagram (1.5B)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    r15 = results['1.5B']
    # Color by layer index
    n = len(r15['avg_T'])
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))
    ax1.scatter(r15['avg_T'], r15['avg_U'], r15['avg_P1'],
               c=range(n), cmap='coolwarm', s=40)
    ax1.plot(r15['avg_T'], r15['avg_U'], r15['avg_P1'], 'k-', alpha=0.3)
    ax1.set_xlabel('T'); ax1.set_ylabel('U'); ax1.set_zlabel('P1')
    ax1.set_title('(a) Phase Diagram (1.5B)')

    # (b) T-P1 phase plane
    ax2 = fig.add_subplot(2, 3, 2)
    for size, r in results.items():
        c = '#3498db' if '0.5' in size else '#e74c3c'
        ax2.scatter(r['avg_T'], r['avg_P1'], c=range(len(r['avg_T'])),
                   cmap='viridis', s=30, alpha=0.7)
        ax2.plot(r['avg_T'], r['avg_P1'], '-', color=c, alpha=0.3, label=size)
    ax2.set_xlabel('T'); ax2.set_ylabel('P1')
    ax2.set_title('(b) T-P1 Phase Plane')
    ax2.legend(fontsize=8)

    # (c) d2T/dl2 (phase boundary detection)
    ax3 = fig.add_subplot(2, 3, 3)
    for size, r in results.items():
        c = '#3498db' if '0.5' in size else '#e74c3c'
        ax3.plot(range(len(r['d2T'])), r['d2T'], '-', color=c, lw=2, label=size)
        for pb in r['phase_boundaries']:
            ax3.axvline(x=pb, color=c, ls='--', alpha=0.3)
    ax3.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax3.set_xlabel('Layer'); ax3.set_ylabel("d2T/dl2")
    ax3.set_title('(c) Curvature (Phase Boundaries)')
    ax3.legend(fontsize=8)

    # (d) Susceptibility chi
    ax4 = fig.add_subplot(2, 3, 4)
    for size, r in results.items():
        c = '#3498db' if '0.5' in size else '#e74c3c'
        ax4.plot(range(len(r['chi'])), r['chi'], '-', color=c, lw=2, label=size)
    ax4.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax4.set_xlabel('Layer'); ax4.set_ylabel('chi = dP1/dT')
    ax4.set_title('(d) Susceptibility')
    ax4.legend(fontsize=8)

    # (e) Individual trajectories
    ax5 = fig.add_subplot(2, 3, 5)
    for ti, traj in enumerate(r15['trajectories']):
        T_traj = [t[0] for t in traj]
        P1_traj = [t[2] for t in traj]
        ax5.plot(T_traj, P1_traj, '-', alpha=0.5, lw=1)
    ax5.set_xlabel('T'); ax5.set_ylabel('P1')
    ax5.set_title('(e) Individual Trajectories (1.5B)')

    # (f) Summary
    ax6 = fig.add_subplot(2, 3, 6)
    summary = "PHASE DIAGRAM\n\n"
    for size, r in results.items():
        summary += f"{size} ({r['n_layers']}L):\n"
        summary += f"  Boundaries: {r['phase_boundaries']}\n"
        for ce in r['critical_exponents']:
            summary += f"  beta(L{ce['layer']})={ce['beta']:.3f}\n"
        summary += "\n"
    ax6.text(0.5, 0.5, summary, ha='center', va='center',
             transform=ax6.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             family='monospace')
    ax6.axis('off')
    ax6.set_title('(f) Summary')

    fig.suptitle("Phase 236: Thermodynamic Phase Diagram", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase236_phase_diagram')
    plt.close()
    save_results('phase236_phase_diagram', {'experiment': 'Phase Diagram', 'results': results})


if __name__ == '__main__':
    main()
