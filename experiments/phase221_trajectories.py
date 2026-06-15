# -*- coding: utf-8 -*-
"""
Phase 221: Stochastic Trajectory Analysis
============================================
Analyze thermodynamic trajectories at single-token level.
Each prompt traces a unique path through (U, T, S) state space.
Test: Do trajectories converge to a universal attractor?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
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


def trace_trajectories(model, tok, device, model_name):
    """Trace thermodynamic trajectories for each prompt."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    trajectories = {}  # prompt -> list of (U, T, S, P1) states

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        states = []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1 = probs.max().item()
            T = S  # Temperature
            states.append([U, T, S, P1])
        trajectories[pi] = states

    n = min(len(s) for s in trajectories.values())

    # Convergence analysis: dispersion at each layer
    dispersions_UT = []
    dispersions_SP1 = []
    for l in range(n):
        ut_points = np.array([[trajectories[p][l][0], trajectories[p][l][1]] for p in range(len(PROMPTS))])
        sp1_points = np.array([[trajectories[p][l][2], trajectories[p][l][3]] for p in range(len(PROMPTS))])
        if len(ut_points) > 1:
            dispersions_UT.append(float(np.mean(pdist(ut_points))))
            dispersions_SP1.append(float(np.mean(pdist(sp1_points))))
        else:
            dispersions_UT.append(0)
            dispersions_SP1.append(0)

    # Convergence ratio: dispersion at last layer / dispersion at first layer
    conv_UT = dispersions_UT[-1] / dispersions_UT[0] if dispersions_UT[0] > 0 else 1
    conv_SP1 = dispersions_SP1[-1] / dispersions_SP1[0] if dispersions_SP1[0] > 0 else 1

    # Lyapunov-like: log dispersion slope
    log_disp = np.log(np.array(dispersions_UT) + 1e-10)
    try:
        from scipy.stats import linregress
        slope, _, r, p, _ = linregress(range(len(log_disp)), log_disp)
    except Exception:
        slope, r, p = 0, 0, 1

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_states': n,
        'trajectories': {str(k): v for k, v in trajectories.items()},
        'dispersions_UT': dispersions_UT,
        'dispersions_SP1': dispersions_SP1,
        'convergence_UT': conv_UT,
        'convergence_SP1': conv_SP1,
        'lyapunov_slope': float(slope),
        'lyapunov_r': float(r),
        'lyapunov_p': float(p),
    }


def main():
    print("=" * 70)
    print("Phase 221: Stochastic Trajectory Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = trace_trajectories(model, tok, device, size)
        results[size] = r
        print(f"  Convergence UT={r['convergence_UT']:.4f}, SP1={r['convergence_SP1']:.4f}")
        print(f"  Lyapunov slope={r['lyapunov_slope']:.4f} (r={r['lyapunov_r']:.3f})")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) T-U trajectory (0.5B)
    r05 = results['0.5B']
    for pi in range(min(10, len(r05['trajectories']))):
        traj = r05['trajectories'][str(pi)]
        us = [s[0] for s in traj]
        ts = [s[1] for s in traj]
        axes[0, 0].plot(us, ts, '-', alpha=0.4, lw=1)
    axes[0, 0].set_xlabel('Internal Energy U')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Trajectories in U-T (0.5B)')

    # (b) T-U trajectory (1.5B)
    r15 = results['1.5B']
    for pi in range(min(10, len(r15['trajectories']))):
        traj = r15['trajectories'][str(pi)]
        us = [s[0] for s in traj]
        ts = [s[1] for s in traj]
        axes[0, 1].plot(us, ts, '-', alpha=0.4, lw=1)
    axes[0, 1].set_xlabel('Internal Energy U')
    axes[0, 1].set_ylabel('Temperature T')
    axes[0, 1].set_title('(b) Trajectories in U-T (1.5B)')

    # (c) Dispersion vs layer
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['dispersions_UT'])), r['dispersions_UT'],
                       '-', color=colors[size], lw=2, label=f'{size} U-T')
        axes[0, 2].plot(range(len(r['dispersions_SP1'])), r['dispersions_SP1'],
                       '--', color=colors[size], lw=1.5, alpha=0.6, label=f'{size} S-P1')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Mean Pairwise Distance')
    axes[0, 2].set_title('(c) Trajectory Dispersion')
    axes[0, 2].legend(fontsize=7)

    # (d) P1 trajectories (0.5B)
    for pi in range(min(10, len(r05['trajectories']))):
        traj = r05['trajectories'][str(pi)]
        p1s = [s[3] for s in traj]
        axes[1, 0].plot(range(len(p1s)), p1s, '-', alpha=0.4, lw=1)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('P1')
    axes[1, 0].set_title('(d) P1 Trajectories (0.5B)')

    # (e) Log dispersion (Lyapunov)
    for size, r in results.items():
        log_d = np.log(np.array(r['dispersions_UT']) + 1e-10)
        axes[1, 1].plot(range(len(log_d)), log_d, '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('log(dispersion)')
    axes[1, 1].set_title('(e) Log Dispersion (Lyapunov)')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Trajectory Analysis\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Conv(UT)  = {r['convergence_UT']:.4f}\n"
        summary += f"  Conv(SP1) = {r['convergence_SP1']:.4f}\n"
        summary += f"  Lyapunov  = {r['lyapunov_slope']:.4f}\n\n"
    converging = all(r['convergence_UT'] < 1 for r in results.values())
    summary += f"Attractor: {'YES' if converging else 'NO'}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 221: Stochastic Trajectory Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase221_trajectories')
    plt.close()
    save_results('phase221_trajectories', {'experiment': 'Stochastic Trajectories', 'results': results})


if __name__ == '__main__':
    main()
