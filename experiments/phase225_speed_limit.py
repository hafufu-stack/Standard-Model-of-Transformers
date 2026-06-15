# -*- coding: utf-8 -*-
"""
Phase 225: Thermodynamic Speed Limit
========================================
Test the quantum/thermodynamic speed limit:
  ds/dl >= 2 * arccos(F(rho_l, rho_{l+1}))
where F is the fidelity and ds is the geodesic distance.

Mandelstam-Tamm: dE * dt >= hbar/2
Here: dU * dl >= C (some constant) for information processing.
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


def measure_speed_limit(model, tok, device, model_name):
    """Measure thermodynamic speed and speed limit at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    K = 200

    all_speed = []      # Geodesic speed ds/dl
    all_energy_var = []  # Energy variance (drives speed limit)
    all_dU = []         # Energy change per layer
    all_dS = []         # Entropy change per layer
    all_fidelity = []   # Fidelity between consecutive layers

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        speed_l, evar_l, dU_l, dS_l, fid_l = [], [], [], [], []
        prev_probs_topk = None
        prev_U = None
        prev_S = None

        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()

            # Energy variance: Var(h_i^2)
            h_sq = h ** 2
            evar = float(h_sq.var().item())
            evar_l.append(evar)

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()

            topk = probs.topk(K)
            p_topk = topk.values.cpu().numpy()
            p_topk = p_topk / (p_topk.sum() + 1e-10)

            if prev_probs_topk is not None:
                # Fidelity: F = sum(sqrt(p*q))
                fid = float(np.sum(np.sqrt(p_topk * prev_probs_topk + 1e-20)))
                fid_l.append(min(fid, 1.0))
                # Geodesic speed: ds = 2*arccos(F)
                speed = 2 * np.arccos(min(fid, 1.0))
                speed_l.append(float(speed))
                dU_l.append(U - prev_U)
                dS_l.append(S - prev_S if not np.isnan(S - prev_S) else 0)

            prev_probs_topk = p_topk.copy()
            prev_U = U
            prev_S = S

        all_speed.append(speed_l)
        all_energy_var.append(evar_l)
        all_dU.append(dU_l)
        all_dS.append(dS_l)
        all_fidelity.append(fid_l)

    n_trans = min(len(s) for s in all_speed)
    n_state = min(len(e) for e in all_energy_var)
    avg_t = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n_trans)]
    avg_s = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n_state)]

    mean_speed = avg_t(all_speed)
    mean_evar = avg_s(all_energy_var)
    mean_dU = avg_t(all_dU)
    mean_dS = avg_t(all_dS)
    mean_fid = avg_t(all_fidelity)

    # Speed limit: ds/dl <= 2 * sqrt(Var(H)) (analog of Mandelstam-Tamm)
    # Use energy variance at each layer as the "bound"
    speed_limit = [2 * np.sqrt(mean_evar[i+1]) if i+1 < len(mean_evar) else 0 for i in range(n_trans)]

    # Saturation ratio: actual_speed / speed_limit
    saturation = [mean_speed[i] / speed_limit[i] if speed_limit[i] > 0 else 0 for i in range(n_trans)]

    # Efficiency: ds / |dU| (information gained per unit energy change)
    info_efficiency = [mean_speed[i] / (abs(mean_dU[i]) + 1e-10) for i in range(n_trans)]

    # Total speed
    total_speed = sum(mean_speed)
    total_limit = sum(speed_limit)

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_speed': mean_speed,
        'speed_limit': speed_limit,
        'saturation': saturation,
        'mean_fidelity': mean_fid,
        'mean_dU': mean_dU,
        'mean_dS': mean_dS,
        'info_efficiency': info_efficiency,
        'total_speed': total_speed,
        'total_limit': total_limit,
        'global_saturation': total_speed / total_limit if total_limit > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Phase 225: Thermodynamic Speed Limit")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_speed_limit(model, tok, device, size)
        results[size] = r
        print(f"  Total speed={r['total_speed']:.4f}, limit={r['total_limit']:.4f}")
        print(f"  Global saturation={r['global_saturation']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        c = colors[size]
        n = len(r['mean_speed'])
        axes[0, 0].plot(range(n), r['mean_speed'], '-', color=c, lw=2, label=f'{size} actual')
        axes[0, 0].plot(range(n), r['speed_limit'], '--', color=c, lw=1.5, alpha=0.5, label=f'{size} limit')
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Speed')
    axes[0, 0].set_title('(a) Speed vs Speed Limit')
    axes[0, 0].legend(fontsize=7)

    for size, r in results.items():
        axes[0, 1].plot(range(len(r['saturation'])), r['saturation'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(y=1, color='red', ls='--', alpha=0.5, label='Limit')
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Speed/Limit')
    axes[0, 1].set_title('(b) Speed Limit Saturation')
    axes[0, 1].legend(fontsize=8)

    for size, r in results.items():
        axes[0, 2].plot(range(len(r['mean_fidelity'])), r['mean_fidelity'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Fidelity')
    axes[0, 2].set_title('(c) Layer Fidelity')
    axes[0, 2].legend(fontsize=8)

    for size, r in results.items():
        axes[1, 0].plot(range(len(r['info_efficiency'])), r['info_efficiency'],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('ds/|dU|')
    axes[1, 0].set_title('(d) Information Efficiency')
    axes[1, 0].legend(fontsize=8)

    for size, r in results.items():
        axes[1, 1].scatter(r['mean_dU'], r['mean_speed'], c=range(len(r['mean_speed'])),
                          cmap='viridis', s=30, alpha=0.7)
    axes[1, 1].set_xlabel('dU'); axes[1, 1].set_ylabel('Geodesic Speed')
    axes[1, 1].set_title('(e) Speed vs Energy Change')

    summary = "Speed Limit\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Speed     = {r['total_speed']:.4f}\n"
        summary += f"  Limit     = {r['total_limit']:.4f}\n"
        summary += f"  Saturate  = {r['global_saturation']:.4f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 225: Thermodynamic Speed Limit", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase225_speed_limit')
    plt.close()
    save_results('phase225_speed_limit', {'experiment': 'Speed Limit', 'results': results})


if __name__ == '__main__':
    main()
