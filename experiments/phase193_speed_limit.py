# -*- coding: utf-8 -*-
"""
Phase 193: Thermodynamic Speed Limit (Mandelstam-Tamm Bound)
==============================================================
Quantum speed limit: the minimum time to evolve between orthogonal
states is pi*hbar/(2*Delta_E).

Transformer analog: the minimum number of layers needed to go from
p_0 to p_L is constrained by the energy resources available.

Speed limit ratio = (Bures angle) / (sum of energy uncertainties)
If ratio ~ 1, the model is at the speed limit.
If ratio << 1, the model has unused capacity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
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


def bures_angle(p, q):
    """Bures angle: arccos(sum(sqrt(p*q)))."""
    fidelity = torch.sum(torch.sqrt(p * q + 1e-20)).item()
    fidelity = min(max(fidelity, 0), 1)
    return np.arccos(fidelity)


def main():
    print("=" * 70)
    print("Phase 193: Thermodynamic Speed Limit")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_speed_ratio = []
    all_bures = []         # Total Bures angle (geodesic distance)
    all_resources = []     # Total energy resources
    all_per_layer_speed = []  # Speed per layer
    all_per_layer_dE = []    # Energy uncertainty per layer

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        probs_all = []
        U_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()
            U_vals.append(U if not np.isnan(U) else 0)

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            probs_all.append(probs)

        # Total Bures angle (direct distance in probability space)
        total_bures = bures_angle(probs_all[0], probs_all[-1])
        all_bures.append(total_bures)

        # Per-layer: Bures step and energy uncertainty
        bures_steps = []
        dE_vals = []
        for i in range(n_layers - 1):
            bs = bures_angle(probs_all[i], probs_all[i + 1])
            bures_steps.append(bs)

            # Energy uncertainty (std of hidden state components)
            h = out.hidden_states[i][0, -1, :].float()
            dE = h.std().item()
            dE_vals.append(dE if not np.isnan(dE) else 0)

        all_per_layer_speed.append(bures_steps)
        all_per_layer_dE.append(dE_vals)

        # Speed limit: total_bures <= pi/(2) * sum(dE_steps) (normalized)
        # Mandelstam-Tamm: tau >= pi * hbar / (2 * Delta_E)
        # In our units: n_layers >= pi / (2 * mean(Delta_E)) * total_bures
        path_length = sum(bures_steps)
        resource = np.pi / 2 * np.mean(dE_vals) * (n_layers - 1)

        speed_ratio = total_bures / (resource + 1e-10)
        all_speed_ratio.append(speed_ratio)
        all_resources.append(resource)

    speed_mean = np.mean(all_per_layer_speed, axis=0)
    dE_mean = np.mean(all_per_layer_dE, axis=0)
    bures_mean = np.mean(all_bures)
    ratio_mean = np.mean(all_speed_ratio)
    ratio_std = np.std(all_speed_ratio)

    layers_t = np.arange(n_layers - 1) + 0.5

    # Cumulative speed vs cumulative resource
    cum_speed = np.cumsum(speed_mean)
    cum_resource = np.cumsum(np.pi / 2 * dE_mean)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Per-layer speed (Bures step)
    axes[0, 0].plot(layers_t, speed_mean, 'o-', color='#e74c3c', markersize=4, linewidth=2,
                    label='Bures step')
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Bures Angle per Step')
    axes[0, 0].set_title('(a) Computation Speed per Layer')
    axes[0, 0].legend(fontsize=8)

    # (b) Energy uncertainty
    axes[0, 1].plot(layers_t, dE_mean, 's-', color='#3498db', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$\\Delta E$ (energy uncertainty)')
    axes[0, 1].set_title('(b) Energy Resources per Layer')

    # (c) Cumulative path vs cumulative resource
    axes[0, 2].plot(layers_t, cum_speed, 'o-', color='#e74c3c', markersize=3, linewidth=2,
                    label='Path (Bures)')
    axes[0, 2].plot(layers_t, cum_resource, 's-', color='#3498db', markersize=3, linewidth=2,
                    label='Resource ($\\pi/2 \\cdot \\Delta E$)')
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Cumulative Value')
    axes[0, 2].set_title('(c) Path vs Resource Budget')
    axes[0, 2].legend(fontsize=8)

    # (d) Speed/resource ratio per layer
    ratio_per_layer = speed_mean / (np.pi / 2 * dE_mean + 1e-10)
    axes[1, 0].plot(layers_t, ratio_per_layer, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1, 0].axhline(y=1, color='black', linestyle='--', linewidth=2, label='Speed limit')
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Speed / Resource')
    axes[1, 0].set_title('(d) Speed Limit Saturation')
    axes[1, 0].legend(fontsize=8)

    # (e) Distribution of speed ratios
    axes[1, 1].hist(all_speed_ratio, bins=12, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=ratio_mean, color='#e74c3c', linewidth=2, linestyle='--',
                        label=f'Mean={ratio_mean:.4f}')
    axes[1, 1].set_xlabel('Speed Limit Ratio')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('(e) Speed Ratio Distribution')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    near_limit = sum(1 for r in ratio_per_layer if r > 0.5)
    summary = (
        f"Thermodynamic Speed Limit\n(Mandelstam-Tamm analog)\n\n"
        f"Overall speed ratio:\n"
        f"  {ratio_mean:.4f} +/- {ratio_std:.4f}\n\n"
        f"Mean Bures angle: {bures_mean:.4f}\n"
        f"Mean resource: {np.mean(all_resources):.2f}\n\n"
        f"Layers near limit (>50%%): {near_limit}/{n_layers-1}\n\n"
        f"Model is {'AT' if ratio_mean > 0.3 else 'FAR FROM'}\n"
        f"the speed limit"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 193: Thermodynamic Speed Limit', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase193_speed_limit')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Speed ratio: {ratio_mean:.4f} +/- {ratio_std:.4f}")
    print(f"Bures angle: {bures_mean:.4f}")
    print(f"Layers near limit: {near_limit}/{n_layers-1}")
    print(f"{'=' * 70}")

    save_results('phase193_speed_limit', {
        'experiment': 'Thermodynamic Speed Limit',
        'speed_mean': [float(x) for x in speed_mean],
        'dE_mean': [float(x) for x in dE_mean],
        'summary': {
            'speed_ratio_mean': float(ratio_mean),
            'speed_ratio_std': float(ratio_std),
            'bures_angle_mean': float(bures_mean),
            'near_limit_layers': int(near_limit),
        }
    })


if __name__ == '__main__':
    main()
