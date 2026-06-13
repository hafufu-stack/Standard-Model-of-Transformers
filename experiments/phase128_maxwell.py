# -*- coding: utf-8 -*-
"""
Phase 128: Maxwell Relation Test
In thermodynamics, Maxwell relations connect partial derivatives:
  (dS/dV)_T = (dP/dT)_V
If our thermodynamic framework is real, analogous relations should hold.
Test: d(entropy)/d(eta) at fixed kT == d(kT)/d(layer) at fixed eta
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "The discovery of antibiotics revolutionized medicine",
]


def main():
    print("=" * 70)
    print("Phase 128: Maxwell Relations Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect state variables at each layer for each prompt
    data_points = []  # (layer, prompt_idx, S, kT, eta)

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
            if np.isnan(S): S = 0
            T_vals.append(S)

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
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0

            data_points.append({
                'layer': li, 'prompt': pi,
                'S': float(S), 'kT': float(kT), 'eta': float(eta)
            })

    # Compute Maxwell relation: dS/d(eta) vs d(kT)/dL
    # Group by layer
    by_layer = {}
    for dp in data_points:
        li = dp['layer']
        if li not in by_layer:
            by_layer[li] = []
        by_layer[li].append(dp)

    layers_valid = sorted([li for li in by_layer if li >= 4])

    # dS/d(eta) at each layer (across prompts)
    dS_deta = []
    for li in layers_valid:
        pts = by_layer[li]
        etas = [p['eta'] for p in pts]
        Ss = [p['S'] for p in pts]
        if np.std(etas) > 1e-5:
            slope, _, r, _, _ = sp_stats.linregress(etas, Ss)
            dS_deta.append(float(slope))
        else:
            dS_deta.append(0)

    # d(kT)/dL at each layer
    avg_kT = [np.mean([p['kT'] for p in by_layer[li]]) for li in layers_valid]
    dkT_dL = np.gradient(avg_kT)

    # Maxwell test: are they correlated?
    if len(dS_deta) > 5 and len(dkT_dL) > 5:
        r_maxwell, p_maxwell = sp_stats.pearsonr(dS_deta, dkT_dL)
    else:
        r_maxwell, p_maxwell = 0, 1

    # Compute second Maxwell: d(kT)/d(eta) vs dS/dL
    avg_S = [np.mean([p['S'] for p in by_layer[li]]) for li in layers_valid]
    dS_dL = np.gradient(avg_S)

    dkT_deta = []
    for li in layers_valid:
        pts = by_layer[li]
        etas = [p['eta'] for p in pts]
        kTs = [p['kT'] for p in pts]
        if np.std(etas) > 1e-5:
            slope, _, _, _, _ = sp_stats.linregress(etas, kTs)
            dkT_deta.append(float(slope))
        else:
            dkT_deta.append(0)

    r_maxwell2, p_maxwell2 = sp_stats.pearsonr(dkT_deta, dS_dL) if len(dkT_deta) > 5 else (0, 1)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) dS/deta and dkT/dL
    ax1 = axes[0,0]
    ax1.plot(layers_valid, dS_deta, 'o-', color='#c0392b', markersize=3, label='$dS/d\\eta$')
    ax2 = ax1.twinx()
    ax2.plot(layers_valid, dkT_dL, 's-', color='#2980b9', markersize=3, label='$dkT/dL$')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('$dS/d\\eta$', color='#c0392b')
    ax2.set_ylabel('$dkT/dL$', color='#2980b9')
    ax1.set_title(f'(a) Maxwell Rel. 1 ($r={r_maxwell:.3f}$)')
    ax1.axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')

    # (b) Scatter: dS/deta vs dkT/dL
    axes[0,1].scatter(dS_deta, dkT_dL, c=layers_valid, cmap='coolwarm',
                      s=60, edgecolors='black')
    axes[0,1].set_xlabel('$dS/d\\eta$')
    axes[0,1].set_ylabel('$dkT/dL$')
    axes[0,1].set_title(f'(b) Maxwell Test ($r={r_maxwell:.3f}$, $p={p_maxwell:.3f}$)')

    # (c) Second Maxwell relation
    ax3 = axes[0,2]
    ax3.plot(layers_valid, dkT_deta, 'o-', color='#27ae60', markersize=3, label='$dkT/d\\eta$')
    ax4 = ax3.twinx()
    ax4.plot(layers_valid, dS_dL, 's-', color='#8e44ad', markersize=3, label='$dS/dL$')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('$dkT/d\\eta$', color='#27ae60')
    ax4.set_ylabel('$dS/dL$', color='#8e44ad')
    ax3.set_title(f'(c) Maxwell Rel. 2 ($r={r_maxwell2:.3f}$)')
    ax3.axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')

    # (d) Phase portrait (S, kT)
    axes[1,0].scatter(avg_S, avg_kT, c=layers_valid, cmap='coolwarm',
                      s=80, edgecolors='black')
    for i, li in enumerate(layers_valid):
        if i % 3 == 0:
            axes[1,0].annotate(f'{li}', (avg_S[i], avg_kT[i]), fontsize=6)
    axes[1,0].set_xlabel('$S$')
    axes[1,0].set_ylabel('$kT$')
    axes[1,0].set_title('(d) Phase Portrait')

    # (e) State trajectory
    avg_eta = [np.mean([p['eta'] for p in by_layer[li]]) for li in layers_valid]
    axes[1,1].scatter(avg_eta, avg_S, c=layers_valid, cmap='coolwarm',
                      s=80, edgecolors='black')
    for i, li in enumerate(layers_valid):
        if i % 3 == 0:
            axes[1,1].annotate(f'{li}', (avg_eta[i], avg_S[i]), fontsize=6)
    axes[1,1].set_xlabel('$\\eta$')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) State Trajectory')

    # (f) Summary
    summary = (
        f"Maxwell Relations Test\n\n"
        f"Relation 1: dS/deta ~ dkT/dL\n"
        f"  r = {r_maxwell:.3f} (p = {p_maxwell:.3f})\n"
        f"  {'HOLDS' if abs(r_maxwell) > 0.5 else 'VIOLATED'}\n\n"
        f"Relation 2: dkT/deta ~ dS/dL\n"
        f"  r = {r_maxwell2:.3f} (p = {p_maxwell2:.3f})\n"
        f"  {'HOLDS' if abs(r_maxwell2) > 0.5 else 'VIOLATED'}\n\n"
        f"Thermodynamic consistency:\n"
        f"{'CONSISTENT' if abs(r_maxwell) > 0.5 or abs(r_maxwell2) > 0.5 else 'PARTIAL'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 128: Maxwell Relations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase128_maxwell')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Maxwell 1: r={r_maxwell:.3f} (p={p_maxwell:.3f})")
    print(f"Maxwell 2: r={r_maxwell2:.3f} (p={p_maxwell2:.3f})")
    print(f"{'='*70}")

    save_results('phase128_maxwell', {
        'experiment': 'Maxwell Relations',
        'summary': {
            'maxwell1_r': float(r_maxwell),
            'maxwell1_p': float(p_maxwell),
            'maxwell2_r': float(r_maxwell2),
            'maxwell2_p': float(p_maxwell2),
        }
    })


if __name__ == '__main__':
    main()
