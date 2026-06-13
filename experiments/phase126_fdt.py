# -*- coding: utf-8 -*-
"""
Phase 126: Fluctuation-Dissipation Theorem (FDT)
In equilibrium, fluctuations and response are related:
  chi = <delta_eta^2> / kT
If FDT holds, chi_measured should match chi_FDT.
Violation of FDT = system is out of equilibrium.
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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "The discovery of antibiotics revolutionized medicine",
]


def main():
    print("=" * 70)
    print("Phase 126: Fluctuation-Dissipation Theorem")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # At each layer, measure:
    # 1. eta fluctuation variance (across prompts)
    # 2. kT (from Boltzmann fit)
    # 3. Response chi (numerical derivative d_eta/d_h)

    all_etas = [[] for _ in range(n_layers)]
    all_kT = [[] for _ in range(n_layers)]
    all_S = [[] for _ in range(n_layers)]

    for prompt in PROMPTS:
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
            all_S[li].append(S)

            # kT from top-k distribution
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

        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                if T_hot > 0.01:
                    all_etas[li].append(1.0 - T_cold / T_hot)
                else:
                    all_etas[li].append(0.0)
            else:
                all_etas[li].append(0.0)

    # Compute FDT quantities
    avg_kT = [np.mean(v) if v else 1 for v in all_kT]
    var_eta = [np.var(v) if v else 0 for v in all_etas]
    avg_eta = [np.mean(v) if v else 0 for v in all_etas]

    # FDT prediction: chi_FDT = var_eta / kT
    chi_fdt = [v / (kT + 1e-10) for v, kT in zip(var_eta, avg_kT)]

    # Measured susceptibility: chi = d<eta>/dL (numerical)
    chi_measured = np.gradient(avg_eta)

    # FDT ratio: R = chi_measured / chi_FDT
    fdt_ratio = [cm / (cf + 1e-10) for cm, cf in zip(chi_measured, chi_fdt)]

    layers = np.arange(n_layers)

    # FDT violation analysis
    pre_fdt = np.mean([abs(r - 1) for r, l in zip(fdt_ratio, layers) if l < L0 and l > 4])
    post_fdt = np.mean([abs(r - 1) for r, l in zip(fdt_ratio, layers) if l >= L0])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Eta variance
    axes[0,0].plot(layers, var_eta, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Var($\\eta$)')
    axes[0,0].set_title('(a) Order Parameter Fluctuations')
    axes[0,0].legend()

    # (b) kT profile
    axes[0,1].plot(layers, avg_kT, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$kT$')
    axes[0,1].set_title('(b) Temperature')

    # (c) chi comparison
    axes[0,2].plot(layers, chi_fdt, 'o-', color='#c0392b', markersize=3, label='$\\chi_{FDT} = Var/kT$')
    axes[0,2].plot(layers, np.abs(chi_measured), 's-', color='#2980b9', markersize=3, label='$|\\chi_{meas}|$')
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Susceptibility')
    axes[0,2].set_title('(c) FDT Comparison')
    axes[0,2].legend(fontsize=8)

    # (d) FDT ratio
    axes[1,0].plot(layers[4:], fdt_ratio[4:], 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axhline(y=1, color='black', linestyle='--', label='FDT satisfied')
    axes[1,0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$\\chi_{meas} / \\chi_{FDT}$')
    axes[1,0].set_title('(d) FDT Ratio (1 = equilibrium)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].set_ylim(-5, 5)

    # (e) FDT violation magnitude
    fdt_viol = [abs(r - 1) for r in fdt_ratio]
    colors_v = ['#27ae60' if v < 1 else '#f39c12' if v < 3 else '#c0392b' for v in fdt_viol]
    axes[1,1].bar(layers[4:], fdt_viol[4:], color=colors_v[4:], alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('$|R - 1|$ (FDT violation)')
    axes[1,1].set_title('(e) FDT Violation')

    # (f) Summary
    summary = (
        f"Fluctuation-Dissipation Theorem\n\n"
        f"FDT violation pre-L0: {pre_fdt:.3f}\n"
        f"FDT violation post-L0: {post_fdt:.3f}\n\n"
        f"Pre-transition: {'NEAR-EQUIL' if pre_fdt < 1 else 'NON-EQUIL'}\n"
        f"Post-transition: {'NEAR-EQUIL' if post_fdt < 1 else 'NON-EQUIL'}\n\n"
        f"FDT better {'BEFORE' if pre_fdt < post_fdt else 'AFTER'} transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 126: FDT (pre={pre_fdt:.2f}, post={post_fdt:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase126_fdt')
    plt.close()

    print(f"\n{'='*70}")
    print(f"FDT violation pre: {pre_fdt:.3f}, post: {post_fdt:.3f}")
    print(f"{'='*70}")

    save_results('phase126_fdt', {
        'experiment': 'Fluctuation-Dissipation Theorem',
        'var_eta': [float(v) for v in var_eta],
        'avg_kT': [float(v) for v in avg_kT],
        'chi_fdt': [float(v) for v in chi_fdt],
        'chi_measured': [float(v) for v in chi_measured],
        'fdt_ratio': [float(v) for v in fdt_ratio],
        'summary': {
            'pre_fdt_violation': float(pre_fdt),
            'post_fdt_violation': float(post_fdt),
        }
    })


if __name__ == '__main__':
    main()
