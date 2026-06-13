# -*- coding: utf-8 -*-
"""
Phase 105: Specific Heat Lambda Transition
The specific heat C = dU/dT should show a peak/divergence at a 2nd order
phase transition (the "lambda transition"). Measure C at each layer.
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
    "The human genome encodes three billion base pairs",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Climate change affects global ecosystems",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
]


def main():
    print("=" * 70)
    print("Phase 105: Specific Heat Lambda Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect U, T at each layer for each prompt
    all_Us = []
    all_Ts = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        Us = []
        Ts = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()
            Us.append(U)

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0.0
            Ts.append(T)

        all_Us.append(Us)
        all_Ts.append(Ts)

    avg_U = np.mean(all_Us, axis=0)
    avg_T = np.mean(all_Ts, axis=0)

    # Specific heat: C = dU/dT at each layer transition
    # Use finite differences
    C_vals = []
    for i in range(1, n_layers):
        dU = avg_U[i] - avg_U[i-1]
        dT = avg_T[i] - avg_T[i-1]
        C = dU / dT if abs(dT) > 1e-6 else 0.0
        C_vals.append(C)

    # Also compute C from variance: C = Var(U) / T^2
    C_fluct = []
    for li in range(n_layers):
        var_U = np.var([r[li] for r in all_Us])
        T = avg_T[li]
        C_fluct.append(var_U / (T**2 + 1e-10))

    layers_c = np.arange(1, n_layers)

    # Find lambda peak
    C_arr = np.array(C_vals)
    # Look for peak in absolute value
    abs_C = np.abs(C_arr)
    peak_idx = np.argmax(abs_C[3:]) + 3  # skip first 3
    peak_layer = peak_idx + 1  # because C is between layers

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Specific heat profile
    colors_c = ['#c0392b' if c > 0 else '#2980b9' for c in C_vals]
    axes[0,0].bar(layers_c, C_vals, color=colors_c, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--',
                      label=f'$L_0={L0:.0f}$')
    axes[0,0].axvline(x=peak_layer, color='#27ae60', linewidth=2, linestyle=':',
                      label=f'C peak: L{peak_layer}')
    axes[0,0].set_xlabel('Layer Transition')
    axes[0,0].set_ylabel('$C = dU/dT$')
    axes[0,0].set_title('(a) Specific Heat Profile')
    axes[0,0].legend(fontsize=8)

    # (b) Fluctuation-based C
    axes[0,1].plot(range(n_layers), C_fluct, 'o-', color='#8e44ad', markersize=4)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    peak_fluct = np.argmax(C_fluct[3:]) + 3
    axes[0,1].axvline(x=peak_fluct, color='#27ae60', linewidth=2, linestyle=':',
                      label=f'Peak: L{peak_fluct}')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$C_{fluct} = Var(U)/T^2$')
    axes[0,1].set_title('(b) Fluctuation Specific Heat')
    axes[0,1].legend(fontsize=8)

    # (c) U(T) phase trajectory
    axes[0,2].scatter(avg_T, avg_U, c=range(n_layers), cmap='coolwarm', s=60,
                      edgecolors='black', zorder=5)
    # Connect with lines
    axes[0,2].plot(avg_T, avg_U, '-', color='gray', alpha=0.3)
    for i in [0, 5, int(L0), n_layers-1]:
        if i < n_layers:
            axes[0,2].annotate(f'L{i}', (avg_T[i], avg_U[i]),
                              textcoords="offset points", xytext=(5,5), fontsize=7)
    cb = plt.colorbar(axes[0,2].collections[0], ax=axes[0,2], shrink=0.7)
    cb.set_label('Layer')
    axes[0,2].set_xlabel('$T$ (entropy)')
    axes[0,2].set_ylabel('$U$ (energy)')
    axes[0,2].set_title('(c) $U$-$T$ Phase Trajectory')

    # (d) |C| profile (lambda shape?)
    axes[1,0].plot(layers_c, abs_C, 'o-', color='#c0392b', markersize=5, linewidth=2)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[1,0].set_xlabel('Layer Transition')
    axes[1,0].set_ylabel('$|C|$')
    axes[1,0].set_title(f'(d) |C| Profile (peak L{peak_layer})')
    axes[1,0].legend()

    # (e) dT/dL profile
    dT_dL = np.gradient(avg_T)
    axes[1,1].plot(range(n_layers), dT_dL, 'o-', color='#27ae60', markersize=4)
    axes[1,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].axhline(y=0, color='black', linewidth=0.5)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('$dT/dL$')
    axes[1,1].set_title('(e) Temperature Gradient')

    # (f) Summary
    # Check if C peak is near L0
    is_lambda = abs(peak_layer - L0) <= 3
    summary = (
        f"Specific Heat Analysis\n\n"
        f"C peak (dU/dT): L{peak_layer}\n"
        f"C peak (fluct): L{peak_fluct}\n"
        f"Transition L0: {L0:.0f}\n\n"
        f"Lambda transition: {'YES' if is_lambda else 'NO'}\n"
        f"(peak within 3 layers of L0)\n\n"
        f"Max |C| = {abs_C.max():.2f}\n"
        f"Max C_fluct = {max(C_fluct):.2f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 105: Specific Heat '
                 f'(C peak at L{peak_layer}, {"LAMBDA" if is_lambda else "NO LAMBDA"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase105_specific_heat')
    plt.close()

    print(f"\n{'='*70}")
    print(f"C (dU/dT) peak: L{peak_layer}")
    print(f"C (fluctuation) peak: L{peak_fluct}")
    print(f"Lambda transition: {'YES' if is_lambda else 'NO'}")
    print(f"{'='*70}")

    save_results('phase105_specific_heat', {
        'experiment': 'Specific Heat Lambda Transition',
        'C_dUdT': [float(c) for c in C_vals],
        'C_fluct': [float(c) for c in C_fluct],
        'U_profile': [float(u) for u in avg_U],
        'T_profile': [float(t) for t in avg_T],
        'summary': {
            'C_peak_layer': int(peak_layer),
            'C_fluct_peak': int(peak_fluct),
            'is_lambda': is_lambda,
            'max_C': float(abs_C.max()),
        }
    })


if __name__ == '__main__':
    main()
