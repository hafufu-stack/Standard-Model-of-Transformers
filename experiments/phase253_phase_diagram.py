# -*- coding: utf-8 -*-
"""
Phase 253: Thermodynamic Phase Diagram (T-sigma plane)
========================================================
Map the complete phase diagram in the (T, sigma) plane
where sigma = noise injection level.
Identify phase boundaries and critical points.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, make_safe_noise_hook, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "Machine learning discovers hidden patterns",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "The brain contains billions of neurons",
]

NOISE_LEVELS = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
INJECTION_LAYERS = [0, 5, 10, 15, 20, 25]  # Will be capped to actual n_layers


def phase_diagram(model, tok, device, model_name):
    """Map T, P1 as function of noise level and injection layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    
    # Cap injection layers
    inject_layers = [l for l in INJECTION_LAYERS if l < n_layers]
    if not inject_layers:
        inject_layers = list(range(0, n_layers, max(1, n_layers // 5)))

    phase_data = []  # (sigma, inject_layer, T_final, P1_final)

    for sigma in NOISE_LEVELS:
        for inject_l in inject_layers:
            T_finals, P1_finals = [], []
            for prompt in PROMPTS:
                # Install noise hook
                if sigma > 0:
                    hook = make_safe_noise_hook(sigma)
                    handle = model.model.layers[inject_l].register_forward_hook(hook)

                inp = tok(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)

                # Measure at final layer
                final_hs = out.hidden_states[-1]
                with torch.no_grad():
                    normed = norm_layer(final_hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                P1 = float(probs.max().item())
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T = float(S) if not np.isnan(S) else 0

                T_finals.append(T)
                P1_finals.append(P1)

                if sigma > 0:
                    handle.remove()

            phase_data.append({
                'sigma': sigma,
                'inject_layer': inject_l,
                'inject_frac': inject_l / (n_layers - 1),
                'T_final': float(np.mean(T_finals)),
                'P1_final': float(np.mean(P1_finals)),
                'T_std': float(np.std(T_finals)),
            })

    # Phase boundary detection: where dP1/d(sigma) is maximally negative
    boundaries = []
    for inject_l in inject_layers:
        pts = [p for p in phase_data if p['inject_layer'] == inject_l]
        pts.sort(key=lambda x: x['sigma'])
        for i in range(len(pts) - 1):
            if pts[i]['sigma'] > 0 and pts[i+1]['sigma'] > 0:
                dP1 = pts[i+1]['P1_final'] - pts[i]['P1_final']
                d_sigma = np.log(pts[i+1]['sigma']) - np.log(pts[i]['sigma'])
                if d_sigma != 0:
                    gradient = dP1 / d_sigma
                    if gradient < -0.05:
                        boundaries.append({
                            'inject_layer': inject_l,
                            'sigma_critical': (pts[i]['sigma'] + pts[i+1]['sigma']) / 2,
                            'gradient': gradient,
                        })

    return {
        'model': model_name,
        'n_layers': n_layers,
        'phase_data': phase_data,
        'inject_layers': inject_layers,
        'boundaries': boundaries,
    }


def main():
    print("=" * 70)
    print("Phase 253: Thermodynamic Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = phase_diagram(model, tok, device, size)
        results[size] = r
        print(f"  {len(r['phase_data'])} phase points")
        print(f"  {len(r['boundaries'])} boundaries detected")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    inject_colors = plt.cm.viridis(np.linspace(0, 1, 8))

    for si, (size, r) in enumerate(results.items()):
        # (a,b) T_final vs sigma for different injection layers
        ax = axes[0, si]
        for ci, inject_l in enumerate(r['inject_layers']):
            pts = [p for p in r['phase_data'] if p['inject_layer'] == inject_l and p['sigma'] > 0]
            sigmas = [p['sigma'] for p in pts]
            T_vals = [p['T_final'] for p in pts]
            ax.semilogx(sigmas, T_vals, '-o', color=inject_colors[ci % 8],
                       lw=1.5, markersize=4, label=f'L{inject_l}')
        # Add sigma=0 point
        pts0 = [p for p in r['phase_data'] if p['sigma'] == 0]
        if pts0:
            ax.axhline(y=np.mean([p['T_final'] for p in pts0]),
                      color='gray', ls='--', alpha=0.3, label='sigma=0')
        ax.set_xlabel('Noise (sigma)')
        ax.set_ylabel('T_final')
        ax.set_title(f'({chr(97+si)}) Phase Diagram ({size})')
        ax.legend(fontsize=5)

    # (c) P1 vs sigma (1.5B, all injection layers)
    r15 = results[list(results.keys())[-1]]
    for ci, inject_l in enumerate(r15['inject_layers']):
        pts = [p for p in r15['phase_data'] if p['inject_layer'] == inject_l and p['sigma'] > 0]
        sigmas = [p['sigma'] for p in pts]
        P1_vals = [p['P1_final'] for p in pts]
        axes[0, 2].semilogx(sigmas, P1_vals, '-o', color=inject_colors[ci % 8],
                           lw=1.5, markersize=4, label=f'L{inject_l}')
    axes[0, 2].set_xlabel('Noise (sigma)')
    axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) Order Parameter vs Noise')
    axes[0, 2].legend(fontsize=5)

    # (d) Phase diagram heatmap: sigma vs inject_layer -> T
    sigma_vals = sorted(set(p['sigma'] for p in r15['phase_data'] if p['sigma'] > 0))
    inject_vals = sorted(set(p['inject_layer'] for p in r15['phase_data']))
    T_grid = np.zeros((len(sigma_vals), len(inject_vals)))
    for i, sigma in enumerate(sigma_vals):
        for j, inj in enumerate(inject_vals):
            pts = [p for p in r15['phase_data'] if p['sigma'] == sigma and p['inject_layer'] == inj]
            T_grid[i, j] = pts[0]['T_final'] if pts else 0
    im = axes[1, 0].imshow(T_grid, aspect='auto', cmap='hot', origin='lower',
                           extent=[0, len(inject_vals)-1, 0, len(sigma_vals)-1])
    axes[1, 0].set_xticks(range(len(inject_vals)))
    axes[1, 0].set_xticklabels([f'L{l}' for l in inject_vals], fontsize=6)
    axes[1, 0].set_yticks(range(len(sigma_vals)))
    axes[1, 0].set_yticklabels([f'{s:.3f}' for s in sigma_vals], fontsize=5)
    axes[1, 0].set_xlabel('Injection Layer')
    axes[1, 0].set_ylabel('Noise Level')
    axes[1, 0].set_title('(d) T Heatmap')
    fig.colorbar(im, ax=axes[1, 0], shrink=0.7)

    # (e) P1 heatmap
    P1_grid = np.zeros_like(T_grid)
    for i, sigma in enumerate(sigma_vals):
        for j, inj in enumerate(inject_vals):
            pts = [p for p in r15['phase_data'] if p['sigma'] == sigma and p['inject_layer'] == inj]
            P1_grid[i, j] = pts[0]['P1_final'] if pts else 0
    im2 = axes[1, 1].imshow(P1_grid, aspect='auto', cmap='viridis', origin='lower',
                            extent=[0, len(inject_vals)-1, 0, len(sigma_vals)-1])
    axes[1, 1].set_xticks(range(len(inject_vals)))
    axes[1, 1].set_xticklabels([f'L{l}' for l in inject_vals], fontsize=6)
    axes[1, 1].set_yticks(range(len(sigma_vals)))
    axes[1, 1].set_yticklabels([f'{s:.3f}' for s in sigma_vals], fontsize=5)
    axes[1, 1].set_xlabel('Injection Layer')
    axes[1, 1].set_ylabel('Noise Level')
    axes[1, 1].set_title('(e) P1 Heatmap')
    fig.colorbar(im2, ax=axes[1, 1], shrink=0.7)

    # (f) Summary
    summary = "PHASE DIAGRAM\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  {len(r['phase_data'])} points\n"
        summary += f"  {len(r['boundaries'])} boundaries\n"
        if r['boundaries']:
            summary += f"  sigma_c ~ {r['boundaries'][0]['sigma_critical']:.4f}\n"
        summary += "\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 253: Thermodynamic Phase Diagram",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase253_phase_diagram')
    plt.close()
    save_results('phase253_phase_diagram', {
        'experiment': 'Phase Diagram',
        'results': results,
    })


if __name__ == '__main__':
    main()
