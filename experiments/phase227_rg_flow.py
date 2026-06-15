# -*- coding: utf-8 -*-
"""
Phase 227: Renormalization Group Flow
========================================
Compute RG flow: how do thermodynamic observables change when we
coarse-grain the layer structure? If layers are self-similar,
RG flow should have fixed points.
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
]


def rg_flow(model, tok, device, model_name):
    """Compute RG flow via layer coarse-graining."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Full resolution profiles
    all_T, all_U, all_S, all_P1 = [], [], [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, U_l, S_l, P1_l = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1_l.append(float(probs.max().item()))
            T_l.append(float(S) if not np.isnan(S) else 0)
            S_l.append(float(S) if not np.isnan(S) else 0)
        all_T.append(T_l); all_U.append(U_l)
        all_S.append(S_l); all_P1.append(P1_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    T_full = avg(all_T)
    U_full = avg(all_U)
    S_full = avg(all_S)
    P1_full = avg(all_P1)

    # RG: coarse-grain at different scales
    rg_scales = [1, 2, 3, 4]  # Block sizes
    rg_results = {}

    for scale in rg_scales:
        n_blocks = n // scale
        if n_blocks < 2:
            continue
        T_cg = [np.mean(T_full[i*scale:(i+1)*scale]) for i in range(n_blocks)]
        U_cg = [np.mean(U_full[i*scale:(i+1)*scale]) for i in range(n_blocks)]
        S_cg = [np.mean(S_full[i*scale:(i+1)*scale]) for i in range(n_blocks)]
        P1_cg = [np.mean(P1_full[i*scale:(i+1)*scale]) for i in range(n_blocks)]

        # dT/dl at this scale
        dT_cg = [T_cg[i+1] - T_cg[i] for i in range(len(T_cg)-1)]

        # Normalized profile (scale-invariant form)
        T_range = max(T_cg) - min(T_cg) if max(T_cg) > min(T_cg) else 1
        T_norm = [(t - min(T_cg)) / T_range for t in T_cg]

        rg_results[scale] = {
            'T': T_cg, 'U': U_cg, 'S': S_cg, 'P1': P1_cg,
            'dT': dT_cg, 'T_norm': T_norm,
            'n_blocks': n_blocks,
        }

    # Fixed point detection: compare normalized profiles at different scales
    # Use Frechet distance between normalized curves
    fixed_point_quality = []
    ref = rg_results[1]['T_norm'] if 1 in rg_results else []
    for scale in rg_scales:
        if scale not in rg_results or scale == 1:
            continue
        cg_norm = rg_results[scale]['T_norm']
        # Interpolate to same length as reference
        if len(cg_norm) > 1 and len(ref) > 1:
            x_ref = np.linspace(0, 1, len(ref))
            x_cg = np.linspace(0, 1, len(cg_norm))
            cg_interp = np.interp(x_ref, x_cg, cg_norm)
            mse = float(np.mean((np.array(ref) - cg_interp) ** 2))
            fixed_point_quality.append({'scale': scale, 'mse': mse})

    # Beta function: how dT/dl changes with scale
    beta_T = {}
    for scale in rg_scales:
        if scale in rg_results:
            dT = rg_results[scale]['dT']
            beta_T[scale] = float(np.mean([abs(x) for x in dT]))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'T_full': T_full,
        'U_full': U_full,
        'rg_results': {str(k): v for k, v in rg_results.items()},
        'fixed_point_quality': fixed_point_quality,
        'beta_T': {str(k): v for k, v in beta_T.items()},
    }


def main():
    print("=" * 70)
    print("Phase 227: Renormalization Group Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = rg_flow(model, tok, device, size)
        results[size] = r
        for fp in r['fixed_point_quality']:
            print(f"  Scale {fp['scale']}: MSE={fp['mse']:.6f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    scale_alpha = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.3}

    # (a) T at different RG scales (0.5B)
    r05 = results['0.5B']
    for scale_str, rg in r05['rg_results'].items():
        scale = int(scale_str)
        x = np.linspace(0, 1, len(rg['T']))
        axes[0, 0].plot(x, rg['T'], '-', lw=2, alpha=scale_alpha.get(scale, 0.3),
                       label=f'scale={scale}')
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) RG Flow: T (0.5B)')
    axes[0, 0].legend(fontsize=7)

    # (b) T at different RG scales (1.5B)
    r15 = results['1.5B']
    for scale_str, rg in r15['rg_results'].items():
        scale = int(scale_str)
        x = np.linspace(0, 1, len(rg['T']))
        axes[0, 1].plot(x, rg['T'], '-', lw=2, alpha=scale_alpha.get(scale, 0.3),
                       label=f'scale={scale}')
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].set_title('(b) RG Flow: T (1.5B)')
    axes[0, 1].legend(fontsize=7)

    # (c) Normalized profiles overlay
    for size, r in results.items():
        for scale_str, rg in r['rg_results'].items():
            scale = int(scale_str)
            x = np.linspace(0, 1, len(rg['T_norm']))
            axes[0, 2].plot(x, rg['T_norm'], '-', lw=1.5,
                           alpha=scale_alpha.get(scale, 0.3),
                           color=colors[size],
                           label=f'{size} s={scale}' if scale <= 2 else None)
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('Normalized T')
    axes[0, 2].set_title('(c) Scale Invariance Test')
    axes[0, 2].legend(fontsize=7)

    # (d) Beta function
    for size, r in results.items():
        scales = sorted([int(k) for k in r['beta_T'].keys()])
        betas = [r['beta_T'][str(s)] for s in scales]
        axes[1, 0].plot(scales, betas, '-o', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('RG Scale')
    axes[1, 0].set_ylabel('Mean |dT/dl|')
    axes[1, 0].set_title('(d) Beta Function')
    axes[1, 0].legend(fontsize=8)

    # (e) Fixed point quality
    for size, r in results.items():
        if r['fixed_point_quality']:
            scales = [fp['scale'] for fp in r['fixed_point_quality']]
            mses = [fp['mse'] for fp in r['fixed_point_quality']]
            axes[1, 1].plot(scales, mses, '-o', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Scale')
    axes[1, 1].set_ylabel('MSE from reference')
    axes[1, 1].set_title('(e) Fixed Point Quality')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "RG Flow Analysis\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        for fp in r['fixed_point_quality']:
            summary += f"  s={fp['scale']}: MSE={fp['mse']:.6f}\n"
        summary += "\n"
    self_sim = all(fp['mse'] < 0.05 for r in results.values() for fp in r['fixed_point_quality'])
    summary += f"Self-similar: {'YES' if self_sim else 'NO'}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 227: Renormalization Group Flow", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase227_rg_flow')
    plt.close()
    save_results('phase227_rg_flow', {'experiment': 'RG Flow', 'results': results})


if __name__ == '__main__':
    main()
