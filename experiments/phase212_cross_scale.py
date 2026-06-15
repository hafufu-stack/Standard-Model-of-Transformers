# -*- coding: utf-8 -*-
"""
Phase 212: Cross-Scale Universality
=====================================
Test whether L0 (the thermodynamic phase transition layer) scales
universally across model sizes.

Compare Qwen2.5-0.5B (24 layers) and 1.5B (28 layers):
- Is L0/N_layers a universal constant?
- Do eta, T_hot, T_cold follow scaling laws?
- Does the dT/dLayer maximum position scale?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
]

MODEL_SIZES = ['0.5B', '1.5B']


def profile_model(model, tok, device, model_name):
    """Full thermodynamic profiling of a model across all prompts."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers) + 1  # +1 for embedding

    all_U = []
    all_T = []
    all_eta = []
    all_T_hot = []
    all_T_cold = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_list, T_list = [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)

        T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
        T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
        eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

        all_U.append(U_list)
        all_T.append(T_list)
        all_eta.append(eta)
        all_T_hot.append(T_hot)
        all_T_cold.append(T_cold)

    # Average profiles
    n_hs = min(len(t) for t in all_T)
    mean_U = [float(np.mean([p[i] for p in all_U if i < len(p)])) for i in range(n_hs)]
    mean_T = [float(np.mean([p[i] for p in all_T if i < len(p)])) for i in range(n_hs)]

    # dT/dLayer
    dT = [abs(mean_T[i+1] - mean_T[i]) for i in range(n_hs - 1)]

    # L0 definition: layer where dT is maximum (phase transition point)
    L0_dT_max = int(np.argmax(dT)) if dT else 0

    # L0 alternative: layer where T first drops below median
    median_T = np.median(mean_T[1:])
    L0_below_median = 0
    for i in range(1, n_hs):
        if mean_T[i] < median_T:
            L0_below_median = i
            break

    # L0 from inflection point of T (second derivative)
    d2T = [dT[i+1] - dT[i] for i in range(len(dT) - 1)] if len(dT) > 1 else []
    L0_inflection = int(np.argmax([abs(d) for d in d2T])) if d2T else 0

    n_transformer_layers = len(model.model.layers)

    return {
        'name': model_name,
        'n_layers': n_transformer_layers,
        'n_hidden_states': n_hs,
        'mean_U': mean_U,
        'mean_T': mean_T,
        'dT': [float(x) for x in dT],
        'L0_dT_max': L0_dT_max,
        'L0_below_median': L0_below_median,
        'L0_inflection': L0_inflection,
        'L0_ratio_dT': L0_dT_max / n_transformer_layers,
        'L0_ratio_median': L0_below_median / n_transformer_layers,
        'eta_mean': float(np.mean(all_eta)),
        'eta_std': float(np.std(all_eta)),
        'T_hot_mean': float(np.mean(all_T_hot)),
        'T_cold_mean': float(np.mean(all_T_cold)),
        'd_model': model.config.hidden_size,
    }


def main():
    print("=" * 70)
    print("Phase 212: Cross-Scale Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    profiles = {}

    for size in MODEL_SIZES:
        print(f"\n--- Profiling {size} ---")
        model, tok = load_model(device, size=size)
        p = profile_model(model, tok, device, size)
        profiles[size] = p
        print(f"  n_layers={p['n_layers']}, L0(dT_max)={p['L0_dT_max']}, "
              f"L0_ratio={p['L0_ratio_dT']:.3f}")
        print(f"  eta={p['eta_mean']:.4f}, T_hot={p['T_hot_mean']:.3f}, "
              f"T_cold={p['T_cold_mean']:.3f}")
        # Free memory
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Temperature profile comparison
    for size, p in profiles.items():
        x_norm = np.linspace(0, 1, len(p['mean_T']))
        axes[0, 0].plot(x_norm, p['mean_T'], '-', color=colors[size], lw=2,
                        label=f'{size} ({p["n_layers"]} layers)')
        axes[0, 0].axvline(x=p['L0_ratio_dT'], color=colors[size],
                           linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Normalized Layer (l/N)')
    axes[0, 0].set_ylabel('Temperature T (nats)')
    axes[0, 0].set_title('(a) Temperature Profile (normalized)')
    axes[0, 0].legend(fontsize=8)

    # (b) dT/dLayer comparison
    for size, p in profiles.items():
        x_norm = np.linspace(0, 1, len(p['dT']))
        axes[0, 1].plot(x_norm, p['dT'], '-', color=colors[size], lw=2,
                        label=f'{size}')
        axes[0, 1].axvline(x=p['L0_ratio_dT'], color=colors[size],
                           linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Normalized Layer (l/N)')
    axes[0, 1].set_ylabel('|dT/dLayer|')
    axes[0, 1].set_title('(b) Temperature Gradient')
    axes[0, 1].legend(fontsize=8)

    # (c) Internal energy comparison
    for size, p in profiles.items():
        x_norm = np.linspace(0, 1, len(p['mean_U']))
        axes[0, 2].plot(x_norm, p['mean_U'], '-', color=colors[size], lw=2,
                        label=f'{size}')
    axes[0, 2].set_xlabel('Normalized Layer (l/N)')
    axes[0, 2].set_ylabel('Internal Energy U')
    axes[0, 2].set_title('(c) Energy Profile')
    axes[0, 2].legend(fontsize=8)

    # (d) L0 ratio comparison
    l0_ratios_dT = [profiles[s]['L0_ratio_dT'] for s in MODEL_SIZES]
    l0_ratios_med = [profiles[s]['L0_ratio_median'] for s in MODEL_SIZES]
    x = np.arange(len(MODEL_SIZES))
    w = 0.35
    axes[1, 0].bar(x - w/2, l0_ratios_dT, w, label='L0 (dT max)', color='#e74c3c', alpha=0.7)
    axes[1, 0].bar(x + w/2, l0_ratios_med, w, label='L0 (below median)', color='#3498db', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(MODEL_SIZES)
    axes[1, 0].set_ylabel('L0 / N_layers')
    axes[1, 0].set_title('(d) L0 Ratio: Universal?')
    axes[1, 0].legend(fontsize=8)
    # Add values on bars
    for i, (v1, v2) in enumerate(zip(l0_ratios_dT, l0_ratios_med)):
        axes[1, 0].text(i - w/2, v1 + 0.02, f'{v1:.3f}', ha='center', fontsize=8)
        axes[1, 0].text(i + w/2, v2 + 0.02, f'{v2:.3f}', ha='center', fontsize=8)

    # (e) Thermodynamic parameters comparison
    params = ['eta_mean', 'T_hot_mean', 'T_cold_mean']
    param_labels = ['eta', 'T_hot', 'T_cold']
    x = np.arange(len(params))
    for si, size in enumerate(MODEL_SIZES):
        vals = [profiles[size][p] for p in params]
        offset = (si - 0.5) * 0.35
        axes[1, 1].bar(x + offset, vals, 0.3, label=size, color=colors[size], alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(param_labels)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('(e) Thermodynamic Parameters')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Cross-Scale Universality\n\n"
    for size in MODEL_SIZES:
        p = profiles[size]
        summary += (f"{size} ({p['n_layers']}L, d={p['d_model']}):\n"
                    f"  L0 = {p['L0_dT_max']} (ratio={p['L0_ratio_dT']:.3f})\n"
                    f"  eta = {p['eta_mean']:.4f}\n"
                    f"  T_hot/T_cold = {p['T_hot_mean']:.2f}/{p['T_cold_mean']:.2f}\n\n")
    ratio_diff = abs(profiles['0.5B']['L0_ratio_dT'] - profiles['1.5B']['L0_ratio_dT'])
    summary += f"L0 ratio difference: {ratio_diff:.3f}\n"
    summary += "UNIVERSAL" if ratio_diff < 0.1 else "NOT UNIVERSAL"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 212: Cross-Scale Universality", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase212_cross_scale')
    plt.close()

    save_results('phase212_cross_scale', {
        'experiment': 'Cross-Scale Universality',
        'profiles': profiles,
    })


if __name__ == '__main__':
    main()
