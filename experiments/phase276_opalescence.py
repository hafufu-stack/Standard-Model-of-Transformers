# -*- coding: utf-8 -*-
"""
Phase 276: Critical Opalescence
==================================
Near the critical layer L0, fluctuations should diverge like a
second-order phase transition. Measure variance of thermodynamic
quantities across 100 prompts at each layer, focusing on L0 +/- 3.

Identify: correlation length xi, critical exponents, slowing down.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from utils import load_model, save_results, save_figure

# 30 diverse prompts for statistical power
PROMPTS = [
    "The cat sat on the mat",
    "Quantum mechanics describes the behavior of particles",
    "In the year 2050 technology will have",
    "The most important discovery in physics was",
    "Machine learning algorithms can process data",
    "The fundamental theorem states that",
    "Climate change affects ecosystems worldwide",
    "Neural networks learn hierarchical representations",
    "The speed of light in vacuum is constant",
    "Economic theories predict market behavior",
    "DNA replication occurs through complementary base pairing",
    "The history of mathematics spans thousands of years",
    "Artificial intelligence raises ethical questions",
    "The structure of proteins determines their function",
    "Renewable energy sources include solar and wind",
    "The universe is approximately fourteen billion years old",
    "Language models process text through attention mechanisms",
    "The laws of thermodynamics govern energy transfer",
    "Evolution by natural selection shapes biological diversity",
    "Computer science fundamentals include algorithms and data structures",
    "The human brain contains billions of neurons",
    "Cryptography protects information through mathematical techniques",
    "The periodic table organizes chemical elements",
    "Statistical mechanics bridges microscopic and macroscopic phenomena",
    "The internet has transformed global communication",
    "Quantum computing leverages superposition and entanglement",
    "Photosynthesis converts light energy into chemical energy",
    "The theory of relativity describes spacetime geometry",
    "Ocean currents distribute heat across the planet",
    "Programming languages provide abstractions for computation",
]


def measure_layer_statistics(model, tok, prompts, device):
    """Measure T, P1, U, PR at every layer for many prompts."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Collect per-layer, per-prompt measurements
    all_layer_data = None  # will be [n_layers x n_prompts x 4] (T, P1, U, PR)

    for pi, prompt in enumerate(prompts):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        n_layers = len(out.hidden_states)
        if all_layer_data is None:
            all_layer_data = np.zeros((n_layers, len(prompts), 4))

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            p1 = probs.max().item()
            t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(t_val): t_val = 0

            all_layer_data[li, pi, :] = [t_val, p1, U, PR]

        if (pi + 1) % 10 == 0:
            print(f"    Processed {pi+1}/{len(prompts)} prompts")

    return all_layer_data, n_layers


def main():
    print("=" * 70)
    print("Phase 276: Critical Opalescence")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        data, n_layers = measure_layer_statistics(model, tok, PROMPTS, device)

        # Compute variance at each layer
        var_T = np.var(data[:, :, 0], axis=1)   # variance of T across prompts
        var_P1 = np.var(data[:, :, 1], axis=1)
        var_U = np.var(data[:, :, 2], axis=1)
        var_PR = np.var(data[:, :, 3], axis=1)

        mean_T = np.mean(data[:, :, 0], axis=1)

        # Find critical layer: maximum variance of T (susceptibility peak)
        L0 = int(np.argmax(var_T[1:]) + 1)  # skip layer 0

        # Correlation length: fit var_T ~ |L - L0|^(-gamma)
        layers = np.arange(n_layers)
        dist_from_L0 = np.abs(layers - L0).astype(float)
        dist_from_L0[L0] = 0.5  # avoid log(0)

        # Only fit layers around L0
        fit_mask = (dist_from_L0 > 0) & (dist_from_L0 < n_layers // 2) & (var_T > 0)
        if fit_mask.sum() >= 3:
            log_dist = np.log(dist_from_L0[fit_mask])
            log_var = np.log(var_T[fit_mask])
            slope, intercept, r_val, _, _ = stats.linregress(log_dist, log_var)
            gamma = -slope  # critical exponent
        else:
            gamma = 0
            r_val = 0

        # Susceptibility chi = var(P1) * n_prompts
        chi = var_P1 * len(PROMPTS)
        chi_peak = float(chi.max())
        chi_L0 = int(np.argmax(chi[1:]) + 1)

        all_results[size] = {
            'n_layers': n_layers,
            'L0_variance_peak': L0,
            'L0_susceptibility_peak': chi_L0,
            'gamma': round(float(gamma), 4),
            'gamma_R2': round(float(r_val**2), 4),
            'chi_peak': round(chi_peak, 6),
            'var_T': var_T.tolist(),
            'var_P1': var_P1.tolist(),
            'var_U': var_U.tolist(),
            'mean_T': mean_T.tolist(),
            'chi': chi.tolist(),
        }

        print(f"  L0 (var peak) = {L0}")
        print(f"  L0 (chi peak) = {chi_L0}")
        print(f"  gamma = {gamma:.3f} (R2={r_val**2:.3f})")
        print(f"  Chi peak = {chi_peak:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        c = colors[size]
        layers = range(data['n_layers'])

        # (a) Variance of T per layer
        axes[0, 0].plot(layers, data['var_T'], '-', color=c, lw=2, label=size)
        axes[0, 0].axvline(data['L0_variance_peak'], color=c, ls='--', alpha=0.5)

    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Var(T)')
    axes[0, 0].set_title('(a) Temperature Fluctuations (Opalescence)', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Susceptibility
    for size, data in all_results.items():
        c = colors[size]
        axes[0, 1].plot(range(data['n_layers']), data['chi'], '-', color=c, lw=2, label=size)
        axes[0, 1].axvline(data['L0_susceptibility_peak'], color=c, ls='--', alpha=0.5)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Susceptibility chi')
    axes[0, 1].set_title('(b) Susceptibility Peak', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Variance of U
    for size, data in all_results.items():
        axes[0, 2].plot(range(data['n_layers']), data['var_U'], '-',
                       color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Var(U)')
    axes[0, 2].set_title('(c) Internal Energy Fluctuations', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Log-log: var(T) vs |L-L0|
    for size, data in all_results.items():
        L0 = data['L0_variance_peak']
        n = data['n_layers']
        dist = np.abs(np.arange(n) - L0).astype(float)
        dist[L0] = 0.5
        var_t = np.array(data['var_T'])
        mask = (dist > 0.5) & (var_t > 0)
        if mask.sum() > 0:
            axes[1, 0].scatter(np.log(dist[mask]), np.log(var_t[mask]),
                              c=colors[size], s=30, label=size)
    axes[1, 0].set_xlabel('log|L - L0|')
    axes[1, 0].set_ylabel('log Var(T)')
    axes[1, 0].set_title('(d) Critical Exponent gamma', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Mean T profile
    for size, data in all_results.items():
        axes[1, 1].plot(range(data['n_layers']), data['mean_T'], '-',
                       color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Mean T')
    axes[1, 1].set_title('(e) Mean Temperature Profile', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "CRITICAL OPALESCENCE\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  L0 (var peak) = {data['L0_variance_peak']}\n"
        summary += f"  L0 (chi peak) = {data['L0_susceptibility_peak']}\n"
        summary += f"  gamma = {data['gamma']:.3f}\n"
        summary += f"  R2 = {data['gamma_R2']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 276: Critical Opalescence at L0",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase276_opalescence')
    plt.close()

    save_results('phase276_opalescence', {
        'experiment': 'Critical Opalescence',
        'n_prompts': len(PROMPTS),
        'results': all_results,
    })


if __name__ == '__main__':
    main()
