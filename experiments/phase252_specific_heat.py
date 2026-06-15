# -*- coding: utf-8 -*-
"""
Phase 252: Specific Heat and Critical Phenomena
==================================================
Compute the specific heat C = dU/dT at each layer and across models.
Critical phenomena: divergence of C at phase transition points.
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

# Use prompts that span a wide temperature range
PROMPTS_COLD = [
    "One plus one equals", "The sky is blue", "Water is wet",
    "The sun is yellow", "Cats have four legs",
]
PROMPTS_WARM = [
    "The fundamental theorem of calculus",
    "Quantum entanglement occurs when particles",
    "Neural networks approximate functions through",
    "Evolution by natural selection explains",
    "The uncertainty principle states that",
]
PROMPTS_HOT = [
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Seven abstract thoughts collided creating new",
    "Silence tasted exactly like growing uncertainty",
    "The moon decided to become a professional",
]


def specific_heat(model, tok, device, model_name):
    """Compute specific heat at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    def profile(prompts):
        all_T, all_U = [], []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            T_l, U_l = [], []
            for hs in out.hidden_states:
                h = hs[0, -1, :].float()
                U_l.append(h.norm().item())
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_l.append(float(S) if not np.isnan(S) else 0)
            all_T.append(T_l); all_U.append(U_l)
        n = min(len(t) for t in all_T)
        avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
        return avg(all_T), avg(all_U)

    cold_T, cold_U = profile(PROMPTS_COLD)
    warm_T, warm_U = profile(PROMPTS_WARM)
    hot_T, hot_U = profile(PROMPTS_HOT)

    n = min(len(cold_T), len(warm_T), len(hot_T))

    # Specific heat at each layer: C = dU/dT
    # Use the three temperature regimes to compute finite differences
    C_layers = []
    for l in range(n):
        T_vals = [cold_T[l], warm_T[l], hot_T[l]]
        U_vals = [cold_U[l], warm_U[l], hot_U[l]]
        if max(T_vals) - min(T_vals) > 0.01:
            slope, _, _, _, _ = stats.linregress(T_vals, U_vals)
            C_layers.append(float(slope))
        else:
            C_layers.append(0.0)

    # Also compute C from layer derivatives: dU/dl / dT/dl
    all_prompts = PROMPTS_COLD + PROMPTS_WARM + PROMPTS_HOT
    all_T_full, all_U_full = profile(all_prompts)
    
    dT = np.diff(all_T_full)
    dU = np.diff(all_U_full)
    C_deriv = dU / (dT + 1e-10)

    # Peak finding (critical point)
    C_abs = np.abs(C_layers)
    peak_layer = int(np.argmax(C_abs))

    # Fluctuation-based specific heat: C = var(U) / T^2
    C_fluct = []
    for l in range(n):
        U_vals = [cold_U[l], warm_U[l], hot_U[l]]
        T_mean = np.mean([cold_T[l], warm_T[l], hot_T[l]])
        var_U = np.var(U_vals)
        C_fluct.append(float(var_U / (T_mean**2 + 1e-10)))

    return {
        'model': model_name,
        'cold_T': cold_T[:n], 'cold_U': cold_U[:n],
        'warm_T': warm_T[:n], 'warm_U': warm_U[:n],
        'hot_T': hot_T[:n], 'hot_U': hot_U[:n],
        'C_layers': C_layers,
        'C_deriv': C_deriv.tolist(),
        'C_fluct': C_fluct,
        'peak_layer': peak_layer,
    }


def main():
    print("=" * 70)
    print("Phase 252: Specific Heat and Critical Phenomena")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = specific_heat(model, tok, device, size)
        results[size] = r
        print(f"  Peak C at layer: {r['peak_layer']}")
        print(f"  C range: {min(r['C_layers']):.3f} to {max(r['C_layers']):.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) C(l) from temperature regime
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['C_layers'])), r['C_layers'], '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
        axes[0, 0].axvline(x=r['peak_layer'], color=colors[size], ls='--', alpha=0.5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('C = dU/dT')
    axes[0, 0].set_title('(a) Specific Heat')
    axes[0, 0].legend(fontsize=8)

    # (b) C from layer derivatives
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['C_deriv'])), r['C_deriv'], '-',
                       color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('C = (dU/dl)/(dT/dl)')
    axes[0, 1].set_title('(b) C from Derivatives')
    axes[0, 1].legend(fontsize=8)

    # (c) Fluctuation-based C
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['C_fluct'])), r['C_fluct'], '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('C_fluct = var(U)/T^2')
    axes[0, 2].set_title('(c) Fluctuation C')
    axes[0, 2].legend(fontsize=8)

    # (d) T-U scatter at different regimes
    r15 = results[list(results.keys())[-1]]
    axes[1, 0].scatter(r15['cold_T'], r15['cold_U'], c='blue', s=20, alpha=0.6, label='Cold')
    axes[1, 0].scatter(r15['warm_T'], r15['warm_U'], c='orange', s=20, alpha=0.6, label='Warm')
    axes[1, 0].scatter(r15['hot_T'], r15['hot_U'], c='red', s=20, alpha=0.6, label='Hot')
    axes[1, 0].set_xlabel('T'); axes[1, 0].set_ylabel('U')
    axes[1, 0].set_title('(d) T-U at Different Regimes')
    axes[1, 0].legend(fontsize=7)

    # (e) |C| log scale (divergence test)
    for size, r in results.items():
        C_abs = [abs(c) + 1e-5 for c in r['C_layers']]
        axes[1, 1].semilogy(range(len(C_abs)), C_abs, '-o',
                           color=colors[size], lw=2, markersize=3, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('|C| (log)')
    axes[1, 1].set_title('(e) |C| Divergence Test')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "SPECIFIC HEAT\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Peak C at layer: {r['peak_layer']}\n"
        summary += f"  C range: [{min(r['C_layers']):.2f}, {max(r['C_layers']):.2f}]\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 252: Specific Heat and Critical Phenomena",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase252_specific_heat')
    plt.close()
    save_results('phase252_specific_heat', {
        'experiment': 'Specific Heat',
        'results': results,
    })


if __name__ == '__main__':
    main()
