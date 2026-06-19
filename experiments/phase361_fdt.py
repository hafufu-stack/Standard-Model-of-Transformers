# -*- coding: utf-8 -*-
"""
Phase 361: Fluctuation-Dissipation Theorem (FDT)
==================================================
Test whether the relationship between spontaneous fluctuations
and response to external perturbation follows the FDT:
  chi(omega) = (1/kT) * C(omega)
where chi is the susceptibility (response) and C is the autocorrelation
of spontaneous fluctuations.

Method:
1. Measure spontaneous fluctuations: run N prompts, measure variance
   of thermodynamic variables across prompts (no perturbation).
2. Measure response (susceptibility): inject small noise at each layer,
   measure how output changes (linear response).
3. Test FDT: check if response ~ fluctuation / temperature.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of relativity states that",
    "In quantum mechanics, the uncertainty principle",
    "The mitochondria is the powerhouse of",
    "Once upon a time in a distant galaxy",
    "The derivative of sin(x) is equal to",
    "According to the second law of thermodynamics",
    "The capital of France is known for",
    "Machine learning algorithms can be categorized",
    "The human genome contains approximately",
    "In philosophy, the concept of free will",
    "Water molecules consist of two hydrogen",
    "The speed of light in vacuum is",
]


def main():
    print("=" * 70)
    print("Phase 361: Fluctuation-Dissipation Theorem")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)
        n_layers = len(model.model.layers) + 1  # +1 for embedding

        # Step 1: Spontaneous fluctuations (no perturbation)
        all_U = []
        all_T = []
        for p in PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, p, device)
            all_U.append([t['U'] for t in thermo])
            all_T.append([t['T'] for t in thermo])

        all_U = np.array(all_U)  # (N_prompts, n_layers)
        all_T = np.array(all_T)

        # Spontaneous fluctuation = variance across prompts at each layer
        fluct_U = np.var(all_U, axis=0)  # (n_layers,)
        fluct_T = np.var(all_T, axis=0)
        mean_T = np.mean(all_T, axis=0)

        # Step 2: Response (susceptibility) via noise injection
        sigma_probe = 0.01
        response_U = np.zeros(n_layers)

        base_prompt = PROMPTS[0]
        thermo_base, _ = measure_full_thermodynamics(model, tok, base_prompt, device)
        U_base = np.array([t['U'] for t in thermo_base])

        # Inject noise at each layer and measure response
        for layer_idx in range(min(n_layers - 1, len(model.model.layers))):
            def noise_hook(module, input, output, sigma=sigma_probe):
                h = output[0] if isinstance(output, tuple) else output
                h_fp32 = h.to(torch.float32)
                noise = torch.randn_like(h_fp32) * sigma
                h_mod = torch.nan_to_num(h_fp32 + noise, nan=0.0)
                result = h_mod.to(h.dtype)
                if isinstance(output, tuple):
                    return (result,) + output[1:]
                return result

            handle = model.model.layers[layer_idx].register_forward_hook(noise_hook)
            thermo_pert, _ = measure_full_thermodynamics(model, tok, base_prompt, device)
            handle.remove()

            U_pert = np.array([t['U'] for t in thermo_pert])
            # Response = |delta U| / sigma at layers after perturbation
            delta = np.abs(U_pert - U_base)
            response_U[layer_idx] = np.mean(delta[layer_idx:]) / sigma_probe

        # Step 3: FDT test
        # FDT predicts: response ~ fluctuation / T
        fdt_predicted = fluct_U / (mean_T + 1e-10)
        valid = (response_U > 0) & (fdt_predicted > 0) & np.isfinite(fdt_predicted)

        if valid.sum() > 2:
            r_fdt, p_fdt = stats.pearsonr(response_U[valid], fdt_predicted[valid])
        else:
            r_fdt, p_fdt = 0.0, 1.0

        results[size] = {
            'fdt_correlation': float(r_fdt),
            'fdt_pvalue': float(p_fdt),
            'fluct_U_mean': float(np.mean(fluct_U)),
            'response_U_mean': float(np.mean(response_U)),
            'mean_T_final': float(mean_T[-1]),
        }
        print(f"  FDT correlation: r={r_fdt:.3f}, p={p_fdt:.4f}")
        print(f"  Fluctuation mean: {np.mean(fluct_U):.4f}")
        print(f"  Response mean: {np.mean(response_U):.4f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 361: Fluctuation-Dissipation Theorem", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        r = results[size]['fdt_correlation']
        ax.text(0.5, 0.5, f'FDT Correlation\nr = {r:.3f}',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.set_xlabel('Layer')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase361_fdt')
    plt.close()

    save_results('phase361_fdt', {
        'experiment': 'Fluctuation-Dissipation Theorem',
        'results': results,
    })


if __name__ == '__main__':
    main()
