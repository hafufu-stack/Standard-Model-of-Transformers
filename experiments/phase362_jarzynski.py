# -*- coding: utf-8 -*-
"""
Phase 362: Jarzynski Equality
================================
Test whether non-equilibrium work relates to free energy differences:
  <exp(-W/kT)> = exp(-delta_F/kT)
  
Method:
1. Define "equilibrium" state as the model processing a standard prompt.
2. Define "non-equilibrium process" as injecting noise at increasing sigmas.
3. Measure "work" W = sum of energy changes during the process.
4. Measure free energy F = U - T*S at start and end states.
5. Test if <exp(-W/T)> = exp(-delta_F/T).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of relativity states that",
    "In quantum mechanics, the uncertainty principle",
    "Machine learning algorithms can be categorized",
    "The human genome contains approximately",
    "Water molecules consist of two hydrogen",
    "The speed of light in vacuum is",
]

SIGMA_RANGE = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


def measure_free_energy(thermo_list):
    """Compute F = U - T*S where S ~ log(PR)."""
    results = []
    for t in thermo_list:
        U = t['U']
        T = t['T']
        S = np.log(t['PR'] + 1e-10)
        F = U - T * S
        results.append({'U': U, 'T': T, 'S': S, 'F': F})
    return results


def main():
    print("=" * 70)
    print("Phase 362: Jarzynski Equality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        jarzynski_data = []

        for prompt in PROMPTS:
            # Equilibrium state (no noise)
            thermo_eq, _ = measure_full_thermodynamics(model, tok, prompt, device)
            fe_eq = measure_free_energy(thermo_eq)
            F_eq = fe_eq[-1]['F']  # Final layer free energy

            work_values = []
            for sigma in SIGMA_RANGE[1:]:  # Skip sigma=0
                # Non-equilibrium: inject noise at all layers
                handles = []
                for layer_idx in range(len(model.model.layers)):
                    def noise_hook(module, input, output, s=sigma):
                        h = output[0] if isinstance(output, tuple) else output
                        h_fp32 = h.to(torch.float32)
                        noise = torch.randn_like(h_fp32) * s
                        h_mod = torch.nan_to_num(h_fp32 + noise, nan=0.0)
                        result = h_mod.to(h.dtype)
                        if isinstance(output, tuple):
                            return (result,) + output[1:]
                        return result
                    handles.append(model.model.layers[layer_idx].register_forward_hook(noise_hook))

                thermo_neq, _ = measure_full_thermodynamics(model, tok, prompt, device)
                for h in handles:
                    h.remove()

                fe_neq = measure_free_energy(thermo_neq)
                F_neq = fe_neq[-1]['F']

                # Work = sum of energy increments through layers
                W = sum(thermo_neq[i+1]['U'] - thermo_neq[i]['U']
                       for i in range(len(thermo_neq)-1))

                T_mean = np.mean([t['T'] for t in thermo_neq]) + 1e-10
                work_values.append({
                    'sigma': sigma,
                    'W': float(W),
                    'F_eq': float(F_eq),
                    'F_neq': float(F_neq),
                    'delta_F': float(F_neq - F_eq),
                    'T_mean': float(T_mean),
                    'exp_neg_W_T': float(np.exp(-W / T_mean)) if abs(W/T_mean) < 50 else 0.0,
                })

            jarzynski_data.append(work_values)

        # Aggregate Jarzynski equality test
        # For each sigma, average exp(-W/T) across prompts
        jarzynski_test = []
        for sigma_idx, sigma in enumerate(SIGMA_RANGE[1:]):
            exp_neg_W_T_values = [jarzynski_data[p][sigma_idx]['exp_neg_W_T']
                                  for p in range(len(PROMPTS))]
            delta_F_values = [jarzynski_data[p][sigma_idx]['delta_F']
                             for p in range(len(PROMPTS))]
            T_values = [jarzynski_data[p][sigma_idx]['T_mean']
                       for p in range(len(PROMPTS))]

            mean_exp = np.mean(exp_neg_W_T_values)
            mean_dF = np.mean(delta_F_values)
            mean_T = np.mean(T_values)

            # Jarzynski: <exp(-W/T)> should equal exp(-dF/T)
            if abs(mean_dF / mean_T) < 50:
                jarzynski_rhs = np.exp(-mean_dF / mean_T)
            else:
                jarzynski_rhs = 0.0

            jarzynski_test.append({
                'sigma': sigma,
                'lhs_mean_exp_neg_W_T': float(mean_exp),
                'rhs_exp_neg_dF_T': float(jarzynski_rhs),
                'ratio': float(mean_exp / (jarzynski_rhs + 1e-10)),
            })
            print(f"  sigma={sigma}: <exp(-W/T)>={mean_exp:.4f}, exp(-dF/T)={jarzynski_rhs:.4f}, ratio={mean_exp/(jarzynski_rhs+1e-10):.4f}")

        results[size] = {
            'jarzynski_test': jarzynski_test,
            'n_prompts': len(PROMPTS),
            'n_sigmas': len(SIGMA_RANGE) - 1,
        }

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 362: Jarzynski Equality Test", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        jt = results[size]['jarzynski_test']
        sigmas = [j['sigma'] for j in jt]
        lhs = [j['lhs_mean_exp_neg_W_T'] for j in jt]
        rhs = [j['rhs_exp_neg_dF_T'] for j in jt]
        ratios = [j['ratio'] for j in jt]

        ax.plot(sigmas, ratios, 'o-', color='#e74c3c', lw=2, label='LHS/RHS ratio')
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Perfect equality')
        ax.set_xscale('log')
        ax.set_xlabel('Noise sigma')
        ax.set_ylabel('Jarzynski Ratio')
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase362_jarzynski')
    plt.close()

    save_results('phase362_jarzynski', {
        'experiment': 'Jarzynski Equality',
        'results': results,
    })


if __name__ == '__main__':
    main()
