# -*- coding: utf-8 -*-
"""
Phase 246: Linear Response Theory
====================================
Test if transformers satisfy linear response: small perturbations in
input produce proportional changes in thermodynamic observables.
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

BASE_PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "Neural networks learn through gradient",
    "The speed of light is",
    "DNA encodes genetic information",
]

# Perturbation levels: append N random tokens
PERTURBATION_LEVELS = [0, 1, 2, 3, 5, 8, 13]

# Random tokens to append as perturbation
NOISE_TOKENS = ["cat", "seven", "blue", "running", "although",
                "banana", "physics", "cloud", "triangle", "coffee",
                "dream", "winter", "binary"]


def linear_response(model, tok, device, model_name):
    """Test linear response by measuring thermodynamic change vs perturbation size."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    def measure_T_profile(prompt):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, P1_l = [], []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
        return T_l, P1_l

    results_per_prompt = []
    for base in BASE_PROMPTS:
        base_T, base_P1 = measure_T_profile(base)
        base_T_final = base_T[-1]
        base_P1_final = base_P1[-1]

        pert_data = []
        for n_noise in PERTURBATION_LEVELS:
            if n_noise == 0:
                pert_data.append({
                    'n_noise': 0,
                    'dT': 0, 'dP1': 0,
                    'T_profile_diff': [0] * len(base_T),
                })
                continue

            # Multiple random perturbations for averaging
            dT_samples, dP1_samples = [], []
            for trial in range(3):
                noise = ' '.join(np.random.choice(NOISE_TOKENS, n_noise))
                perturbed = base + ' ' + noise
                pert_T, pert_P1 = measure_T_profile(perturbed)
                dT_samples.append(pert_T[-1] - base_T_final)
                dP1_samples.append(pert_P1[-1] - base_P1_final)

            pert_data.append({
                'n_noise': n_noise,
                'dT': float(np.mean(dT_samples)),
                'dP1': float(np.mean(dP1_samples)),
                'dT_std': float(np.std(dT_samples)),
            })

        results_per_prompt.append({
            'prompt': base[:40],
            'base_T_final': base_T_final,
            'base_P1_final': base_P1_final,
            'perturbations': pert_data,
        })

    # Aggregate linearity test
    all_n = []
    all_dT = []
    for rp in results_per_prompt:
        for pd in rp['perturbations']:
            if pd['n_noise'] > 0:
                all_n.append(pd['n_noise'])
                all_dT.append(abs(pd['dT']))

    r_linear, p_linear = stats.pearsonr(all_n, all_dT) if len(all_n) >= 3 else (0, 1)

    return {
        'model': model_name,
        'per_prompt': results_per_prompt,
        'r_linear': float(r_linear),
        'p_linear': float(p_linear),
    }


def main():
    print("=" * 70)
    print("Phase 246: Linear Response Theory")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = linear_response(model, tok, device, size)
        results[size] = r
        print(f"  Linearity: r={r['r_linear']:.3f} (p={r['p_linear']:.4f})")
        for rp in r['per_prompt']:
            dTs = [p['dT'] for p in rp['perturbations'] if p['n_noise'] > 0]
            print(f"    {rp['prompt'][:25]:>25}: dT range = {min(dTs):.2f} to {max(dTs):.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    prompt_colors = plt.cm.Set2(np.linspace(0, 1, len(BASE_PROMPTS)))

    # (a) |dT| vs perturbation size
    for size, r in results.items():
        for pi, rp in enumerate(r['per_prompt']):
            n_vals = [p['n_noise'] for p in rp['perturbations'] if p['n_noise'] > 0]
            dT_vals = [abs(p['dT']) for p in rp['perturbations'] if p['n_noise'] > 0]
            c = colors[size]
            axes[0, 0].scatter(n_vals, dT_vals, color=c, s=20, alpha=0.5)
        # Average
        from collections import defaultdict
        by_n = defaultdict(list)
        for rp in r['per_prompt']:
            for p in rp['perturbations']:
                if p['n_noise'] > 0:
                    by_n[p['n_noise']].append(abs(p['dT']))
        ns = sorted(by_n.keys())
        avg_dT = [np.mean(by_n[n]) for n in ns]
        axes[0, 0].plot(ns, avg_dT, '-o', color=colors[size], lw=2, label=f'{size} (r={r["r_linear"]:.2f})')
    axes[0, 0].set_xlabel('Perturbation Size (tokens)')
    axes[0, 0].set_ylabel('|dT|')
    axes[0, 0].set_title('(a) Response vs Perturbation')
    axes[0, 0].legend(fontsize=8)

    # (b) dP1 vs perturbation size
    for size, r in results.items():
        from collections import defaultdict
        by_n = defaultdict(list)
        for rp in r['per_prompt']:
            for p in rp['perturbations']:
                if p['n_noise'] > 0:
                    by_n[p['n_noise']].append(p['dP1'])
        ns = sorted(by_n.keys())
        avg_dP1 = [np.mean(by_n[n]) for n in ns]
        axes[0, 1].plot(ns, avg_dP1, '-o', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].set_xlabel('Perturbation Size')
    axes[0, 1].set_ylabel('dP1')
    axes[0, 1].set_title('(b) Order Parameter Response')
    axes[0, 1].legend(fontsize=8)

    # (c) Per-prompt response curves
    r15 = results[list(results.keys())[-1]]
    for pi, rp in enumerate(r15['per_prompt']):
        n_vals = [p['n_noise'] for p in rp['perturbations']]
        dT_vals = [p['dT'] for p in rp['perturbations']]
        axes[0, 2].plot(n_vals, dT_vals, '-o', color=prompt_colors[pi],
                       lw=1.5, markersize=4, label=rp['prompt'][:20])
    axes[0, 2].set_xlabel('Perturbation Size')
    axes[0, 2].set_ylabel('dT')
    axes[0, 2].set_title('(c) Per-Prompt Response')
    axes[0, 2].legend(fontsize=5)

    # (d) dT vs dP1 (correlation)
    for size, r in results.items():
        dTs, dP1s = [], []
        for rp in r['per_prompt']:
            for p in rp['perturbations']:
                if p['n_noise'] > 0:
                    dTs.append(p['dT']); dP1s.append(p['dP1'])
        axes[1, 0].scatter(dTs, dP1s, color=colors[size], s=20, alpha=0.5, label=size)
    axes[1, 0].set_xlabel('dT'); axes[1, 0].set_ylabel('dP1')
    axes[1, 0].set_title('(d) dT vs dP1 (Cross-Response)')
    axes[1, 0].legend(fontsize=8)

    # (e) Linearity test: residuals from linear fit
    for size, r in results.items():
        all_n, all_dT = [], []
        for rp in r['per_prompt']:
            for p in rp['perturbations']:
                if p['n_noise'] > 0:
                    all_n.append(p['n_noise'])
                    all_dT.append(abs(p['dT']))
        if len(all_n) >= 3:
            slope, intercept, _, _, _ = stats.linregress(all_n, all_dT)
            residuals = [dt - (slope * n + intercept) for n, dt in zip(all_n, all_dT)]
            axes[1, 1].scatter(all_n, residuals, color=colors[size], s=20, alpha=0.5, label=size)
    axes[1, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 1].set_xlabel('Perturbation Size')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('(e) Linearity Residuals')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "LINEAR RESPONSE\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Linearity: r={r['r_linear']:.3f}\n"
        summary += f"  p-value: {r['p_linear']:.4f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 246: Linear Response Theory",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase246_linear_response')
    plt.close()
    save_results('phase246_linear_response', {
        'experiment': 'Linear Response',
        'results': results,
    })


if __name__ == '__main__':
    main()
