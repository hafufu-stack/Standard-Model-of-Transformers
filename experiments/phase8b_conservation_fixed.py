# -*- coding: utf-8 -*-
"""
Phase 8b: Conservation Law Stress Test (FIXED)
================================================
Season 1 bug: fp16 noise was absorbed by machine epsilon.
Fix: Generate noise in fp32 space, then downcast.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 8b: Conservation Stress Test (FIXED)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The fundamental theorem of calculus states",
        "In statistical mechanics entropy is",
        "Quantum entanglement allows particles to",
        "The speed of light is approximately",
        "Neural networks learn by adjusting weights",
        "The second law of thermodynamics states",
        "Black holes emit Hawking radiation because",
        "Attention mechanism computes weighted sums",
    ]

    def measure_conservation(model, tok, prompts, device, label=""):
        all_prt = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            for layer_idx in range(1, len(out.hidden_states)):
                h = out.hidden_states[layer_idx][0, -1, :].float()
                T = h.norm().item()
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                layer_pr = 1.0 / (h_prob ** 2).sum().item()
                all_prt.append(layer_pr * T)

        mean_prt = np.mean(all_prt)
        std_prt = np.std(all_prt)
        cv = std_prt / (mean_prt + 1e-10) * 100
        if label:
            print(f"  {label}: PR*T = {mean_prt:.2f} +/- {std_prt:.2f} (CV={cv:.1f}%)")
        return mean_prt, std_prt, cv, all_prt

    # Baseline
    print("\n--- Baseline ---")
    baseline_mean, baseline_std, baseline_cv, _ = \
        measure_conservation(model, tok, prompts, device, "Baseline")

    # FIX: fp32 noise generation
    print("\n--- Noise Injection (fp32 fix) ---")
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    noise_results = []

    for sigma in noise_levels:
        handles = []
        if sigma > 0:
            for li in range(n_layers):
                def make_fp32_noise_hook(s):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            noise = torch.randn(h.shape, dtype=torch.float32, device=h.device) * s
                            h_mod = h.float() + noise
                            return (h_mod.to(h.dtype),) + output[1:]
                        noise = torch.randn(output.shape, dtype=torch.float32, device=output.device) * s
                        return (output.float() + noise).to(output.dtype)
                    return hook
                h = model.model.layers[li].register_forward_hook(make_fp32_noise_hook(sigma))
                handles.append(h)

        mean_prt, std_prt, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Noise sigma={sigma:.2f}")

        for h in handles:
            h.remove()

        noise_results.append({
            'sigma': sigma, 'mean_prt': mean_prt, 'cv': cv,
            'deviation': abs(mean_prt - baseline_mean) / (baseline_mean + 1e-10) * 100,
        })

    # Quantization (fp32 simulation)
    print("\n--- Quantization (fp32 simulation) ---")
    quant_bits = [32, 16, 8, 4, 2]
    quant_results = []

    for bits in quant_bits:
        handles = []
        if bits < 32:
            for li in range(n_layers):
                def make_quant_hook(b):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].float()
                            scale = h.abs().max() / (2 ** (b - 1) - 1 + 1e-10)
                            h_q = torch.round(h / (scale + 1e-10)) * scale
                            return (h_q.to(output[0].dtype),) + output[1:]
                        return output
                    return hook
                h = model.model.layers[li].register_forward_hook(make_quant_hook(bits))
                handles.append(h)

        mean_prt, _, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Quant {bits}-bit")

        for h in handles:
            h.remove()

        quant_results.append({
            'bits': bits, 'mean_prt': mean_prt, 'cv': cv,
            'deviation': abs(mean_prt - baseline_mean) / (baseline_mean + 1e-10) * 100,
        })

    # Pruning
    print("\n--- Layer Pruning ---")
    prune_counts = [0, 1, 2, 4, 8, 14]
    prune_results = []

    for n_prune in prune_counts:
        handles = []
        if n_prune > 0:
            mid = n_layers // 2
            prune_layers = list(range(mid - n_prune // 2, mid + (n_prune + 1) // 2))
            prune_layers = [l for l in prune_layers if 0 <= l < n_layers]
            for li in prune_layers:
                def make_identity():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            inp_h = input[0] if isinstance(input, tuple) else input
                            return (inp_h,) + output[1:]
                        return input[0] if isinstance(input, tuple) else input
                    return hook
                h = model.model.layers[li].register_forward_hook(make_identity())
                handles.append(h)

        mean_prt, _, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Prune {n_prune} layers")

        for h in handles:
            h.remove()

        prune_results.append({
            'n_pruned': n_prune, 'mean_prt': mean_prt, 'cv': cv,
            'deviation': abs(mean_prt - baseline_mean) / (baseline_mean + 1e-10) * 100,
        })

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sigmas = [r['sigma'] for r in noise_results]
    means = [r['mean_prt'] for r in noise_results]
    ax.plot(sigmas, means, 'o-', color='#e74c3c', ms=8, lw=2)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5, label=f'Baseline={baseline_mean:.0f}')
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('PR x T')
    ax.set_title('(a) Noise Robustness (fp32 fix)')
    ax.legend()

    ax = axes[1]
    ax.bar([str(b) for b in [r['bits'] for r in quant_results]],
           [r['mean_prt'] for r in quant_results], color='#f39c12', alpha=0.8)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('PR x T')
    ax.set_title('(b) Quantization Robustness')

    ax = axes[2]
    ax.bar([str(r['n_pruned']) for r in prune_results],
           [r['mean_prt'] for r in prune_results], color='#2ecc71', alpha=0.8)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layers Pruned')
    ax.set_ylabel('PR x T')
    ax.set_title('(c) Pruning Robustness')

    all_devs = ([r['deviation'] for r in noise_results[1:]] +
                [r['deviation'] for r in quant_results] +
                [r['deviation'] for r in prune_results])
    avg_dev = np.mean(all_devs)

    fig.suptitle(
        f"Phase 8b: Conservation Stress (FIXED)\n"
        f"Baseline PR*T={baseline_mean:.0f} | Avg deviation={avg_dev:.1f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase8b_conservation_fixed")
    plt.close()

    if avg_dev < 20:
        verdict = f"GENUINE CONSERVATION: PR*T={baseline_mean:.0f}, avg deviation={avg_dev:.1f}% under perturbation."
    else:
        verdict = f"FRAGILE: PR*T={baseline_mean:.0f}, avg deviation={avg_dev:.1f}%. Breaks under stress."

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 8b: Conservation Stress Test (FIXED)',
        'summary': {'verdict': verdict, 'baseline_mean': baseline_mean, 'avg_deviation': avg_dev},
        'noise': noise_results, 'quant': quant_results, 'prune': prune_results,
    }
    save_results("phase8b_conservation_fixed", result)
    return result


if __name__ == '__main__':
    main()
