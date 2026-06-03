# -*- coding: utf-8 -*-
"""
Phase 8: Conservation Law Stress Test
=======================================
Test whether PR x T = 50.1 is a genuine Noether conservation
quantity by subjecting it to various perturbations:
- Temperature scaling
- Noise injection
- Layer pruning
- Quantization (8-bit, 4-bit simulation)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 8: Conservation Law Stress Test")
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
        """Measure PR x T at each layer for given prompts."""
        all_prt = []
        all_pr = []
        all_t = []

        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            pr = 1.0 / (probs ** 2).sum().item()

            for layer_idx in range(1, len(out.hidden_states)):
                h = out.hidden_states[layer_idx][0, -1, :].float()
                T = h.norm().item()

                # Layer-specific PR from hidden state
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                layer_pr = 1.0 / (h_prob ** 2).sum().item()

                prt = layer_pr * T
                all_prt.append(prt)
                all_pr.append(layer_pr)
                all_t.append(T)

        mean_prt = np.mean(all_prt)
        std_prt = np.std(all_prt)
        cv = std_prt / (mean_prt + 1e-10) * 100

        if label:
            print(f"  {label}: PR*T = {mean_prt:.2f} +/- {std_prt:.2f} (CV={cv:.1f}%)")

        return mean_prt, std_prt, cv, all_prt

    # ================================================================
    # Baseline measurement
    # ================================================================
    print("\n--- Baseline ---")
    baseline_mean, baseline_std, baseline_cv, baseline_prt = \
        measure_conservation(model, tok, prompts, device, "Baseline")

    # ================================================================
    # Perturbation 1: Noise injection at various levels
    # ================================================================
    print("\n--- Perturbation 1: Noise Injection ---")
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    noise_results = []

    for sigma in noise_levels:
        handles = []
        if sigma > 0:
            for li in range(n_layers):
                def make_noise_hook(s):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            return (h + torch.randn_like(h) * s,) + output[1:]
                        return output
                    return hook
                h = model.model.layers[li].register_forward_hook(make_noise_hook(sigma))
                handles.append(h)

        mean_prt, std_prt, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Noise sigma={sigma:.2f}")

        for h in handles:
            h.remove()

        noise_results.append({
            'sigma': sigma,
            'mean_prt': mean_prt,
            'std_prt': std_prt,
            'cv': cv,
            'deviation_from_baseline': abs(mean_prt - baseline_mean) / baseline_mean * 100,
        })

    # ================================================================
    # Perturbation 2: Temperature scaling
    # ================================================================
    print("\n--- Perturbation 2: Temperature Scaling ---")
    temp_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    temp_results = []

    for temp in temp_scales:
        handles = []
        for li in range(n_layers):
            def make_scale_hook(s):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (output[0] * s,) + output[1:]
                    return output * s
                return hook
            h = model.model.layers[li].register_forward_hook(make_scale_hook(temp))
            handles.append(h)

        mean_prt, std_prt, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Temp scale={temp:.1f}")

        for h in handles:
            h.remove()

        temp_results.append({
            'temp_scale': temp,
            'mean_prt': mean_prt,
            'std_prt': std_prt,
            'cv': cv,
        })

    # ================================================================
    # Perturbation 3: Layer pruning (remove 1, 2, 4, 8 layers)
    # ================================================================
    print("\n--- Perturbation 3: Layer Pruning ---")
    prune_counts = [0, 1, 2, 4, 8, 14]
    prune_results = []

    for n_prune in prune_counts:
        if n_prune == 0:
            mean_prt, std_prt, cv, _ = measure_conservation(
                model, tok, prompts, device, f"Prune {n_prune} layers")
        else:
            # Prune from middle layers
            mid = n_layers // 2
            prune_layers = list(range(mid - n_prune // 2, mid + (n_prune + 1) // 2))
            prune_layers = [l for l in prune_layers if 0 <= l < n_layers]

            handles = []
            for li in prune_layers:
                def make_identity():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            return (input[0] if isinstance(input, tuple) else input,) + output[1:]
                        return input[0] if isinstance(input, tuple) else input
                    return hook
                h = model.model.layers[li].register_forward_hook(make_identity())
                handles.append(h)

            mean_prt, std_prt, cv, _ = measure_conservation(
                model, tok, prompts, device, f"Prune {n_prune} layers")

            for h in handles:
                h.remove()

        prune_results.append({
            'n_pruned': n_prune,
            'mean_prt': mean_prt,
            'cv': cv,
        })

    # ================================================================
    # Perturbation 4: Simulated quantization
    # ================================================================
    print("\n--- Perturbation 4: Simulated Quantization ---")
    quant_bits = [32, 16, 8, 4, 2]
    quant_results = []

    for bits in quant_bits:
        handles = []
        if bits < 32:
            for li in range(n_layers):
                def make_quant_hook(b):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            # Simulate quantization
                            scale = h.abs().max() / (2 ** (b - 1) - 1 + 1e-10)
                            h_q = torch.round(h / (scale + 1e-10)) * scale
                            return (h_q,) + output[1:]
                        return output
                    return hook
                h = model.model.layers[li].register_forward_hook(make_quant_hook(bits))
                handles.append(h)

        mean_prt, std_prt, cv, _ = measure_conservation(
            model, tok, prompts, device, f"Quantize {bits}-bit")

        for h in handles:
            h.remove()

        quant_results.append({
            'bits': bits,
            'mean_prt': mean_prt,
            'cv': cv,
        })

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Noise robustness
    ax = axes[0][0]
    sigmas = [r['sigma'] for r in noise_results]
    means = [r['mean_prt'] for r in noise_results]
    ax.plot(sigmas, means, 'o-', color='#e74c3c', ms=8, lw=2)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5,
               label=f'Baseline = {baseline_mean:.1f}')
    ax.fill_between(sigmas,
                    [baseline_mean - baseline_std] * len(sigmas),
                    [baseline_mean + baseline_std] * len(sigmas),
                    alpha=0.1, color='gray')
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('PR x T')
    ax.set_title('(a) Noise Robustness')
    ax.legend()

    # (b) Temperature scaling
    ax = axes[0][1]
    temps = [r['temp_scale'] for r in temp_results]
    means_t = [r['mean_prt'] for r in temp_results]
    ax.semilogx(temps, means_t, 's-', color='#3498db', ms=8, lw=2)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Temperature Scale Factor')
    ax.set_ylabel('PR x T')
    ax.set_title('(b) Temperature Scaling')

    # (c) Layer pruning
    ax = axes[1][0]
    prunes = [r['n_pruned'] for r in prune_results]
    means_p = [r['mean_prt'] for r in prune_results]
    ax.bar(prunes, means_p, color='#2ecc71', alpha=0.8)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layers Pruned')
    ax.set_ylabel('PR x T')
    ax.set_title('(c) Layer Pruning Robustness')

    # (d) Quantization
    ax = axes[1][1]
    bits_list = [r['bits'] for r in quant_results]
    means_q = [r['mean_prt'] for r in quant_results]
    ax.bar([str(b) for b in bits_list], means_q, color='#f39c12', alpha=0.8)
    ax.axhline(y=baseline_mean, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('PR x T')
    ax.set_title('(d) Quantization Robustness')

    # Overall robustness score
    all_deviations = []
    for r in noise_results[1:]:  # Skip sigma=0
        all_deviations.append(r['deviation_from_baseline'])
    for r in prune_results[1:]:
        all_deviations.append(abs(r['mean_prt'] - baseline_mean) / baseline_mean * 100)
    for r in quant_results:
        all_deviations.append(abs(r['mean_prt'] - baseline_mean) / baseline_mean * 100)

    avg_deviation = np.mean(all_deviations)

    fig.suptitle(
        f"Phase 8: Conservation Law Stress Test\n"
        f"Baseline PR*T = {baseline_mean:.1f} +/- {baseline_std:.1f} | "
        f"Avg perturbation deviation = {avg_deviation:.1f}%",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase8_conservation_stress")
    plt.close()

    # ================================================================
    # Verdict
    # ================================================================
    if avg_deviation < 20:
        verdict = (f"GENUINE NOETHER CONSERVATION: PR*T = {baseline_mean:.1f} +/- {baseline_std:.1f} "
                   f"is robust under perturbation (avg deviation = {avg_deviation:.1f}%). "
                   f"This is a true conserved quantity.")
    elif avg_deviation < 50:
        verdict = (f"APPROXIMATE CONSERVATION: PR*T = {baseline_mean:.1f} is partially "
                   f"robust (avg deviation = {avg_deviation:.1f}%). "
                   f"Conservation holds under mild perturbation but breaks under stress.")
    else:
        verdict = (f"FRAGILE CONSERVATION: PR*T = {baseline_mean:.1f} is not robust "
                   f"(avg deviation = {avg_deviation:.1f}%). "
                   f"This may be a coincidence rather than a fundamental law.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 8: Conservation Law Stress Test',
        'summary': {
            'verdict': verdict,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_cv': baseline_cv,
            'avg_perturbation_deviation': avg_deviation,
        },
        'noise_results': noise_results,
        'temp_results': temp_results,
        'prune_results': prune_results,
        'quant_results': quant_results,
    }
    save_results("phase8_conservation_stress", result)
    return result


if __name__ == '__main__':
    main()
