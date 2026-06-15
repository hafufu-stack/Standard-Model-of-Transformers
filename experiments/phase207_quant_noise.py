# -*- coding: utf-8 -*-
"""
Phase 207: Quantization Noise as Free Energy
==============================================
4-bit/8-bit quantization adds persistent noise to hidden states.
Instead of treating this as degradation, we tune the quantization
scale so the rounding error RMS matches the optimal stochastic
resonance amplitude (sigma ~ 0.15 from Phase 179).

If quantization noise IS the stochastic resonance fuel, then a
properly-tuned quantized model may OUTPERFORM the full-precision one.
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
]

# Simulate quantization at different bit widths
QUANT_BITS = [16, 8, 4, 3, 2]
# Sigma-tuned: adjust Q-scale so quantization noise RMS = target sigma
TARGET_SIGMAS = [None, 0.05, 0.10, 0.15, 0.20, 0.30]


def make_quant_hook(bits=None, target_sigma=None):
    """Hook that simulates quantization on hidden states.
    
    If target_sigma is set, overrides bits to achieve desired noise RMS.
    """
    quant_errors = []

    def hook(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        h_fp32 = h.float()

        if target_sigma is not None:
            # Inject noise with exact sigma (bypass quantization math)
            noise = torch.randn_like(h_fp32) * target_sigma
            h_mod = h_fp32 + noise
        elif bits is not None and bits < 16:
            # Simulate min-max quantization per-tensor
            h_min = h_fp32.min()
            h_max = h_fp32.max()
            span = h_max - h_min
            if span < 1e-8:
                return output
            n_levels = 2 ** bits
            scale = span / (n_levels - 1)
            h_q = torch.round((h_fp32 - h_min) / scale) * scale + h_min
            error_rms = (h_q - h_fp32).pow(2).mean().sqrt().item()
            quant_errors.append(error_rms)
            h_mod = h_q
        else:
            return output

        h_mod = torch.nan_to_num(h_mod, nan=0.0, posinf=65000.0, neginf=-65000.0)
        result = h_mod.to(h.dtype)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    hook.quant_errors = quant_errors
    return hook


def run_with_quantization(model, tok, device, prompt, bits=None, target_sigma=None):
    """Forward pass with simulated hidden-state quantization."""
    hooks_list = []
    handles = []

    # Install hooks on each layer's output
    for li, layer in enumerate(model.model.layers):
        qh = make_quant_hook(bits=bits, target_sigma=target_sigma)
        handle = layer.register_forward_hook(qh)
        hooks_list.append(qh)
        handles.append(handle)

    try:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure output quality
        norm_layer = model.model.norm
        lm_head = model.lm_head

        # Temperature profile
        T_list = []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)

        T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
        T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
        eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

        final_logits = out.logits[0, -1, :].float()
        final_probs = torch.softmax(final_logits, dim=-1)
        output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
        top1_prob = final_probs.max().item()
        top_token = tok.decode(final_logits.argmax().item())

        # Collect quantization error stats
        all_errors = []
        for qh in hooks_list:
            all_errors.extend(qh.quant_errors)
        mean_error = float(np.mean(all_errors)) if all_errors else 0.0

    finally:
        for h in handles:
            h.remove()

    return {
        'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
        'output_entropy': output_entropy, 'top1_prob': top1_prob,
        'top_token': top_token,
        'mean_quant_error': mean_error,
        'T': T_list,
    }


def main():
    print("=" * 70)
    print("Phase 207: Quantization Noise as Free Energy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    # Part 1: Standard quantization at various bit widths
    print("\n=== Part 1: Standard Quantization ===")
    quant_results = {}
    for bits in QUANT_BITS:
        label = f"{bits}bit"
        b = None if bits == 16 else bits
        all_eta, all_ent, all_top1, all_err = [], [], [], []
        for prompt in PROMPTS:
            r = run_with_quantization(model, tok, device, prompt, bits=b)
            all_eta.append(r['eta'])
            all_ent.append(r['output_entropy'])
            all_top1.append(r['top1_prob'])
            all_err.append(r['mean_quant_error'])

        quant_results[label] = {
            'bits': bits,
            'eta_mean': float(np.mean(all_eta)),
            'entropy_mean': float(np.mean(all_ent)),
            'top1_mean': float(np.mean(all_top1)),
            'error_mean': float(np.mean(all_err)),
        }
        print(f"  {label}: eta={np.mean(all_eta):.4f}, "
              f"entropy={np.mean(all_ent):.3f}, "
              f"top1={np.mean(all_top1):.4f}, "
              f"quant_err={np.mean(all_err):.4f}")

    # Part 2: Sigma-tuned quantization
    print("\n=== Part 2: Sigma-Tuned Quantization ===")
    sigma_results = {}
    for sigma in TARGET_SIGMAS:
        label = f"sigma_{sigma}" if sigma else "baseline_fp16"
        all_eta, all_ent, all_top1 = [], [], []
        for prompt in PROMPTS:
            r = run_with_quantization(model, tok, device, prompt, target_sigma=sigma)
            all_eta.append(r['eta'])
            all_ent.append(r['output_entropy'])
            all_top1.append(r['top1_prob'])

        sigma_results[label] = {
            'sigma': sigma if sigma else 0,
            'eta_mean': float(np.mean(all_eta)),
            'entropy_mean': float(np.mean(all_ent)),
            'top1_mean': float(np.mean(all_top1)),
        }
        s_str = f"sigma={sigma}" if sigma else "fp16 baseline"
        print(f"  {s_str}: eta={np.mean(all_eta):.4f}, "
              f"entropy={np.mean(all_ent):.3f}, top1={np.mean(all_top1):.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Entropy vs bits
    bits_list = [r['bits'] for r in quant_results.values()]
    ents_bits = [r['entropy_mean'] for r in quant_results.values()]
    axes[0, 0].plot(range(len(bits_list)), ents_bits, 'o-', color='#e74c3c',
                    markersize=8, lw=2)
    axes[0, 0].set_xticks(range(len(bits_list)))
    axes[0, 0].set_xticklabels([str(b) for b in bits_list])
    axes[0, 0].set_xlabel('Quantization Bits')
    axes[0, 0].set_ylabel('Output Entropy (nats)')
    axes[0, 0].set_title('(a) Quality vs Quantization Level')

    # (b) eta vs bits
    etas_bits = [r['eta_mean'] for r in quant_results.values()]
    axes[0, 1].plot(range(len(bits_list)), etas_bits, 's-', color='#3498db',
                    markersize=8, lw=2)
    axes[0, 1].set_xticks(range(len(bits_list)))
    axes[0, 1].set_xticklabels([str(b) for b in bits_list])
    axes[0, 1].set_xlabel('Quantization Bits')
    axes[0, 1].set_ylabel('Carnot Efficiency eta')
    axes[0, 1].set_title('(b) Efficiency vs Quantization')

    # (c) Quant error vs bits
    errs_bits = [r['error_mean'] for r in quant_results.values()]
    axes[0, 2].bar(range(len(bits_list)), errs_bits, color='#e67e22', alpha=0.7)
    axes[0, 2].set_xticks(range(len(bits_list)))
    axes[0, 2].set_xticklabels([str(b) for b in bits_list])
    axes[0, 2].set_xlabel('Quantization Bits')
    axes[0, 2].set_ylabel('Mean Quant Error (RMS)')
    axes[0, 2].axhline(y=0.15, color='#e74c3c', linestyle='--', label='optimal sigma')
    axes[0, 2].set_title('(c) Quantization Error vs Bits')
    axes[0, 2].legend(fontsize=8)

    # (d) Entropy vs target sigma (sigma-tuned)
    sigmas = [r['sigma'] for r in sigma_results.values()]
    ents_sig = [r['entropy_mean'] for r in sigma_results.values()]
    axes[1, 0].plot(range(len(sigmas)), ents_sig, 'D-', color='#9b59b6',
                    markersize=8, lw=2)
    axes[1, 0].set_xticks(range(len(sigmas)))
    axes[1, 0].set_xticklabels([str(s) for s in sigmas], fontsize=8)
    axes[1, 0].set_xlabel('Target Sigma')
    axes[1, 0].set_ylabel('Output Entropy (nats)')
    axes[1, 0].set_title('(d) Sigma-Tuned: Quality vs Noise Level')

    # (e) Top-1 prob vs sigma
    top1s_sig = [r['top1_mean'] for r in sigma_results.values()]
    axes[1, 1].plot(range(len(sigmas)), top1s_sig, '^-', color='#2ecc71',
                    markersize=8, lw=2)
    axes[1, 1].set_xticks(range(len(sigmas)))
    axes[1, 1].set_xticklabels([str(s) for s in sigmas], fontsize=8)
    axes[1, 1].set_xlabel('Target Sigma')
    axes[1, 1].set_ylabel('Top-1 Probability')
    axes[1, 1].set_title('(e) Confidence vs Sigma')

    # (f) Summary
    base_16 = quant_results.get('16bit', {})
    q4 = quant_results.get('4bit', {})
    sig015 = sigma_results.get('sigma_0.15', {})
    summary_text = (
        f"Quantization Noise as Free Energy\n\n"
        f"FP16 baseline:\n"
        f"  entropy = {base_16.get('entropy_mean', 0):.3f}\n"
        f"  top1 = {base_16.get('top1_mean', 0):.4f}\n\n"
        f"4-bit quant:\n"
        f"  entropy = {q4.get('entropy_mean', 0):.3f}\n"
        f"  noise RMS = {q4.get('error_mean', 0):.4f}\n\n"
        f"Sigma-tuned (0.15):\n"
        f"  entropy = {sig015.get('entropy_mean', 0):.3f}\n"
        f"  top1 = {sig015.get('top1_mean', 0):.4f}\n"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 207: Quantization Noise as Free Energy",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase207_quant_noise')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"FP16 entropy: {base_16.get('entropy_mean', 0):.3f}")
    print(f"4-bit entropy: {q4.get('entropy_mean', 0):.3f}")
    if sig015:
        print(f"Sigma=0.15 entropy: {sig015.get('entropy_mean', 0):.3f}")
    print(f"{'=' * 70}")

    save_results('phase207_quant_noise', {
        'experiment': 'Quantization Noise as Free Energy',
        'standard_quant': quant_results,
        'sigma_tuned': sigma_results,
    })


if __name__ == '__main__':
    main()
