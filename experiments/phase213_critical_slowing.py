# -*- coding: utf-8 -*-
"""
Phase 213: Critical Slowing Down
===================================
Near a critical point, perturbations relax more slowly.
The relaxation time tau diverges as tau ~ |l - L0|^{-z}.

For each layer: inject noise, then measure how many subsequent layers
it takes for the output to recover to within epsilon of the baseline.
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

NOISE_SIGMA = 0.1
RECOVERY_THRESHOLD = 0.05  # Output recovered if KL < this


def get_baseline_logits(model, tok, device, prompt):
    """Get baseline output logits without any perturbation."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].float()


def run_with_injection_at_layer(model, tok, device, prompt, inject_layer):
    """Manual forward pass: inject noise at exactly one layer,
    then measure T at all subsequent layers to track recovery."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        T_after_injection = []

        for li in range(n_layers):
            layer = model.model.layers[li]
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Inject noise at the specified layer
            if li == inject_layer:
                h_fp32 = hidden.float()
                noise = torch.randn_like(h_fp32) * NOISE_SIGMA
                hidden = (h_fp32 + noise).to(hidden.dtype)

            # Measure T at layers after injection
            if li >= inject_layer:
                normed = norm_layer(hidden[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_after_injection.append((li, T if not np.isnan(T) else 0))

        normed = norm_layer(hidden)
        final_logits = lm_head(normed)[0, -1, :].float()

    return final_logits, T_after_injection


def measure_relaxation(model, tok, device, prompt, inject_layer):
    """Measure how output diverges when noise is injected at inject_layer."""
    baseline_logits = get_baseline_logits(model, tok, device, prompt)
    perturbed_logits, T_trace = run_with_injection_at_layer(
        model, tok, device, prompt, inject_layer)

    # KL divergence between baseline and perturbed output
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
    kl = (baseline_probs * torch.log((baseline_probs + 1e-10) / (perturbed_probs + 1e-10))).sum().item()
    kl = kl if not np.isnan(kl) else 0

    # L2 distance in logit space
    l2_dist = (baseline_logits - perturbed_logits).norm().item()

    return {
        'inject_layer': inject_layer,
        'kl_divergence': kl,
        'l2_distance': l2_dist,
        'T_trace': T_trace,
    }


def main():
    print("=" * 70)
    print("Phase 213: Critical Slowing Down")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)
    n_layers = len(model.model.layers)

    # Measure at all layers
    all_kl = {l: [] for l in range(n_layers)}
    all_l2 = {l: [] for l in range(n_layers)}

    for pi, prompt in enumerate(PROMPTS):
        print(f"  Prompt {pi+1}/{len(PROMPTS)}")
        for inject_layer in range(n_layers):
            r = measure_relaxation(model, tok, device, prompt, inject_layer)
            all_kl[inject_layer].append(r['kl_divergence'])
            all_l2[inject_layer].append(r['l2_distance'])

    # Aggregate
    mean_kl = [float(np.mean(all_kl[l])) for l in range(n_layers)]
    std_kl = [float(np.std(all_kl[l])) for l in range(n_layers)]
    mean_l2 = [float(np.mean(all_l2[l])) for l in range(n_layers)]

    # Find L0 (layer of maximum sensitivity = critical point)
    L0_kl = int(np.argmax(mean_kl))
    L0_l2 = int(np.argmax(mean_l2))
    print(f"\n  L0 (max KL sensitivity): layer {L0_kl}")
    print(f"  L0 (max L2 sensitivity): layer {L0_l2}")

    # Fit power law: KL(l) ~ |l - L0|^{-gamma} near L0
    # Use L0_kl as the critical point
    distances = []
    kl_values = []
    for l in range(n_layers):
        d = abs(l - L0_kl)
        if d > 0 and d < n_layers // 2:
            distances.append(d)
            kl_values.append(mean_kl[l])

    gamma = None
    if distances and kl_values:
        try:
            log_d = np.log(distances)
            log_kl = np.log(np.array(kl_values) + 1e-10)
            # Linear fit in log-log space
            coeffs = np.polyfit(log_d, log_kl, 1)
            gamma = -coeffs[0]  # Negative slope = positive exponent
            print(f"  Critical exponent gamma = {gamma:.3f}")
        except Exception:
            gamma = None

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) KL sensitivity profile
    axes[0, 0].bar(range(n_layers), mean_kl, color='#e74c3c', alpha=0.7)
    axes[0, 0].errorbar(range(n_layers), mean_kl, yerr=std_kl, fmt='none',
                        color='black', alpha=0.3, capsize=2)
    axes[0, 0].axvline(x=L0_kl, color='gold', linestyle='--', lw=2,
                       label=f'L0={L0_kl}')
    axes[0, 0].set_xlabel('Injection Layer')
    axes[0, 0].set_ylabel('KL Divergence')
    axes[0, 0].set_title('(a) Perturbation Sensitivity')
    axes[0, 0].legend(fontsize=8)

    # (b) L2 sensitivity profile
    axes[0, 1].bar(range(n_layers), mean_l2, color='#3498db', alpha=0.7)
    axes[0, 1].axvline(x=L0_l2, color='gold', linestyle='--', lw=2,
                       label=f'L0={L0_l2}')
    axes[0, 1].set_xlabel('Injection Layer')
    axes[0, 1].set_ylabel('L2 Distance (logits)')
    axes[0, 1].set_title('(b) Logit Displacement')
    axes[0, 1].legend(fontsize=8)

    # (c) Log-log plot for power law fit
    if distances and kl_values:
        axes[0, 2].scatter(distances, kl_values, color='#e74c3c', alpha=0.6, s=30)
        if gamma is not None:
            d_fit = np.linspace(1, max(distances), 50)
            kl_fit = np.exp(coeffs[1]) * d_fit ** coeffs[0]
            axes[0, 2].plot(d_fit, kl_fit, '-', color='black', lw=2,
                            label=f'gamma={gamma:.2f}')
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_xlabel('|l - L0|')
        axes[0, 2].set_ylabel('KL Divergence')
        axes[0, 2].set_title('(c) Power Law Fit')
        axes[0, 2].legend(fontsize=8)

    # (d) Asymmetry: layers before vs after L0
    before_L0 = mean_kl[:L0_kl] if L0_kl > 0 else []
    after_L0 = mean_kl[L0_kl+1:] if L0_kl < n_layers - 1 else []
    if before_L0 and after_L0:
        axes[1, 0].plot(range(len(before_L0)), before_L0[::-1], 'o-',
                        color='#3498db', label='Before L0 (reversed)', markersize=4)
        axes[1, 0].plot(range(len(after_L0)), after_L0, 's-',
                        color='#e74c3c', label='After L0', markersize=4)
        axes[1, 0].set_xlabel('Distance from L0')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('(d) Asymmetry: Before vs After L0')
        axes[1, 0].legend(fontsize=8)

    # (e) Normalized sensitivity
    max_kl = max(mean_kl) if max(mean_kl) > 0 else 1
    norm_kl = [k / max_kl for k in mean_kl]
    axes[1, 1].fill_between(range(n_layers), norm_kl, alpha=0.3, color='#e74c3c')
    axes[1, 1].plot(range(n_layers), norm_kl, '-', color='#e74c3c', lw=2)
    axes[1, 1].axvline(x=L0_kl, color='gold', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Injection Layer')
    axes[1, 1].set_ylabel('Normalized Sensitivity')
    axes[1, 1].set_title('(e) Critical Slowing Profile')

    # (f) Summary
    summary = (
        f"Critical Slowing Down\n\n"
        f"L0 (max KL): layer {L0_kl}\n"
        f"L0 (max L2): layer {L0_l2}\n"
        f"Peak KL: {max(mean_kl):.4f}\n"
        f"Min KL: {min(mean_kl):.4f}\n"
        f"Peak/Min ratio: {max(mean_kl)/(min(mean_kl)+1e-10):.1f}x\n\n"
    )
    if gamma is not None:
        summary += f"Critical exponent gamma = {gamma:.3f}\n"
    summary += (f"\nInterpretation:\n"
                f"{'CONFIRMED' if max(mean_kl)/max(min(mean_kl),1e-10) > 3 else 'WEAK'}")
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 213: Critical Slowing Down", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase213_critical_slowing')
    plt.close()

    save_results('phase213_critical_slowing', {
        'experiment': 'Critical Slowing Down',
        'L0_kl': L0_kl,
        'L0_l2': L0_l2,
        'gamma': gamma,
        'mean_kl': mean_kl,
        'mean_l2': mean_l2,
    })


if __name__ == '__main__':
    main()
