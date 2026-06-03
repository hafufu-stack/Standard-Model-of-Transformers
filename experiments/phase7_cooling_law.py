# -*- coding: utf-8 -*-
"""
Phase 7: Cooling Law Analytical Derivation
============================================
Attempt to derive T ~ l^0.67 from first principles.
Hypothesis: 0.67 ~ 2/3 arises from the interaction between
LayerNorm, Attention, and FFN in representation space.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from utils import load_model, get_hidden_states, save_results, save_figure


def power_law(x, a, b):
    return a * np.power(x, b)


def main():
    print("=" * 70)
    print("Phase 7: Cooling Law Analytical Derivation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # ================================================================
    # Step 1: Measure detailed thermodynamic quantities per layer
    # ================================================================
    print("\n--- Step 1: Detailed thermodynamic measurement ---")

    prompts = [
        "The fundamental nature of reality",
        "Quantum mechanics describes the behavior of",
        "In deep neural networks the gradient",
        "The temperature of a black hole is",
        "Statistical mechanics connects microscopic",
        "The arrow of time points from",
        "Information theory quantifies the",
        "Entropy always increases in",
        "The Boltzmann distribution gives the probability",
        "Phase transitions occur when a system",
    ]

    all_norms = []
    all_variances = []
    all_entropies = []
    all_kurtosis = []
    all_layernorm_scales = []

    for prompt in prompts:
        hs = get_hidden_states(model, tok, prompt, device=device)

        norms = [h.norm().item() for h in hs]
        variances = [h.var().item() for h in hs]

        # Shannon entropy of squared hidden state (as pseudo-distribution)
        entropies = []
        kurtoses = []
        for h in hs:
            h_sq = h ** 2
            p = h_sq / (h_sq.sum() + 1e-10)
            ent = -(p * torch.log(p + 1e-10)).sum().item()
            entropies.append(ent)

            # Kurtosis (measure of distribution shape)
            kurt = ((h - h.mean()) ** 4).mean().item() / (h.var().item() ** 2 + 1e-10) - 3
            kurtoses.append(kurt)

        all_norms.append(norms)
        all_variances.append(variances)
        all_entropies.append(entropies)
        all_kurtosis.append(kurtoses)

    avg_norms = np.mean(all_norms, axis=0)
    avg_vars = np.mean(all_variances, axis=0)
    avg_ents = np.mean(all_entropies, axis=0)
    avg_kurt = np.mean(all_kurtosis, axis=0)

    # ================================================================
    # Step 2: Extract LayerNorm parameters
    # ================================================================
    print("\n--- Step 2: LayerNorm parameter analysis ---")
    ln_weights = []
    ln_biases = []

    for li in range(n_layers):
        layer = model.model.layers[li]
        # Input LayerNorm
        if hasattr(layer, 'input_layernorm'):
            w = layer.input_layernorm.weight.detach().cpu().float()
            ln_weights.append(w.norm().item())
        # Post-attention LayerNorm
        if hasattr(layer, 'post_attention_layernorm'):
            w = layer.post_attention_layernorm.weight.detach().cpu().float()

    # ================================================================
    # Step 3: Attention weight analysis
    # ================================================================
    print("\n--- Step 3: Attention weight statistics ---")
    attn_norms = []
    ffn_norms = []

    for li in range(n_layers):
        layer = model.model.layers[li]
        # Attention projection norms
        q_norm = layer.self_attn.q_proj.weight.detach().cpu().float().norm().item()
        k_norm = layer.self_attn.k_proj.weight.detach().cpu().float().norm().item()
        v_norm = layer.self_attn.v_proj.weight.detach().cpu().float().norm().item()
        o_norm = layer.self_attn.o_proj.weight.detach().cpu().float().norm().item()
        attn_norms.append((q_norm + k_norm + v_norm + o_norm) / 4)

        # FFN norms
        if hasattr(layer.mlp, 'gate_proj'):
            gate = layer.mlp.gate_proj.weight.detach().cpu().float().norm().item()
            up = layer.mlp.up_proj.weight.detach().cpu().float().norm().item()
            down = layer.mlp.down_proj.weight.detach().cpu().float().norm().item()
            ffn_norms.append((gate + up + down) / 3)
        else:
            ffn_norms.append(0)

    # ================================================================
    # Step 4: Fit cooling law and test theoretical predictions
    # ================================================================
    print("\n--- Step 4: Cooling law fits ---")

    layers = np.arange(1, len(avg_norms) + 1).astype(float)

    # Fit T ~ l^alpha for norms
    popt_norm, _ = curve_fit(power_law, layers, avg_norms, p0=[avg_norms[0], -0.5], maxfev=5000)
    alpha_norm = popt_norm[1]
    pred_norm = power_law(layers, *popt_norm)

    # Fit for variance
    popt_var, _ = curve_fit(power_law, layers, avg_vars, p0=[avg_vars[0], -1.0], maxfev=5000)
    alpha_var = popt_var[1]

    # Fit for entropy
    try:
        popt_ent, _ = curve_fit(power_law, layers, avg_ents, p0=[avg_ents[0], -0.1], maxfev=5000)
        alpha_ent = popt_ent[1]
    except Exception:
        alpha_ent = 0.0

    print(f"  ||h|| ~ l^{alpha_norm:.4f}")
    print(f"  Var(h) ~ l^{alpha_var:.4f}")
    print(f"  S(h) ~ l^{alpha_ent:.4f}")

    # Theoretical prediction: if T = ||h||, then Var(h) = T^2/d
    # So alpha_var should be ~ 2 * alpha_norm
    predicted_alpha_var = 2 * alpha_norm
    print(f"\n  Theory check: alpha_var = 2 * alpha_norm?")
    print(f"    Predicted: {predicted_alpha_var:.4f}, Actual: {alpha_var:.4f}")
    print(f"    Ratio: {alpha_var / predicted_alpha_var:.3f}")

    # Is alpha = 2/3?
    deviation_from_2_3 = abs(alpha_norm - (-2/3))
    print(f"\n  Is alpha = -2/3?")
    print(f"    alpha = {alpha_norm:.6f}, -2/3 = {-2/3:.6f}")
    print(f"    Deviation: {deviation_from_2_3:.6f}")

    # ================================================================
    # Step 5: Correlation between weight norms and temperature
    # ================================================================
    print("\n--- Step 5: Weight-Temperature correlations ---")
    r_attn, p_attn = pearsonr(attn_norms, avg_norms[1:])
    r_ffn, p_ffn = pearsonr(ffn_norms, avg_norms[1:])
    print(f"  Attention weight norm vs Temperature: r={r_attn:.4f}, p={p_attn:.4e}")
    print(f"  FFN weight norm vs Temperature:       r={r_ffn:.4f}, p={p_ffn:.4e}")

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) Cooling law (log-log)
    ax = axes[0][0]
    ax.loglog(layers, avg_norms, 'o', color='#e74c3c', ms=6, label='Data')
    ax.loglog(layers, pred_norm, '--', color='black', lw=2,
              label=f'Fit: T ~ l^{{{alpha_norm:.3f}}}')
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.set_xlabel('Layer (log)')
    ax.set_ylabel('Temperature ||h|| (log)')
    ax.set_title(f'(a) Cooling Law: alpha = {alpha_norm:.4f}')
    ax.legend()

    # (b) Variance scaling
    ax = axes[0][1]
    ax.semilogy(layers, avg_vars, 'o-', color='#3498db', ms=5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Var(h)')
    ax.set_title(f'(b) Variance: alpha_var = {alpha_var:.4f}\n'
                 f'Theory: 2*alpha_norm = {predicted_alpha_var:.4f}')

    # (c) Entropy profile
    ax = axes[0][2]
    ax.plot(layers, avg_ents, 'o-', color='#2ecc71', ms=5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Shannon Entropy S(h)')
    ax.set_title(f'(c) Information Entropy: alpha_S = {alpha_ent:.4f}')

    # (d) Kurtosis
    ax = axes[1][0]
    ax.plot(layers, avg_kurt, 'o-', color='#9b59b6', ms=5)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5, label='Gaussian (kurt=0)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Excess Kurtosis')
    ax.set_title('(d) Distribution Shape (Kurtosis)')
    ax.legend()

    # (e) Weight norms vs temperature
    ax = axes[1][1]
    ax.scatter(attn_norms, avg_norms[1:], color='#e74c3c', label=f'Attn (r={r_attn:.3f})', s=40)
    ax.scatter(ffn_norms, avg_norms[1:], color='#3498db', label=f'FFN (r={r_ffn:.3f})', s=40)
    ax.set_xlabel('Weight Norm')
    ax.set_ylabel('Temperature')
    ax.set_title('(e) Weight Norm vs Temperature')
    ax.legend()

    # (f) LayerNorm weights
    ax = axes[1][2]
    ax.plot(range(len(ln_weights)), ln_weights, 'o-', color='#f39c12', ms=5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('LayerNorm Weight Norm')
    ax.set_title('(f) LayerNorm Scale Parameters')

    fig.suptitle(
        f"Phase 7: Cooling Law Derivation\n"
        f"alpha = {alpha_norm:.4f} (vs -2/3 = {-2/3:.4f}, deviation = {deviation_from_2_3:.4f})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase7_cooling_law_derivation")
    plt.close()

    # Verdict
    if deviation_from_2_3 < 0.05:
        verdict = (f"ALPHA = -2/3 CONFIRMED: alpha={alpha_norm:.4f}, "
                   f"deviation from -2/3 = {deviation_from_2_3:.4f}. "
                   f"Cooling law likely arises from 3D diffusion in representation space.")
    else:
        verdict = (f"ALPHA != -2/3: alpha={alpha_norm:.4f}, "
                   f"deviation from -2/3 = {deviation_from_2_3:.4f}. "
                   f"Cooling law has non-trivial origin (possibly attention geometry).")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 7: Cooling Law Derivation',
        'summary': {
            'verdict': verdict,
            'alpha_norm': alpha_norm,
            'alpha_var': alpha_var,
            'alpha_entropy': alpha_ent,
            'deviation_from_2_3': deviation_from_2_3,
            'attn_temp_correlation': r_attn,
            'ffn_temp_correlation': r_ffn,
        },
    }
    save_results("phase7_cooling_law_derivation", result)
    return result


if __name__ == '__main__':
    main()
