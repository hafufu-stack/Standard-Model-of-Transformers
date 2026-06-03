# -*- coding: utf-8 -*-
"""
Phase 16: Attention Entropy Cascade (Opus Original)
=====================================================
Track how attention entropy evolves across layers.
Does it follow a cascade pattern similar to turbulence?
Is there an "attention energy spectrum" analogous to
Kolmogorov's -5/3 law in fluid dynamics?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 16: Attention Entropy Cascade")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    prompts = [
        "The theory of relativity revolutionized our understanding of space time and gravity",
        "Machine learning models learn patterns from data through optimization of loss functions",
        "The human brain contains approximately one hundred billion neurons connected by synapses",
        "Quantum computers use qubits that can exist in superposition of zero and one states",
        "Climate change is driven by increasing concentrations of greenhouse gases in atmosphere",
    ]

    # Collect attention entropy at each layer and head
    all_entropies = []  # (prompt, layer, head)
    all_attn_spectra = []  # SVD spectra of attention matrices

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)

        prompt_entropies = []
        prompt_spectra = []

        for layer_idx, attn_layer in enumerate(out.attentions):
            # attn_layer: (batch, heads, seq, seq)
            layer_ent = []
            layer_spec = []

            if attn_layer is None or torch.isnan(attn_layer).all():
                layer_ent = [0.0] * n_heads
                layer_spec = [np.zeros(1)] * n_heads
                prompt_entropies.append(layer_ent)
                prompt_spectra.append(layer_spec)
                continue

            for head_idx in range(n_heads):
                attn = attn_layer[0, head_idx, :, :].float()

                if torch.isnan(attn).any():
                    layer_ent.append(0.0)
                    layer_spec.append(np.zeros(1))
                    continue

                # Entropy of attention distribution for each query
                row_entropies = []
                for row in attn:
                    row_prob = row + 1e-10
                    ent = -(row_prob * torch.log(row_prob)).sum().item()
                    if not np.isnan(ent):
                        row_entropies.append(ent)
                avg_entropy = np.mean(row_entropies) if row_entropies else 0.0
                if np.isnan(avg_entropy):
                    avg_entropy = 0.0
                layer_ent.append(avg_entropy)

                # SVD spectrum of attention matrix (energy cascade)
                try:
                    U, S, V = torch.svd(attn)
                    spectrum = S.cpu().numpy()
                    layer_spec.append(spectrum)
                except Exception:
                    layer_spec.append(np.zeros(1))

            prompt_entropies.append(layer_ent)
            prompt_spectra.append(layer_spec)

        all_entropies.append(prompt_entropies)
        all_attn_spectra.append(prompt_spectra)

    # Average entropy across prompts: shape (layers, heads)
    avg_entropy = np.mean(all_entropies, axis=0)  # (layers, heads)
    avg_entropy_per_layer = np.mean(avg_entropy, axis=1)  # (layers,)

    # Per-head entropy profile
    print("\n--- Attention Entropy per Layer (averaged) ---")
    for layer_idx in range(n_layers):
        ent = avg_entropy_per_layer[layer_idx]
        bar_len = int(ent * 5) if not np.isnan(ent) else 0
        bar = "#" * bar_len
        print(f"  L{layer_idx:2d}: {ent:.4f} {bar}")

    # Fit entropy cascade: E(l) ~ l^beta
    layers = np.arange(1, n_layers + 1).astype(float)
    try:
        def power(x, a, b):
            return a * np.power(x, b)
        popt, _ = curve_fit(power, layers, avg_entropy_per_layer, p0=[1.0, -0.5], maxfev=5000)
        cascade_exp = popt[1]
        print(f"\n  Cascade exponent (entropy ~ l^beta): beta = {cascade_exp:.4f}")
    except Exception:
        cascade_exp = 0.0

    # SVD spectrum analysis: average across prompts
    avg_spectra = []
    for layer_idx in range(n_layers):
        layer_specs = []
        for prompt_idx in range(len(prompts)):
            for head_spec in all_attn_spectra[prompt_idx][layer_idx]:
                if len(head_spec) > 1:
                    layer_specs.append(head_spec[:10])  # Top 10 singular values
        if layer_specs:
            max_len = max(len(s) for s in layer_specs)
            padded = [np.pad(s, (0, max_len - len(s))) for s in layer_specs]
            avg_spectra.append(np.mean(padded, axis=0))
        else:
            avg_spectra.append(np.zeros(10))

    # Kolmogorov-like: fit S_k ~ k^(-alpha) for average spectrum
    overall_spectrum = np.mean(avg_spectra, axis=0)
    try:
        k = np.arange(1, len(overall_spectrum) + 1).astype(float)
        popt_k, _ = curve_fit(power, k, overall_spectrum + 1e-10, p0=[1.0, -1.0], maxfev=5000)
        kolmogorov_exp = popt_k[1]
        print(f"  Kolmogorov exponent (S_k ~ k^alpha): alpha = {kolmogorov_exp:.4f}")
        print(f"  (cf. Kolmogorov -5/3 = -1.667)")
    except Exception:
        kolmogorov_exp = 0.0

    # Head specialization: entropy variance across heads
    head_variance = np.var(avg_entropy, axis=1)  # (layers,)
    specialization_layers = np.where(head_variance > np.percentile(head_variance, 75))[0]

    print(f"\n  High specialization layers: {specialization_layers.tolist()}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) Entropy cascade
    ax = axes[0][0]
    ax.plot(range(n_layers), avg_entropy_per_layer, 'o-', color='#e74c3c', ms=5)
    if cascade_exp != 0:
        fit_y = power(layers, *popt)
        ax.plot(range(n_layers), fit_y, '--', color='gray',
                label=f'Fit: ~ l^{cascade_exp:.2f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Attention Entropy')
    ax.set_title(f'(a) Entropy Cascade (beta={cascade_exp:.3f})')
    ax.legend()

    # (b) Entropy heatmap (layer x head)
    ax = axes[0][1]
    im = ax.imshow(avg_entropy.T, cmap='inferno', aspect='auto')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_title('(b) Attention Entropy Map')
    plt.colorbar(im, ax=ax)

    # (c) Head specialization variance
    ax = axes[0][2]
    ax.bar(range(n_layers), head_variance, color='#3498db', alpha=0.7)
    ax.axhline(y=np.percentile(head_variance, 75), color='red', ls='--', alpha=0.5,
               label='75th percentile')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head Entropy Variance')
    ax.set_title('(c) Head Specialization')
    ax.legend()

    # (d) SVD spectrum by layer
    ax = axes[1][0]
    for layer_idx in range(0, n_layers, max(1, n_layers // 5)):
        spec = avg_spectra[layer_idx]
        ax.plot(range(len(spec)), spec, 'o-', ms=3, label=f'L{layer_idx}')
    ax.set_xlabel('Singular Value Index k')
    ax.set_ylabel('Singular Value')
    ax.set_title('(d) SVD Spectra by Layer')
    if np.any(np.array([avg_spectra[i] for i in range(0, n_layers, max(1, n_layers//5))]) > 0):
        ax.set_yscale('log')
    ax.legend(fontsize=7)

    # (e) Overall Kolmogorov spectrum
    ax = axes[1][1]
    ax.plot(range(1, len(overall_spectrum) + 1), overall_spectrum, 'o-',
            color='#e74c3c', ms=8, label='Data')
    if kolmogorov_exp != 0:
        k_fit = np.arange(1, len(overall_spectrum) + 1).astype(float)
        fit_vals = power(k_fit, *popt_k)
        ax.plot(k_fit, fit_vals, '--', color='gray',
                label=f'k^{kolmogorov_exp:.2f}')
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.set_xlabel('Mode k')
    ax.set_ylabel('Average Singular Value')
    ax.set_title(f'(e) Kolmogorov Spectrum\n(alpha={kolmogorov_exp:.3f} vs -5/3=-1.667)')
    if np.any(overall_spectrum > 0):
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend()

    # (f) Entropy gradient (dE/dl)
    ax = axes[1][2]
    gradient = np.gradient(avg_entropy_per_layer)
    ax.plot(range(n_layers), gradient, 'o-', color='#9b59b6', ms=4)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('dEntropy/dLayer')
    ax.set_title('(f) Entropy Gradient')

    fig.suptitle(
        f"Phase 16: Attention Entropy Cascade\n"
        f"Cascade beta={cascade_exp:.3f} | Kolmogorov alpha={kolmogorov_exp:.3f} "
        f"(vs -5/3)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase16_entropy_cascade")
    plt.close()

    kolm_diff = abs(kolmogorov_exp - (-5/3))
    if kolm_diff < 0.3:
        verdict = (f"KOLMOGOROV TURBULENCE ANALOGY: alpha={kolmogorov_exp:.3f} "
                   f"(diff from -5/3 = {kolm_diff:.3f}). "
                   f"Attention follows a turbulent energy cascade!")
    else:
        verdict = (f"NON-KOLMOGOROV CASCADE: alpha={kolmogorov_exp:.3f} "
                   f"(diff from -5/3 = {kolm_diff:.3f}). "
                   f"Cascade exponent beta={cascade_exp:.3f}. "
                   f"Attention has its own unique cascade dynamics.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 16: Attention Entropy Cascade',
        'summary': {'verdict': verdict, 'cascade_beta': cascade_exp,
                    'kolmogorov_alpha': kolmogorov_exp,
                    'specialization_layers': specialization_layers.tolist()},
    }
    save_results("phase16_entropy_cascade", result)
    return result


if __name__ == '__main__':
    main()
