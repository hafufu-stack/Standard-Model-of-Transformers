# -*- coding: utf-8 -*-
"""
Phase 214: Fluctuation-Dissipation Theorem (FDT)
==================================================
In thermal equilibrium: chi = sigma^2 / kT

Test whether spontaneous fluctuations (variance across prompts)
predict forced response (sensitivity to noise injection) at each layer.
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
    "Superconductors carry current without resistance",
    "The universe is expanding at an accelerating rate",
    "Mitochondria produce most of the cell energy",
    "Thermodynamics governs the direction of chemical reactions",
    "Antibiotics target specific bacterial mechanisms",
    "The Fourier transform decomposes signals into frequencies",
    "Dark matter interacts only through gravity",
    "Ribosomes translate messenger RNA into proteins",
    "Statistical mechanics connects microscopic and macroscopic behavior",
    "The double helix structure stores hereditary information",
    "Quantum entanglement correlates distant particles",
    "Cellular automata demonstrate emergent complexity",
    "The renormalization group explains scale invariance",
    "Gene expression is regulated by transcription factors",
    "Maxwell equations unify electricity and magnetism",
]

NOISE_SIGMA = 0.1


def measure_spontaneous_fluctuations(model, tok, device):
    """Measure hidden state variance across prompts at each layer."""
    n_layers = len(model.model.layers) + 1  # +1 for embedding
    all_hidden = {l: [] for l in range(n_layers)}

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float().cpu()
            all_hidden[li].append(h)

    # Variance at each layer
    variance_per_layer = []
    for li in range(n_layers):
        if all_hidden[li]:
            stacked = torch.stack(all_hidden[li])  # (N_prompts, d_model)
            var = stacked.var(dim=0).mean().item()  # Mean variance across dimensions
            variance_per_layer.append(var)
        else:
            variance_per_layer.append(0)

    return variance_per_layer


def measure_forced_response(model, tok, device):
    """Measure sensitivity to noise injection at each layer."""
    n_transformer_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    chi_per_layer = []  # Susceptibility

    for inject_layer in range(n_transformer_layers):
        kl_values = []
        for prompt in PROMPTS[:10]:  # Use subset for speed
            inp = tok(prompt, return_tensors='pt').to(device)
            input_ids = inp['input_ids']

            # Baseline
            with torch.no_grad():
                baseline_out = model(**inp)
            baseline_logits = baseline_out.logits[0, -1, :].float()
            baseline_probs = torch.softmax(baseline_logits, dim=-1)

            # Perturbed
            with torch.no_grad():
                hidden = model.model.embed_tokens(input_ids)
                seq_len = hidden.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(hidden, position_ids)

                for li in range(n_transformer_layers):
                    layer = model.model.layers[li]
                    layer_out = layer(hidden, position_embeddings=position_embeddings)
                    hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
                    if li == inject_layer:
                        noise = torch.randn_like(hidden.float()) * NOISE_SIGMA
                        hidden = (hidden.float() + noise).to(hidden.dtype)

                normed = norm_layer(hidden)
                perturbed_logits = lm_head(normed)[0, -1, :].float()
            perturbed_probs = torch.softmax(perturbed_logits, dim=-1)

            kl = (baseline_probs * torch.log(
                (baseline_probs + 1e-10) / (perturbed_probs + 1e-10)
            )).sum().item()
            kl_values.append(kl if not np.isnan(kl) else 0)

        chi = float(np.mean(kl_values)) / NOISE_SIGMA  # chi = response / field
        chi_per_layer.append(chi)

    return chi_per_layer


def measure_temperature_profile(model, tok, device):
    """Measure T at each layer (averaged across prompts)."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_T = []
    for prompt in PROMPTS[:10]:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_list = []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)
        all_T.append(T_list)

    n_hs = min(len(t) for t in all_T)
    mean_T = [float(np.mean([t[i] for t in all_T])) for i in range(n_hs)]
    return mean_T


def main():
    print("=" * 70)
    print("Phase 214: Fluctuation-Dissipation Theorem")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)
    n_layers = len(model.model.layers)

    # Step 1: Spontaneous fluctuations
    print("\n  Measuring spontaneous fluctuations (30 prompts)...")
    sigma2 = measure_spontaneous_fluctuations(model, tok, device)
    print(f"  Variance range: {min(sigma2):.4f} to {max(sigma2):.4f}")

    # Step 2: Forced response (susceptibility)
    print("\n  Measuring forced response (susceptibility)...")
    chi = measure_forced_response(model, tok, device)
    print(f"  Chi range: {min(chi):.4f} to {max(chi):.4f}")

    # Step 3: Temperature profile
    print("\n  Measuring temperature profile...")
    T_profile = measure_temperature_profile(model, tok, device)

    # FDT test: chi should correlate with sigma^2 / T
    # Align dimensions (sigma2 has n_layers+1, chi has n_layers)
    sigma2_aligned = sigma2[1:n_layers+1]  # Skip embedding, align with transformer layers
    T_aligned = T_profile[1:n_layers+1] if len(T_profile) > n_layers else T_profile[1:]

    min_len = min(len(sigma2_aligned), len(chi), len(T_aligned))
    sigma2_aligned = sigma2_aligned[:min_len]
    chi_aligned = chi[:min_len]
    T_aligned = T_aligned[:min_len]

    # FDT prediction: chi ~ sigma^2 / T
    fdt_prediction = [s / (t + 1e-10) for s, t in zip(sigma2_aligned, T_aligned)]

    # Correlation test
    r_fdt, p_fdt = stats.pearsonr(chi_aligned, fdt_prediction) if min_len > 2 else (0, 1)
    r_simple, p_simple = stats.pearsonr(chi_aligned, sigma2_aligned) if min_len > 2 else (0, 1)

    print(f"\n  FDT correlation (chi vs sigma^2/T): r={r_fdt:.4f}, p={p_fdt:.4e}")
    print(f"  Simple correlation (chi vs sigma^2): r={r_simple:.4f}, p={p_simple:.4e}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    layers = range(min_len)

    # (a) sigma^2 profile
    axes[0, 0].plot(layers, sigma2_aligned, 'o-', color='#3498db', markersize=5, lw=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Variance (sigma^2)')
    axes[0, 0].set_title('(a) Spontaneous Fluctuations')

    # (b) chi profile
    axes[0, 1].plot(layers, chi_aligned, 's-', color='#e74c3c', markersize=5, lw=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Susceptibility (chi)')
    axes[0, 1].set_title('(b) Forced Response')

    # (c) FDT scatter: chi vs sigma^2/T
    axes[0, 2].scatter(fdt_prediction, chi_aligned, color='#2ecc71', alpha=0.7, s=40)
    # Fit line
    if min_len > 2:
        slope, intercept = np.polyfit(fdt_prediction, chi_aligned, 1)
        x_fit = np.linspace(min(fdt_prediction), max(fdt_prediction), 50)
        axes[0, 2].plot(x_fit, slope * x_fit + intercept, '--', color='black', lw=2,
                        label=f'r={r_fdt:.3f}, p={p_fdt:.2e}')
    axes[0, 2].set_xlabel('FDT prediction: sigma^2 / T')
    axes[0, 2].set_ylabel('Measured chi')
    axes[0, 2].set_title('(c) FDT Test')
    axes[0, 2].legend(fontsize=8)

    # (d) Overlay: sigma^2 and chi (dual y-axis)
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    l1 = ax1.plot(layers, sigma2_aligned, '-', color='#3498db', lw=2, label='sigma^2')
    l2 = ax2.plot(layers, chi_aligned, '-', color='#e74c3c', lw=2, label='chi')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('sigma^2', color='#3498db')
    ax2.set_ylabel('chi', color='#e74c3c')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8)
    axes[1, 0].set_title('(d) Overlay')

    # (e) Temperature profile
    axes[1, 1].plot(range(len(T_profile)), T_profile, '-', color='#f39c12', lw=2)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Temperature T')
    axes[1, 1].set_title('(e) Temperature Profile')

    # (f) Summary
    fdt_status = "CONFIRMED" if r_fdt > 0.5 and p_fdt < 0.05 else (
        "SUGGESTIVE" if r_fdt > 0.3 else "NOT CONFIRMED")
    summary = (
        f"Fluctuation-Dissipation Theorem\n\n"
        f"FDT (chi vs sigma^2/T):\n"
        f"  r = {r_fdt:.4f}\n"
        f"  p = {p_fdt:.2e}\n\n"
        f"Simple (chi vs sigma^2):\n"
        f"  r = {r_simple:.4f}\n"
        f"  p = {p_simple:.2e}\n\n"
        f"Status: {fdt_status}\n"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 214: Fluctuation-Dissipation Theorem",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase214_fdt')
    plt.close()

    save_results('phase214_fdt', {
        'experiment': 'Fluctuation-Dissipation Theorem',
        'r_fdt': r_fdt, 'p_fdt': p_fdt,
        'r_simple': r_simple, 'p_simple': p_simple,
        'sigma2': [float(x) for x in sigma2_aligned],
        'chi': [float(x) for x in chi_aligned],
        'fdt_prediction': [float(x) for x in fdt_prediction],
        'T_profile': [float(x) for x in T_profile],
    })


if __name__ == '__main__':
    main()
