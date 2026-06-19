# -*- coding: utf-8 -*-
"""
Phase 303: Anomalous Dimensions -- Quantum Corrections
========================================================
In QFT, anomalous dimensions gamma_O tell how operator scaling deviates
from its classical (engineering) dimension due to quantum corrections.
For the transformer:
  - Classical dimension: how singular values should scale by dimension counting
  - Anomalous dimension: actual deviation from naive scaling
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
    "The chemical composition of water molecules is",
    "Artificial intelligence will transform how we live and work",
]


def measure_anomalous_dimensions(model, tok, prompts, device):
    """Measure anomalous dimensions at each layer."""
    inp = tok(prompts[0], return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    D = model.config.hidden_size

    # Classical dimension: s_n ~ n^(-1) for random matrix (Marchenko-Pastur)
    # Anomalous dimension: actual exponent minus classical
    layer_anomalous = []
    layer_spectral_exp = []
    layer_eff_dim = []

    for li in range(n_layers + 1):
        # Collect over prompts
        spectra = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[li][0].float().cpu().numpy()
            _, s, _ = np.linalg.svd(h, full_matrices=False)
            s_norm = s / (s[0] + 1e-10)
            spectra.append(s_norm)

        # Truncate to minimum length across prompts
        min_len = min(len(s) for s in spectra)
        min_len = min(min_len, 20)
        spectra_trunc = [s[:min_len] for s in spectra]
        avg_spec = np.mean(spectra_trunc, axis=0)

        # Fit: s_n ~ n^(-alpha)
        n_modes = len(avg_spec)
        log_n = np.log(np.arange(1, n_modes + 1, dtype=float))
        log_s = np.log(avg_spec + 1e-15)

        slope, _, r, _, _ = stats.linregress(log_n, log_s)
        alpha = -slope  # spectral decay exponent
        layer_spectral_exp.append(alpha)

        # Classical dimension (random matrix): alpha_classical ~ 1.0
        # Anomalous dimension = alpha - alpha_classical
        gamma_anomalous = alpha - 1.0
        layer_anomalous.append(gamma_anomalous)

        # Effective dimension
        s_p = avg_spec / (avg_spec.sum() + 1e-10)
        eff = np.exp(-np.sum(s_p * np.log(s_p + 1e-15)))
        layer_eff_dim.append(float(eff))

    # Running of anomalous dimension
    # gamma(l) should evolve along RG flow
    gamma_arr = np.array(layer_anomalous)
    # UV anomalous dim
    uv_gamma = float(np.mean(gamma_arr[:3]))
    ir_gamma = float(np.mean(gamma_arr[-3:]))

    return {
        'spectral_exponents': [round(a, 4) for a in layer_spectral_exp],
        'anomalous_dims': [round(g, 4) for g in layer_anomalous],
        'eff_dims': [round(d, 4) for d in layer_eff_dim],
        'uv_gamma': round(uv_gamma, 4),
        'ir_gamma': round(ir_gamma, 4),
        'mean_gamma': round(float(np.mean(layer_anomalous)), 4),
        'gamma_std': round(float(np.std(layer_anomalous)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 303: Anomalous Dimensions")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        result = measure_anomalous_dimensions(model, tok, PROMPTS, device)
        all_results[size] = result

        print(f"  UV gamma = {result['uv_gamma']:.4f}")
        print(f"  IR gamma = {result['ir_gamma']:.4f}")
        print(f"  Mean gamma = {result['mean_gamma']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Anomalous dimensions by layer
    for size, data in all_results.items():
        axes[0, 0].plot(data['anomalous_dims'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(0, color='gold', ls='--', lw=2, label='gamma=0 (classical)')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Anomalous Dimension gamma')
    axes[0, 0].set_title('(a) Anomalous Dimension Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Spectral exponent by layer
    for size, data in all_results.items():
        axes[0, 1].plot(data['spectral_exponents'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(1.0, color='gold', ls='--', lw=2, label='alpha=1 (classical)')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Spectral Exponent alpha')
    axes[0, 1].set_title('(b) Spectral Decay Exponent', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Effective dimension
    for size, data in all_results.items():
        axes[0, 2].plot(data['eff_dims'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Effective Dimension')
    axes[0, 2].set_title('(c) Effective Dimension', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) UV vs IR anomalous dim
    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.35
    axes[1, 0].bar(x - w/2, [all_results[s]['uv_gamma'] for s in sizes], w,
                  label='UV (early)', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['ir_gamma'] for s in sizes], w,
                  label='IR (late)', color='#e74c3c')
    axes[1, 0].axhline(0, color='gold', ls='--', lw=1)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_ylabel('Anomalous Dimension')
    axes[1, 0].set_title('(d) UV vs IR', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Normalized depth
    for size, data in all_results.items():
        n = len(data['anomalous_dims'])
        x_norm = np.linspace(0, 1, n)
        axes[1, 1].plot(x_norm, data['anomalous_dims'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].axhline(0, color='gold', ls='--', lw=1)
    axes[1, 1].set_xlabel('Normalized Depth')
    axes[1, 1].set_ylabel('gamma')
    axes[1, 1].set_title('(e) Anomalous Dim vs Depth', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "ANOMALOUS DIMENSIONS\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  UV gamma = {d['uv_gamma']:.3f}\n"
        txt += f"  IR gamma = {d['ir_gamma']:.3f}\n\n"
    txt += "gamma > 0: relevant operator\n"
    txt += "gamma < 0: irrelevant operator\n"
    txt += "gamma = 0: marginal (CFT)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 303: Anomalous Dimensions -- Quantum Corrections",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase303_anomalous_dims')
    plt.close()

    save_results('phase303_anomalous_dims', {
        'experiment': 'Anomalous Dimensions',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
