# -*- coding: utf-8 -*-
"""
Phase 245: Thermodynamic Fluctuation Spectrum
================================================
Measure the full fluctuation spectrum of thermodynamic variables.
Power spectral density of T(l) reveals characteristic frequencies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Chemical reactions follow conservation of mass",
    "Algorithms determine computational complexity",
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
    "Plate tectonics shapes the Earth surface",
    "Superconductors have zero electrical resistance",
    "Photosynthesis converts sunlight to energy",
    "The Higgs boson gives particles their mass",
]


def fluctuation_spectrum(model, tok, device, model_name):
    """Compute power spectral density and fluctuation statistics."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_T, all_P1, all_U = [], [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, P1_l, U_l = [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
        all_T.append(T_l); all_P1.append(P1_l); all_U.append(U_l)

    n = min(len(t) for t in all_T)
    avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
    mean_T = np.array(avg(all_T))
    mean_P1 = np.array(avg(all_P1))
    mean_U = np.array(avg(all_U))

    # Power spectral density
    def compute_psd(x):
        x_detrend = x - np.mean(x)
        freqs, psd = signal.periodogram(x_detrend)
        return freqs, psd

    freq_T, psd_T = compute_psd(mean_T)
    freq_P1, psd_P1 = compute_psd(mean_P1)
    freq_U, psd_U = compute_psd(mean_U)

    # 1/f noise test: fit log(PSD) ~ alpha * log(freq)
    mask = freq_T > 0
    if mask.sum() >= 3:
        log_f = np.log(freq_T[mask])
        log_psd = np.log(psd_T[mask] + 1e-20)
        alpha_T, _, r_alpha, _, _ = stats.linregress(log_f, log_psd)
    else:
        alpha_T, r_alpha = 0, 0

    # Fluctuation-dissipation: var(T) at each layer across prompts
    T_var_per_layer = [float(np.var([all_T[p][l] for p in range(len(PROMPTS))])) for l in range(n)]
    T_mean_per_layer = [float(np.mean([all_T[p][l] for p in range(len(PROMPTS))])) for l in range(n)]
    # FDT: var(T) proportional to T^2 * C (susceptibility)
    r_fdt, _ = stats.pearsonr(T_mean_per_layer, T_var_per_layer) if n > 2 else (0, 1)

    # Kurtosis (deviation from Gaussian)
    T_fluctuations = [all_T[p][l] - mean_T[l] for p in range(len(PROMPTS)) for l in range(n)]
    kurtosis_T = float(stats.kurtosis(T_fluctuations))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'freq_T': freq_T.tolist(), 'psd_T': psd_T.tolist(),
        'freq_P1': freq_P1.tolist(), 'psd_P1': psd_P1.tolist(),
        'freq_U': freq_U.tolist(), 'psd_U': psd_U.tolist(),
        'alpha_T': float(alpha_T), 'r_alpha': float(r_alpha),
        'T_var_per_layer': T_var_per_layer,
        'T_mean_per_layer': T_mean_per_layer,
        'r_fdt': float(r_fdt),
        'kurtosis_T': kurtosis_T,
        'mean_T': mean_T.tolist(),
        'mean_P1': mean_P1.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase 245: Fluctuation Spectrum")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = fluctuation_spectrum(model, tok, device, size)
        results[size] = r
        print(f"  1/f exponent: alpha={r['alpha_T']:.3f} (r={r['r_alpha']:.3f})")
        print(f"  FDT: r(T, var_T) = {r['r_fdt']:.3f}")
        print(f"  Kurtosis(T) = {r['kurtosis_T']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) PSD of T
    for size, r in results.items():
        mask = np.array(r['freq_T']) > 0
        axes[0, 0].loglog(np.array(r['freq_T'])[mask], np.array(r['psd_T'])[mask],
                         '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].set_ylabel('PSD')
    r_last = results[list(results.keys())[-1]]
    axes[0, 0].set_title(f"(a) T Power Spectrum (alpha={r_last['alpha_T']:.2f})")
    axes[0, 0].legend(fontsize=8)

    # (b) PSD of P1
    for size, r in results.items():
        mask = np.array(r['freq_P1']) > 0
        axes[0, 1].loglog(np.array(r['freq_P1'])[mask], np.array(r['psd_P1'])[mask],
                         '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[0, 1].set_xlabel('Frequency'); axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('(b) P1 Power Spectrum')
    axes[0, 1].legend(fontsize=8)

    # (c) Variance vs Mean T per layer (FDT test)
    for size, r in results.items():
        axes[0, 2].scatter(r['T_mean_per_layer'], r['T_var_per_layer'],
                          color=colors[size], s=30, alpha=0.7, label=f"{size} (r={r['r_fdt']:.2f})")
    axes[0, 2].set_xlabel('Mean T'); axes[0, 2].set_ylabel('Var(T)')
    axes[0, 2].set_title('(c) Fluctuation-Dissipation')
    axes[0, 2].legend(fontsize=7)

    # (d) T variance profile
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['T_var_per_layer'])), r['T_var_per_layer'],
                       '-o', color=colors[size], lw=2, markersize=3, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Var(T)')
    axes[1, 0].set_title('(d) T Variance by Layer')
    axes[1, 0].legend(fontsize=8)

    # (e) Fluctuation distribution
    from collections import defaultdict
    for size, r in results.items():
        # Reconstruct fluctuations
        T_fluct = []
        for prompt_T in all_T if False else []:  # placeholder
            pass
        axes[1, 1].text(0.5, 0.5, f"kurtosis(T):\n0.5B={results.get('0.5B',{}).get('kurtosis_T',0):.3f}\n"
                       f"1.5B={results.get('1.5B',{}).get('kurtosis_T',0):.3f}",
                       ha='center', va='center', fontsize=12,
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 1].set_title('(e) Kurtosis (Gaussianity)')
    axes[1, 1].axis('off')

    # (f) Summary
    summary = "FLUCTUATION SPECTRUM\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  1/f exponent: {r['alpha_T']:.3f}\n"
        summary += f"  FDT corr: {r['r_fdt']:.3f}\n"
        summary += f"  Kurtosis: {r['kurtosis_T']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 245: Thermodynamic Fluctuation Spectrum",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase245_fluctuation')
    plt.close()
    save_results('phase245_fluctuation', {'experiment': 'Fluctuation Spectrum', 'results': results})


if __name__ == '__main__':
    main()
