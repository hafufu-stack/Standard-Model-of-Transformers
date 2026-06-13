# -*- coding: utf-8 -*-
"""
Phase 96: Ergodicity Phase Transition Critical Exponent
Phase 85 found T-ergodicity breaks at L3. Measure the critical exponent
of this phase transition: how does the KS p-value scale near L_c?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

ENSEMBLE_PROMPTS = [
    "The capital of France is", "Water boils at one hundred",
    "The speed of light equals", "Photosynthesis converts sunlight into",
    "DNA stands for deoxyribonucleic", "The largest planet is Jupiter",
    "Gravity pulls objects toward the", "The periodic table organizes",
    "Machine learning uses data to", "Black holes form when massive",
    "The human genome contains three", "Evolution works through natural",
    "Quantum mechanics describes particles at", "Neural networks learn through",
    "The ocean covers seventy percent", "Electricity flows through conductors",
    "The mitochondria is the powerhouse", "Chemical reactions involve breaking",
    "Plate tectonics drives continental", "The speed of sound in air",
    "Semiconductors enable modern computing", "Protein folding determines function",
    "The Turing test measures intelligence", "Cryptography relies on prime numbers",
    "Thermodynamics governs energy transfer", "General relativity warps spacetime",
    "The standard model classifies particles", "Entropy always increases in closed",
    "Superconductors have zero electrical", "Photons are massless particles of",
]

TIME_PROMPTS = [
    "The history of science is a tale of discovery and innovation that spans millennia from the ancient philosophers who first pondered the nature of matter and the cosmos to modern researchers who probe the fundamental constituents of reality with particle accelerators and space telescopes revealing ever deeper layers of structure and complexity in the natural world",
    "Artificial intelligence represents one of humanitys most ambitious intellectual projects aiming to create machines that can think reason and learn with the sophistication and flexibility of biological minds drawing on advances in neuroscience mathematics and computer engineering to build systems that could one day rival or surpass human cognitive capabilities",
    "The ocean covers more than seventy percent of the earths surface and contains an extraordinary diversity of life from microscopic phytoplankton that produce half the planets oxygen to the great whales that migrate thousands of miles across entire ocean basins connecting distant ecosystems through their movements and feeding behaviors",
]


def main():
    print("=" * 70)
    print("Phase 96: Ergodicity Phase Transition Critical Exponent")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # === Collect ensemble T at every layer ===
    print("  Collecting ensemble...")
    ens_T = {l: [] for l in range(n_layers)}

    for prompt in ENSEMBLE_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li, hs in enumerate(out.hidden_states):
            if li >= n_layers:
                break
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T):
                ens_T[li].append(T)

    # === Collect time series T ===
    print("  Collecting time series...")
    tim_T = {l: [] for l in range(n_layers)}

    for prompt in TIME_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li, hs in enumerate(out.hidden_states):
            if li >= n_layers:
                break
            for pos in range(seq_len):
                with torch.no_grad():
                    normed = model.model.norm(hs[:, pos:pos+1, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                if not np.isnan(T):
                    tim_T[li].append(T)

    # === KS tests ===
    print("  Running KS tests...")
    ks_results = []
    for li in range(n_layers):
        e = np.array(ens_T[li])
        t = np.array(tim_T[li])
        if len(e) > 2 and len(t) > 2:
            ks = sp_stats.ks_2samp(e, t)
            ks_results.append({
                'layer': li,
                'ks_stat': float(ks.statistic),
                'ks_p': float(ks.pvalue),
                'ergodic': bool(ks.pvalue > 0.05),
            })

    # === Find critical layer and exponent ===
    layers = np.array([r['layer'] for r in ks_results])
    ks_stats = np.array([r['ks_stat'] for r in ks_results])
    ks_ps = np.array([r['ks_p'] for r in ks_results])

    # Find L_c: first layer where p < 0.05
    L_c = None
    for r in ks_results:
        if r['ks_p'] < 0.05:
            L_c = r['layer']
            break

    # Fit critical exponent: KS_stat ~ |L - L_c|^beta near L_c
    fit_b = None
    fit_r2 = 0
    if L_c is not None and L_c > 0:
        try:
            def critical_law(L, A, beta):
                return A * np.abs(L - L_c + 0.1) ** beta

            # Use layers around L_c
            mask = (layers >= max(0, L_c - 3)) & (layers <= L_c + 10)
            if mask.sum() >= 4:
                popt, _ = curve_fit(critical_law, layers[mask], ks_stats[mask],
                                    p0=[0.5, 0.5], maxfev=5000,
                                    bounds=([0, 0], [10, 3]))
                pred = critical_law(layers[mask], *popt)
                ss_res = np.sum((ks_stats[mask] - pred)**2)
                ss_tot = np.sum((ks_stats[mask] - np.mean(ks_stats[mask]))**2)
                fit_r2 = 1 - ss_res / (ss_tot + 1e-10)
                fit_b = popt[1]
                print(f"  Critical exponent beta = {fit_b:.3f} (R2={fit_r2:.3f})")
        except Exception as e:
            print(f"  Fit failed: {e}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) KS statistic profile
    colors = ['#27ae60' if r['ergodic'] else '#c0392b' for r in ks_results]
    axes[0].bar(layers, ks_stats, color=colors, alpha=0.7, edgecolor='black')
    if L_c is not None:
        axes[0].axvline(x=L_c, color='#f39c12', linewidth=2, linestyle='--',
                        label=f'$L_c = {L_c}$')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('KS Statistic')
    axes[0].set_title('(a) Ergodicity Breaking Profile')
    axes[0].legend()

    # (b) p-value profile (log scale)
    axes[1].semilogy(layers, ks_ps, 'o-', color='#2980b9', markersize=4)
    axes[1].axhline(y=0.05, color='#c0392b', linestyle='--', label='$p = 0.05$')
    if L_c is not None:
        axes[1].axvline(x=L_c, color='#f39c12', linewidth=2, linestyle='--',
                        label=f'$L_c = {L_c}$')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('KS p-value')
    axes[1].set_title('(b) P-value Profile')
    axes[1].legend(fontsize=8)

    # (c) Critical exponent fit
    if fit_b is not None and L_c is not None:
        mask = (layers >= max(0, L_c - 3)) & (layers <= L_c + 10)
        axes[2].scatter(layers[mask] - L_c, ks_stats[mask], s=80, c='#c0392b',
                       edgecolors='black', zorder=5, label='Data')
        x_fit = np.linspace(-3, 10, 100)
        axes[2].plot(x_fit, popt[0] * np.abs(x_fit + 0.1)**popt[1], '--',
                     color='#2980b9', linewidth=2,
                     label=f'$\\propto |L-L_c|^{{{fit_b:.2f}}}$')
        axes[2].set_xlabel('$L - L_c$')
        axes[2].set_ylabel('KS Statistic')
        axes[2].set_title(f'(c) Critical Exponent $\\beta = {fit_b:.3f}$ ($R^2={fit_r2:.3f}$)')
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, 'No critical point\ndetected',
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('(c) Critical Exponent')

    ergodic_count = sum(1 for r in ks_results if r['ergodic'])
    fig.suptitle(f'Phase 96: Ergodicity Phase Transition ($L_c={L_c}$, '
                 f'$\\beta={fit_b:.3f}$)' if fit_b else
                 f'Phase 96: Ergodicity Phase Transition ($L_c={L_c}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase96_critical_exponent')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Ergodic layers: {ergodic_count}/{len(ks_results)}")
    print(f"Critical layer L_c: {L_c}")
    if fit_b:
        print(f"Critical exponent beta: {fit_b:.3f}")
    print(f"{'='*70}")

    save_results('phase96_critical_exponent', {
        'experiment': 'Ergodicity Critical Exponent',
        'ks_results': ks_results,
        'summary': {
            'n_ergodic': ergodic_count,
            'L_c': L_c,
            'beta': float(fit_b) if fit_b else None,
            'fit_r2': float(fit_r2),
        }
    })


if __name__ == '__main__':
    main()
