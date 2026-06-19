# -*- coding: utf-8 -*-
"""
Phase 311: Goldstone Modes -- Massless Excitations from SSB
============================================================
Goldstone's theorem: spontaneous symmetry breaking produces massless
excitations (Nambu-Goldstone bosons). In the transformer context:
- Look for zero-energy modes in the SVD spectrum
- These are directions in hidden space with no "cost" to excite
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
]


def measure_goldstone_modes(model, tok, prompt, device):
    """Detect Goldstone (massless) modes in hidden state spectrum."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    layer_n_goldstone = []
    layer_gap = []
    layer_mass_spectrum = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0].float()
        _, s, _ = torch.linalg.svd(h, full_matrices=False)
        s = s.cpu().numpy()

        # Normalize
        s_norm = s / (s[0] + 1e-10)

        # "Mass" = singular value (energy scale)
        # Goldstone modes: s ~ 0 (near-zero singular values)
        threshold = 0.01  # relative to largest
        n_goldstone = int(np.sum(s_norm < threshold))
        layer_n_goldstone.append(n_goldstone)

        # Mass gap: difference between first and second singular values
        if len(s_norm) > 1:
            gap = float(s_norm[0] - s_norm[1])
        else:
            gap = 0
        layer_gap.append(gap)

        # First 10 masses
        layer_mass_spectrum.append([round(float(m), 4) for m in s_norm[:10]])

    # Goldstone fraction
    D = model.config.hidden_size
    goldstone_frac = [n / min(D, out.hidden_states[0].shape[1]) for n in layer_n_goldstone]

    return {
        'n_goldstone': layer_n_goldstone,
        'goldstone_frac': [round(f, 4) for f in goldstone_frac],
        'mass_gap': [round(g, 4) for g in layer_gap],
        'mass_spectrum_L0': layer_mass_spectrum[0] if layer_mass_spectrum else [],
        'mass_spectrum_Lmid': layer_mass_spectrum[n_layers // 2] if layer_mass_spectrum else [],
        'mass_spectrum_Lfinal': layer_mass_spectrum[-1] if layer_mass_spectrum else [],
    }


def main():
    print("=" * 70)
    print("Phase 311: Goldstone Modes")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        gm_data = []
        for prompt in PROMPTS:
            gm = measure_goldstone_modes(model, tok, prompt, device)
            gm_data.append(gm)

        n = len(gm_data[0]['n_goldstone'])
        avg_ng = [float(np.mean([g['n_goldstone'][i] for g in gm_data])) for i in range(n)]
        avg_gf = [float(np.mean([g['goldstone_frac'][i] for g in gm_data])) for i in range(n)]
        avg_gap = [float(np.mean([g['mass_gap'][i] for g in gm_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n - 1,
            'avg_n_goldstone': [round(g, 1) for g in avg_ng],
            'avg_goldstone_frac': [round(f, 4) for f in avg_gf],
            'avg_mass_gap': [round(g, 4) for g in avg_gap],
            'mean_n_goldstone': round(float(np.mean(avg_ng)), 1),
            'mean_goldstone_frac': round(float(np.mean(avg_gf)), 4),
        }
        print(f"  Mean Goldstone modes: {all_results[size]['mean_n_goldstone']:.0f}")
        print(f"  Mean Goldstone fraction: {all_results[size]['mean_goldstone_frac']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_n_goldstone'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('N Goldstone Modes')
    axes[0, 0].set_title('(a) Goldstone Mode Count', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_goldstone_frac'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Goldstone Fraction')
    axes[0, 1].set_title('(b) Goldstone Fraction', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_mass_gap'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Mass Gap')
    axes[0, 2].set_title('(c) Mass Gap Profile', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['mean_n_goldstone'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Mean N Goldstone'); axes[1, 0].set_title('(d) Mean Goldstone Count', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')

    txt = "GOLDSTONE MODES\n\n"
    txt += "SSB -> massless excitations\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}: {d['mean_n_goldstone']:.0f} modes\n"
        txt += f"  frac: {d['mean_goldstone_frac']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 311: Goldstone Modes -- Massless Excitations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase311_goldstone')
    plt.close()
    save_results('phase311_goldstone', {'experiment': 'Goldstone Modes', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
