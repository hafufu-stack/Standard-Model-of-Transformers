# -*- coding: utf-8 -*-
"""
Phase 318: Lamb Shift -- Radiative Corrections
=================================================
The Lamb shift is a small energy difference caused by vacuum
fluctuations (quantum electrodynamics correction).
Measure small systematic shifts in hidden state energies that
deviate from the "naive" prediction.
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


def measure_lamb_shift(model, tok, prompt, device):
    """Measure Lamb shift (radiative corrections) in hidden states."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # "Naive" prediction: energy should change smoothly (linear interpolation)
    layer_E = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        E = float((h ** 2).sum().item())
        layer_E.append(E)

    E_arr = np.array(layer_E)

    # Linear interpolation (naive prediction)
    x = np.arange(len(E_arr))
    slope, intercept, _, _, _ = stats.linregress(x, E_arr)
    E_naive = slope * x + intercept

    # Lamb shift = deviation from naive prediction
    lamb_shift = E_arr - E_naive

    # Relative shift
    rel_shift = lamb_shift / (np.abs(E_naive) + 1e-10)

    # Spectral analysis of the shift (is it periodic?)
    fft = np.fft.fft(lamb_shift)
    power = np.abs(fft[1:len(fft)//2]) ** 2
    if len(power) > 0:
        dominant_freq = int(np.argmax(power)) + 1
        dominant_power = float(power.max())
        total_power = float(power.sum())
        freq_concentration = dominant_power / (total_power + 1e-10)
    else:
        dominant_freq = 0
        dominant_power = 0
        freq_concentration = 0

    # RMS shift
    rms_shift = float(np.sqrt(np.mean(lamb_shift ** 2)))
    rms_rel = float(np.sqrt(np.mean(rel_shift ** 2)))

    return {
        'lamb_shift': [round(float(s), 4) for s in lamb_shift],
        'rel_shift': [round(float(s), 6) for s in rel_shift],
        'rms_shift': round(rms_shift, 4),
        'rms_rel_shift': round(rms_rel, 6),
        'dominant_freq': dominant_freq,
        'freq_concentration': round(freq_concentration, 4),
        'naive_slope': round(float(slope), 4),
    }


def main():
    print("=" * 70)
    print("Phase 318: Lamb Shift")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ls_data = []
        for prompt in PROMPTS:
            ls = measure_lamb_shift(model, tok, prompt, device)
            ls_data.append(ls)

        n = len(ls_data[0]['lamb_shift'])
        avg_shift = [float(np.mean([ls['lamb_shift'][i] for ls in ls_data])) for i in range(n)]
        avg_rel = [float(np.mean([ls['rel_shift'][i] for ls in ls_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n - 1,
            'avg_lamb_shift': [round(s, 4) for s in avg_shift],
            'avg_rel_shift': [round(s, 6) for s in avg_rel],
            'rms_shift': round(float(np.mean([ls['rms_shift'] for ls in ls_data])), 4),
            'rms_rel_shift': round(float(np.mean([ls['rms_rel_shift'] for ls in ls_data])), 6),
            'dominant_freq': round(float(np.mean([ls['dominant_freq'] for ls in ls_data])), 1),
            'freq_concentration': round(float(np.mean([ls['freq_concentration'] for ls in ls_data])), 4),
        }
        print(f"  RMS Lamb shift: {all_results[size]['rms_shift']:.4f}")
        print(f"  RMS relative: {all_results[size]['rms_rel_shift']:.6f}")
        print(f"  Dominant frequency: {all_results[size]['dominant_freq']:.1f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_lamb_shift'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Lamb Shift')
    axes[0, 0].set_title('(a) Lamb Shift Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_rel_shift'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Relative Shift')
    axes[0, 1].set_title('(b) Relative Lamb Shift', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['rms_shift'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('RMS Shift'); axes[0, 2].set_title('(c) RMS Lamb Shift', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "LAMB SHIFT\n\n"
    txt += "E_observed - E_naive\n"
    txt += "(radiative correction)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  RMS = {d['rms_shift']:.1f}\n"
        txt += f"  rel = {d['rms_rel_shift']:.5f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 318: Lamb Shift -- Radiative Corrections",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase318_lamb')
    plt.close()
    save_results('phase318_lamb', {'experiment': 'Lamb Shift', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
