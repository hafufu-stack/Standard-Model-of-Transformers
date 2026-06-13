# -*- coding: utf-8 -*-
"""
Phase 101: kT Reheating Universality
Phase 100 found kT minimum at L11-17 and 2.73x reheating surge.
Test if this "cooling valley + reheating" pattern is universal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
]

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
]


def boltzmann(E, A, kT):
    return A * np.exp(-E / (kT + 1e-10))


def measure_kT_profile(model, tok, device, n_layers):
    """Measure kT at each layer via Boltzmann fit."""
    kT_profile = []
    r2_profile = []

    for li in range(n_layers):
        all_norms = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            if li < len(out.hidden_states):
                h = out.hidden_states[li][0, -1, :].float()
                norms = h.abs().cpu().numpy()
                all_norms.extend(norms[norms > 0].tolist())

        if len(all_norms) < 50:
            kT_profile.append(0.0)
            r2_profile.append(0.0)
            continue

        all_norms = np.array(all_norms)
        hist, edges = np.histogram(all_norms, bins=30, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = hist > 0
        hv, bc = hist[mask], centers[mask]

        try:
            popt, _ = curve_fit(boltzmann, bc, hv, p0=[hv[0], np.mean(all_norms)],
                                maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
            pred = boltzmann(bc, *popt)
            ss_res = np.sum((hv - pred)**2)
            ss_tot = np.sum((hv - np.mean(hv))**2)
            r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))
            kT_profile.append(float(popt[1]))
            r2_profile.append(float(r2))
        except Exception:
            kT_profile.append(0.0)
            r2_profile.append(0.0)

    return kT_profile, r2_profile


def main():
    print("=" * 70)
    print("Phase 101: kT Reheating Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_data = {}

    for model_id, model_name in MODELS:
        print(f"\n--- {model_name} ---")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device,
                trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"  Failed: {str(e)[:80]}")
            continue

        n_layers = len(model.model.layers) + 1
        kT_profile, r2_profile = measure_kT_profile(model, tok, device, n_layers)

        # Find cooling valley (minimum kT in L2..L-3)
        inner_kTs = kT_profile[2:-2] if len(kT_profile) > 4 else kT_profile
        if inner_kTs:
            min_idx = np.argmin(inner_kTs) + 2
            min_kT = kT_profile[min_idx]
        else:
            min_idx = 0
            min_kT = 0

        # Find peak (max kT)
        peak_idx = np.argmax(kT_profile[1:]) + 1  # skip embedding
        peak_kT = kT_profile[peak_idx]

        # Reheating ratio
        reheat_ratio = peak_kT / (min_kT + 1e-10)

        # Normalized positions
        min_frac = min_idx / (n_layers - 1)
        peak_frac = peak_idx / (n_layers - 1)

        print(f"  Layers: {n_layers}")
        print(f"  Cooling valley: L{min_idx} (kT={min_kT:.3f}, frac={min_frac:.3f})")
        print(f"  Reheating peak: L{peak_idx} (kT={peak_kT:.3f}, frac={peak_frac:.3f})")
        print(f"  Reheating ratio: {reheat_ratio:.2f}x")

        all_data[model_name] = {
            'n_layers': n_layers,
            'kT_profile': kT_profile,
            'r2_profile': r2_profile,
            'min_layer': int(min_idx),
            'min_kT': float(min_kT),
            'min_frac': float(min_frac),
            'peak_layer': int(peak_idx),
            'peak_kT': float(peak_kT),
            'peak_frac': float(peak_frac),
            'reheat_ratio': float(reheat_ratio),
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Qwen2.5-1.5B': '#c0392b', 'Qwen2.5-0.5B': '#2980b9', 'TinyLlama-1.1B': '#27ae60'}
    model_names = list(all_data.keys())

    # (a) kT profiles (normalized x-axis)
    for mname, data in all_data.items():
        n = data['n_layers']
        x_norm = np.linspace(0, 1, n)
        axes[0].plot(x_norm, data['kT_profile'], 'o-', color=colors.get(mname, 'gray'),
                    markersize=3, linewidth=1.5, label=mname)
        # Mark valley and peak
        axes[0].scatter(data['min_frac'], data['min_kT'], marker='v', s=100,
                       color=colors.get(mname, 'gray'), edgecolors='black', zorder=10)
        axes[0].scatter(data['peak_frac'], data['peak_kT'], marker='^', s=100,
                       color=colors.get(mname, 'gray'), edgecolors='black', zorder=10)
    axes[0].set_xlabel('Normalized Layer ($l/L$)')
    axes[0].set_ylabel('$kT$')
    axes[0].set_title('(a) kT Profiles')
    axes[0].legend(fontsize=7)

    # (b) Valley and peak positions
    min_fracs = [all_data[m]['min_frac'] for m in model_names]
    peak_fracs = [all_data[m]['peak_frac'] for m in model_names]
    x = np.arange(len(model_names))
    axes[1].bar(x - 0.2, min_fracs, 0.35, color='#3498db', alpha=0.8, label='Valley $l/L$')
    axes[1].bar(x + 0.2, peak_fracs, 0.35, color='#c0392b', alpha=0.8, label='Peak $l/L$')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=8)
    axes[1].set_ylabel('Normalized Position')
    mean_min = np.mean(min_fracs)
    mean_peak = np.mean(peak_fracs)
    axes[1].axhline(y=mean_min, color='#3498db', linestyle='--', alpha=0.5)
    axes[1].axhline(y=mean_peak, color='#c0392b', linestyle='--', alpha=0.5)
    axes[1].set_title(f'(b) Valley={mean_min:.2f}, Peak={mean_peak:.2f}')
    axes[1].legend(fontsize=8)

    # (c) Reheating ratio
    ratios = [all_data[m]['reheat_ratio'] for m in model_names]
    bar_colors = [colors.get(m, 'gray') for m in model_names]
    axes[2].bar(range(len(model_names)), ratios, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, fontsize=8)
    axes[2].set_ylabel('Reheating Ratio $kT_{peak}/kT_{valley}$')
    mean_ratio = np.mean(ratios)
    cv_ratio = np.std(ratios) / (mean_ratio + 1e-10)
    axes[2].axhline(y=mean_ratio, color='black', linestyle='--',
                    label=f'Mean = {mean_ratio:.1f}x')
    axes[2].set_title(f'(c) Reheating (CV={cv_ratio:.2f})')
    axes[2].legend()

    is_universal = cv_ratio < 0.5 and all(r > 1.5 for r in ratios)
    fig.suptitle(f'Phase 101: kT Reheating '
                 f'(mean {mean_ratio:.1f}x, {"UNIVERSAL" if is_universal else "VARIES"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase101_reheating')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Valley positions: {[f'{f:.2f}' for f in min_fracs]}")
    print(f"Peak positions: {[f'{f:.2f}' for f in peak_fracs]}")
    print(f"Reheating ratios: {[f'{r:.1f}x' for r in ratios]}")
    print(f"Mean reheating: {mean_ratio:.1f}x, CV={cv_ratio:.2f}")
    print(f"{'='*70}")

    save_results('phase101_reheating', {
        'experiment': 'kT Reheating Universality',
        'models': {m: {k: v for k, v in d.items() if k not in ['kT_profile', 'r2_profile']}
                   for m, d in all_data.items()},
        'profiles': {m: {'kT': d['kT_profile'], 'r2': d['r2_profile']}
                    for m, d in all_data.items()},
        'summary': {
            'mean_valley_frac': float(mean_min),
            'mean_peak_frac': float(mean_peak),
            'mean_reheat_ratio': float(mean_ratio),
            'cv': float(cv_ratio),
            'is_universal': is_universal,
        }
    })


if __name__ == '__main__':
    main()
