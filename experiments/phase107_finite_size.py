# -*- coding: utf-8 -*-
"""
Phase 107: Finite-Size Scaling
In physics, phase transitions are verified by checking how critical properties
scale with system size. Test variance peak height vs model size (n_layers).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats
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
    "The human genome encodes three billion base pairs",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Climate change affects global ecosystems",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
]


def measure_variance_profile(model, tok, device, n_layers, norm_layer, lm_head):
    """Measure eta variance at each effective depth."""
    results = []
    for L in range(4, n_layers):
        etas = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            T_vals = []
            for li in range(min(L + 1, len(out.hidden_states))):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                if not np.isnan(T):
                    T_vals.append(T)
            if len(T_vals) >= 4:
                T_hot = max(T_vals)
                T_cold = min(T_vals[len(T_vals)//2:])
                if T_hot > 0.01:
                    etas.append(1.0 - T_cold / T_hot)
        if etas:
            results.append({
                'L': L,
                'eta_mean': float(np.mean(etas)),
                'eta_var': float(np.var(etas)),
                'eta_std': float(np.std(etas)),
            })
    return results


def main():
    print("=" * 70)
    print("Phase 107: Finite-Size Scaling")
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
        norm_layer = model.model.norm
        lm_head = model.lm_head

        results = measure_variance_profile(model, tok, device, n_layers, norm_layer, lm_head)

        # Find variance peak
        vars_ = [r['eta_var'] for r in results]
        peak_idx = np.argmax(vars_)
        peak_L = results[peak_idx]['L']
        peak_var = vars_[peak_idx]
        peak_frac = peak_L / (n_layers - 1)

        print(f"  n_layers={n_layers-1}, var peak at L={peak_L} (frac={peak_frac:.3f}), "
              f"peak_var={peak_var:.6f}")

        all_data[model_name] = {
            'n_layers': n_layers - 1,
            'results': results,
            'peak_L': int(peak_L),
            'peak_var': float(peak_var),
            'peak_frac': float(peak_frac),
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # === Finite-size scaling analysis ===
    model_names = list(all_data.keys())
    n_layers_arr = np.array([all_data[m]['n_layers'] for m in model_names])
    peak_vars = np.array([all_data[m]['peak_var'] for m in model_names])
    peak_fracs = np.array([all_data[m]['peak_frac'] for m in model_names])

    # Fit: peak_var ~ L^(gamma/nu)
    scaling_exp = 0
    scaling_r2 = 0
    if len(n_layers_arr) >= 3:
        try:
            def power(L, A, exp):
                return A * L**exp
            popt, _ = curve_fit(power, n_layers_arr, peak_vars, p0=[0.001, 1.0],
                                maxfev=5000)
            scaling_exp = popt[1]
            pred = power(n_layers_arr, *popt)
            ss_res = np.sum((peak_vars - pred)**2)
            ss_tot = np.sum((peak_vars - np.mean(peak_vars))**2)
            scaling_r2 = 1 - ss_res / (ss_tot + 1e-10)
        except Exception:
            pass

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Qwen2.5-1.5B': '#c0392b', 'Qwen2.5-0.5B': '#2980b9', 'TinyLlama-1.1B': '#27ae60'}

    # (a) Variance profiles (normalized x-axis)
    for mname, data in all_data.items():
        Ls = [r['L'] for r in data['results']]
        x_norm = [l / data['n_layers'] for l in Ls]
        vs = [r['eta_var'] for r in data['results']]
        axes[0].plot(x_norm, vs, 'o-', color=colors.get(mname, 'gray'),
                    markersize=3, linewidth=1.5, label=f'{mname} (L={data["n_layers"]})')
    axes[0].set_xlabel('Normalized Layer $l/L$')
    axes[0].set_ylabel('$\\mathrm{Var}(\\eta)$')
    axes[0].set_title('(a) Variance Profiles')
    axes[0].legend(fontsize=7)

    # (b) Peak variance vs system size
    axes[1].scatter(n_layers_arr, peak_vars, s=120, c=[colors.get(m, 'gray') for m in model_names],
                   edgecolors='black', zorder=5)
    for i, m in enumerate(model_names):
        axes[1].annotate(m, (n_layers_arr[i], peak_vars[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    if scaling_exp != 0:
        L_fit = np.linspace(min(n_layers_arr)-2, max(n_layers_arr)+2, 50)
        axes[1].plot(L_fit, power(L_fit, *popt), '--', color='gray',
                    label=f'$L^{{{scaling_exp:.2f}}}$ ($R^2={scaling_r2:.3f}$)')
    axes[1].set_xlabel('System Size $L$')
    axes[1].set_ylabel('Peak $\\mathrm{Var}(\\eta)$')
    axes[1].set_title(f'(b) Finite-Size Scaling ($\\gamma/\\nu \\approx {scaling_exp:.2f}$)')
    axes[1].legend(fontsize=8)

    # (c) Peak position universality
    axes[2].bar(range(len(model_names)), peak_fracs,
               color=[colors.get(m, 'gray') for m in model_names],
               alpha=0.8, edgecolor='black')
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, fontsize=8)
    axes[2].set_ylabel('Peak Position $l_{peak}/L$')
    mean_frac = np.mean(peak_fracs)
    cv_frac = np.std(peak_fracs) / (mean_frac + 1e-10)
    axes[2].axhline(y=mean_frac, color='black', linestyle='--',
                    label=f'Mean = {mean_frac:.3f}')
    axes[2].set_title(f'(c) Peak Position (CV={cv_frac:.3f})')
    axes[2].legend()

    fig.suptitle(f'Phase 107: Finite-Size Scaling '
                 f'(peak at {mean_frac:.2f}L, $\\gamma/\\nu \\approx {scaling_exp:.2f}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase107_finite_size')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Peak positions: {[f'{f:.3f}' for f in peak_fracs]}")
    print(f"Scaling exponent gamma/nu: {scaling_exp:.3f}")
    print(f"Mean peak fraction: {mean_frac:.3f}")
    print(f"{'='*70}")

    save_results('phase107_finite_size', {
        'experiment': 'Finite-Size Scaling',
        'models': {m: {k: v for k, v in d.items() if k != 'results'}
                   for m, d in all_data.items()},
        'summary': {
            'scaling_exp': float(scaling_exp),
            'scaling_r2': float(scaling_r2),
            'mean_peak_frac': float(mean_frac),
            'cv_peak_frac': float(cv_frac),
        }
    })


if __name__ == '__main__':
    main()
