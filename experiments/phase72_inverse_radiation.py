# -*- coding: utf-8 -*-
"""
Phase 72: Inverse Radiation Law (Anti-Stefan-Boltzmann)
Phase 69 found L ~ T^(-0.87). This is ANTI-Stefan-Boltzmann!
In normal physics: hotter -> brighter. In LLM: cooler -> more confident.
This is CONSISTENT with negative specific heat (C_v < 0).
Verify this across multiple models and define the Inverse Radiation Constant.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def measure_radiation(model, tok, prompts, device):
    """Measure T and L (luminosity) for all layers and prompts."""
    all_T, all_L, all_layers = [], [], []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(len(out.hidden_states)):
            with torch.no_grad():
                normed = model.model.norm(out.hidden_states[li][:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T) or T < 0.01:
                continue
            L_peak = probs.max().item()
            all_T.append(T)
            all_L.append(L_peak)
            all_layers.append(li)

    return np.array(all_T), np.array(all_L), all_layers


def main():
    print("=" * 70)
    print("Phase 72: Inverse Radiation Law")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration",
        "Quantum mechanics describes the behavior of particles at atomic scale",
        "The human genome contains three billion base pairs encoding all genetic",
        "Neural networks learn representations through layers of interconnected nodes",
        "Black holes form when massive stars exhaust their nuclear fuel",
        "The periodic table organizes elements by atomic number and electron",
        "Evolution by natural selection acts on heritable variation in populations",
        "Climate models simulate atmospheric dynamics using partial differential equations",
        "Photosynthesis converts light energy into chemical energy stored in glucose",
        "Machine learning discovers patterns in data without explicit programming",
        "The cosmic microwave background is remnant radiation from the early universe",
        "General relativity describes gravity as curvature of spacetime caused by mass",
    ]

    model_results = {}

    # Test on all available models
    for model_size, model_name in [('1.5B', 'Qwen2.5-1.5B'), ('0.5B', 'Qwen2.5-0.5B')]:
        print(f"\n--- {model_name} ---")
        try:
            model, tok = load_model(device=device, size=model_size)
            T_arr, L_arr, layers = measure_radiation(model, tok, prompts, device)

            valid = (T_arr > 0.1) & (L_arr > 1e-6)
            log_T = np.log(T_arr[valid])
            log_L = np.log(L_arr[valid])
            slope, intercept, r_val, p_val, _ = stats.linregress(log_T, log_L)

            print(f"  L ~ T^{slope:.2f}, R2={r_val**2:.3f}")
            model_results[model_name] = {
                'slope': float(slope), 'intercept': float(intercept),
                'r_squared': float(r_val**2), 'p_value': float(p_val),
                'sigma': float(np.exp(intercept)),
                'T': T_arr.tolist(), 'L': L_arr.tolist(), 'layers': layers,
            }
            del model
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error: {e}")

    # Also test with TinyLlama
    try:
        import os as _os
        _HF_CACHE = _os.path.expanduser("~/.cache/huggingface/hub")
        _SNAP_TL = _os.path.join(_HF_CACHE, "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0",
                                  "snapshots")
        if _os.path.exists(_SNAP_TL):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            snap_dir = _os.path.join(_SNAP_TL, _os.listdir(_SNAP_TL)[0])
            tok_tl = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
            model_tl = AutoModelForCausalLM.from_pretrained(
                snap_dir, torch_dtype=torch.float16, device_map=device, local_files_only=True)
            model_tl.eval()
            print(f"\n--- TinyLlama-1.1B ---")
            T_arr, L_arr, layers = measure_radiation(model_tl, tok_tl, prompts, device)
            valid = (T_arr > 0.1) & (L_arr > 1e-6)
            log_T = np.log(T_arr[valid])
            log_L = np.log(L_arr[valid])
            slope, intercept, r_val, p_val, _ = stats.linregress(log_T, log_L)
            print(f"  L ~ T^{slope:.2f}, R2={r_val**2:.3f}")
            model_results['TinyLlama-1.1B'] = {
                'slope': float(slope), 'intercept': float(intercept),
                'r_squared': float(r_val**2), 'p_value': float(p_val),
                'sigma': float(np.exp(intercept)),
                'T': T_arr.tolist(), 'L': L_arr.tolist(), 'layers': layers,
            }
            del model_tl
            import gc; gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TinyLlama error: {e}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'Qwen2.5-1.5B': '#e74c3c', 'Qwen2.5-0.5B': '#3498db', 'TinyLlama-1.1B': '#2ecc71'}

    # (a) Log-log all models
    for mname, mr in model_results.items():
        T = np.array(mr['T'])
        L = np.array(mr['L'])
        valid = (T > 0.1) & (L > 1e-6)
        axes[0, 0].scatter(np.log(T[valid])[::3], np.log(L[valid])[::3],
                          s=10, alpha=0.3, color=colors.get(mname, 'gray'), label=mname)
        # Fit line
        t_fit = np.linspace(np.log(T[valid]).min(), np.log(T[valid]).max(), 50)
        axes[0, 0].plot(t_fit, mr['slope'] * t_fit + mr['intercept'],
                       '--', color=colors.get(mname, 'gray'), linewidth=2)
    axes[0, 0].set_xlabel('log(T)')
    axes[0, 0].set_ylabel('log(L)')
    axes[0, 0].set_title('(a) Inverse Radiation Law (all models)')
    axes[0, 0].legend(fontsize=8)

    # (b) Exponents comparison
    mnames = list(model_results.keys())
    exponents = [model_results[m]['slope'] for m in mnames]
    axes[0, 1].bar(mnames, exponents, color=[colors.get(m, 'gray') for m in mnames], alpha=0.8)
    axes[0, 1].axhline(y=np.mean(exponents), color='black', linestyle='--',
                       label=f'Mean n={np.mean(exponents):.2f}')
    axes[0, 1].axhline(y=-1, color='red', linestyle=':', label='n=-1')
    axes[0, 1].set_ylabel('Exponent n')
    axes[0, 1].set_title('(b) Universal Exponent')
    axes[0, 1].legend(fontsize=8)

    # (c) R-squared comparison
    r2s = [model_results[m]['r_squared'] for m in mnames]
    axes[0, 2].bar(mnames, r2s, color=[colors.get(m, 'gray') for m in mnames], alpha=0.8)
    axes[0, 2].set_ylabel('R-squared')
    axes[0, 2].set_title('(c) Fit Quality')

    # (d) Physical interpretation
    interpretation = (
        "ANTI-STEFAN-BOLTZMANN LAW:\n\n"
        "Normal physics: L ~ T^4\n"
        "(hotter = brighter)\n\n"
        f"LLM physics: L ~ T^{np.mean(exponents):.2f}\n"
        "(cooler = more confident)\n\n"
        "This is CONSISTENT with\n"
        "negative specific heat (C_v < 0):\n"
        "Losing energy -> higher T\n"
        "-> lower confidence"
    )
    axes[1, 0].text(0.5, 0.5, interpretation, transform=axes[1, 0].transAxes,
                    fontsize=11, va='center', ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('(d) Physical Interpretation')

    # (e) sigma (radiation constant) per model
    sigmas = [model_results[m]['sigma'] for m in mnames]
    axes[1, 1].bar(mnames, sigmas, color=[colors.get(m, 'gray') for m in mnames], alpha=0.8)
    axes[1, 1].set_ylabel('sigma (radiation constant)')
    axes[1, 1].set_title(f'(e) sigma (mean={np.mean(sigmas):.3f})')

    # (f) Unified radiation constant
    mean_n = np.mean(exponents)
    std_n = np.std(exponents)
    cv_n = std_n / (abs(mean_n) + 1e-10)
    axes[1, 2].bar(['Mean n', 'Std n', 'CV'],
                   [abs(mean_n), std_n, cv_n],
                   color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.8)
    axes[1, 2].set_title(f'(f) Universality (CV={cv_n:.2f})')

    is_universal = cv_n < 0.3 and all(n < 0 for n in exponents)

    fig.suptitle(f'Phase 72: Inverse Radiation Law (n={mean_n:.2f}, '
                 f'{"UNIVERSAL" if is_universal else "model-dependent"})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase72_inverse_radiation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Mean exponent n={mean_n:.2f} +/- {std_n:.2f} (CV={cv_n:.2f}). "
          f"All negative: {all(n < 0 for n in exponents)}. "
          f"Inverse radiation {'UNIVERSAL' if is_universal else 'model-dependent'}.")
    print(f"{'='*70}")

    save_results('phase72_inverse_radiation', {
        'experiment': 'Inverse Radiation Law',
        'per_model': {m: {'slope': model_results[m]['slope'],
                         'r_squared': model_results[m]['r_squared']}
                     for m in model_results},
        'summary': {
            'mean_exponent': float(mean_n),
            'std_exponent': float(std_n),
            'is_universal': bool(is_universal),
        }
    })


if __name__ == '__main__':
    main()
