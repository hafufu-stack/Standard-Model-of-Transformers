# -*- coding: utf-8 -*-
"""
Phase 290: Mach Scaling Law -- Does Mach -> 1.0 at Large Scale?
================================================================
Phase 281 found Mach=0.78 (0.5B) and Mach=0.99 (1.5B).
Test whether Mach number converges to 1.0 as model size grows.
If true: transformers operate at the sonic barrier.
Include 7B model for the third data point to test scaling.
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
    "The fundamental theorem of calculus connects",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
]


def measure_speed_of_sound(model, tok, prompt, device, sigma=0.1):
    """Measure perturbation propagation speed through layers."""
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    clean_states = [h[0].float().cpu() for h in out_clean.hidden_states]

    # Embed-level noise injection
    def make_embed_hook(sigma_val):
        def hook_fn(module, input, output):
            h = output
            if h.dim() == 3:
                noise = torch.randn(1, 1, h.shape[-1], device=h.device, dtype=h.dtype) * sigma_val
                h_new = h.clone()
                h_new[:, 0:1, :] = h[:, 0:1, :] + noise
            elif h.dim() == 2:
                noise = torch.randn(1, h.shape[-1], device=h.device, dtype=h.dtype) * sigma_val
                h_new = h.clone()
                h_new[0:1, :] = h[0:1, :] + noise
            else:
                return output
            return h_new
        return hook_fn

    handle = model.model.embed_tokens.register_forward_hook(make_embed_hook(sigma))
    with torch.no_grad():
        out_noisy = model(**inp, output_hidden_states=True)
    handle.remove()

    noisy_states = [h[0].float().cpu() for h in out_noisy.hidden_states]

    speeds = []
    for li in range(1, len(clean_states)):
        diff = (noisy_states[li] - clean_states[li]).norm(dim=-1)
        if diff.sum() < 1e-8:
            continue
        threshold = diff.max() * 0.1
        affected = (diff > threshold).nonzero(as_tuple=True)[0]
        if len(affected) > 0:
            max_reach = affected.max().item()
            speeds.append(max_reach / li)

    return float(np.mean(speeds)) if speeds else 0.0


def measure_attention_light_cone(model, tok, prompt, device):
    """Measure effective attention reach per layer."""
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    reaches = []
    for li, hs in enumerate(out.hidden_states[1:], 1):
        h = hs[0].float()
        if h.shape[0] < 3:
            continue
        last = h[-1]
        sims = torch.nn.functional.cosine_similarity(h[:-1], last.unsqueeze(0), dim=-1)
        effective_reach = (sims > 0.5).sum().item()
        reaches.append(effective_reach)

    if len(reaches) >= 2:
        x = np.arange(len(reaches))
        slope, _, _, _, _ = stats.linregress(x, reaches)
        return float(slope), reaches
    return 0.0, reaches


def main():
    print("=" * 70)
    print("Phase 290: Mach Scaling Law -- Does Mach -> 1.0?")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    param_counts = {'0.5B': 5e8, '1.5B': 1.5e9, '7B': 7e9}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        n_layers = len(model.model.layers)

        c_s_values = []
        c_l_values = []
        for prompt in PROMPTS:
            cs = measure_speed_of_sound(model, tok, prompt, device)
            cl, _ = measure_attention_light_cone(model, tok, prompt, device)
            c_s_values.append(cs)
            c_l_values.append(cl)

        c_s = float(np.mean(c_s_values))
        c_l = float(np.mean(c_l_values))
        mach = c_s / max(c_l, 1e-10)

        all_results[size] = {
            'n_params': param_counts[size],
            'n_layers': n_layers,
            'c_sound': round(c_s, 4),
            'c_light': round(c_l, 4),
            'mach_number': round(mach, 4),
            'subsonic': mach < 1.0,
            '1_minus_mach': round(1.0 - mach, 4),
        }
        print(f"  c_s = {c_s:.3f}, c_light = {c_l:.3f}")
        print(f"  Mach = {mach:.3f} (1-M = {1-mach:.4f})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Scaling Analysis ===
    sizes_list = list(all_results.keys())
    log_params = [np.log(all_results[s]['n_params']) for s in sizes_list]
    machs = [all_results[s]['mach_number'] for s in sizes_list]
    one_minus_m = [all_results[s]['1_minus_mach'] for s in sizes_list]

    # Fit: 1-M ~ N^(-alpha)
    log_1mm = [np.log(max(x, 1e-10)) for x in one_minus_m]
    if len(log_params) >= 2:
        slope, intercept, r, p, se = stats.linregress(log_params, log_1mm)
        scaling_alpha = -slope
    else:
        scaling_alpha = 0
        r = 0
        p = 1

    # Predicted Mach for 70B (extrapolation)
    log_70b = np.log(7e10)
    predicted_1mm_70b = np.exp(intercept + slope * log_70b)
    predicted_mach_70b = 1.0 - predicted_1mm_70b

    scaling = {
        'alpha': round(float(scaling_alpha), 4),
        'R2': round(float(r**2), 4),
        'p_value': round(float(p), 6),
        'predicted_mach_70B': round(float(predicted_mach_70b), 6),
        'converges_to_1': scaling_alpha > 0,
    }
    print(f"\n--- Scaling Law ---")
    print(f"  1-M ~ N^(-{scaling_alpha:.3f})")
    print(f"  R2 = {r**2:.4f}, p = {p:.4f}")
    print(f"  Predicted Mach(70B) = {predicted_mach_70b:.6f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_list = ['#3498db', '#e74c3c', '#2ecc71']

    # (a) Mach number vs model size
    params = [all_results[s]['n_params'] for s in sizes_list]
    axes[0, 0].semilogx(params, machs, 'o-', color='#e74c3c', lw=2, markersize=10)
    axes[0, 0].axhline(1.0, color='gold', ls='--', lw=2, label='Mach = 1.0 (sonic barrier)')
    axes[0, 0].set_xlabel('Model Parameters')
    axes[0, 0].set_ylabel('Mach Number')
    axes[0, 0].set_title('(a) Mach Number vs Scale', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) 1-M scaling (log-log)
    axes[0, 1].loglog(params, one_minus_m, 'o-', color='#3498db', lw=2, markersize=10)
    # Fit line
    if scaling_alpha > 0:
        fit_params = np.logspace(np.log10(min(params)*0.5), np.log10(7e10), 50)
        fit_1mm = np.exp(intercept + slope * np.log(fit_params))
        axes[0, 1].loglog(fit_params, fit_1mm, '--', color='grey', alpha=0.5,
                         label=f'1-M ~ N^(-{scaling_alpha:.2f})')
    axes[0, 1].set_xlabel('Model Parameters')
    axes[0, 1].set_ylabel('1 - Mach')
    axes[0, 1].set_title('(b) Sonic Barrier Approach', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) c_s and c_light per model
    x = np.arange(len(sizes_list))
    w = 0.35
    axes[0, 2].bar(x - w/2, [all_results[s]['c_sound'] for s in sizes_list], w,
                  label='c_sound', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['c_light'] for s in sizes_list], w,
                  label='c_light', color='#e74c3c')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(sizes_list)
    axes[0, 2].set_ylabel('Speed (pos/layer)')
    axes[0, 2].set_title('(c) Speed Comparison', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) N_layers vs Mach
    n_layers_list = [all_results[s]['n_layers'] for s in sizes_list]
    axes[1, 0].plot(n_layers_list, machs, 'o-', color='#9b59b6', lw=2, markersize=10)
    axes[1, 0].axhline(1.0, color='gold', ls='--', lw=1)
    axes[1, 0].set_xlabel('Number of Layers')
    axes[1, 0].set_ylabel('Mach Number')
    axes[1, 0].set_title('(d) Mach vs Depth', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Extrapolation
    extrap_sizes = [5e8, 1.5e9, 7e9, 1.3e10, 7e10]
    extrap_labels = ['0.5B', '1.5B', '7B', '13B', '70B']
    if scaling_alpha > 0:
        extrap_mach = [1.0 - np.exp(intercept + slope * np.log(n)) for n in extrap_sizes]
    else:
        extrap_mach = machs + [0.99, 0.999]
    axes[1, 1].semilogx(extrap_sizes[:len(machs)], machs, 'o', color='#e74c3c',
                        markersize=10, label='Measured', zorder=5)
    axes[1, 1].semilogx(extrap_sizes, extrap_mach[:len(extrap_sizes)], '--', color='grey',
                        alpha=0.5, label='Extrapolation')
    axes[1, 1].axhline(1.0, color='gold', ls='--', lw=2)
    for i, label in enumerate(extrap_labels[:len(extrap_mach)]):
        axes[1, 1].annotate(label, (extrap_sizes[i], extrap_mach[i]),
                           textcoords="offset points", xytext=(5, 10), fontsize=8)
    axes[1, 1].set_xlabel('Parameters')
    axes[1, 1].set_ylabel('Mach Number')
    axes[1, 1].set_title('(e) Extrapolation to 70B', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "MACH SCALING LAW\n\n"
    for s in sizes_list:
        r = all_results[s]
        txt += f"{s}: Mach = {r['mach_number']:.3f}\n"
    txt += f"\n1-M ~ N^(-{scaling_alpha:.3f})\n"
    txt += f"R2 = {scaling['R2']:.4f}\n"
    txt += f"\nPrediction:\n"
    txt += f"  Mach(70B) = {predicted_mach_70b:.4f}\n"
    txt += f"\nVerdict: {'Converges to Mach=1' if scaling_alpha > 0 else 'No convergence'}"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 290: Mach Scaling Law -- Does Mach -> 1.0?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase290_mach_scaling')
    plt.close()

    save_results('phase290_mach_scaling', {
        'experiment': 'Mach Scaling Law',
        'results': all_results,
        'scaling': scaling,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
