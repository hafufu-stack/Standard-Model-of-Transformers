# -*- coding: utf-8 -*-
"""
Phase 6b: Architecture Comparison (FIXED)
==========================================
Season 1 bug: self_attn hook didn't block attention because of
residual connection (x_out = x_in + Attn(x_in)).
Fix: Zero out attention OUTPUT explicitly before residual addition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def power_law(x, a, b):
    return a * np.power(x, b)


def main():
    print("=" * 70)
    print("Phase 6b: Architecture Comparison (FIXED)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The meaning of life is",
        "In quantum mechanics,",
        "Machine learning algorithms",
        "The temperature of the sun is",
        "Water is composed of",
    ]

    def measure_thermodynamics(model, tok, prompts, device):
        all_temps = []
        all_prs = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            temps = []
            prs = []
            for hs in out.hidden_states:
                h = hs[0, -1, :].float()
                temps.append(h.norm().item())
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                prs.append(1.0 / (h_prob ** 2).sum().item())
            all_temps.append(temps)
            all_prs.append(prs)
        return np.mean(all_temps, axis=0), np.mean(all_prs, axis=0)

    # Full Attention
    print("\n--- Full Attention Mode ---")
    full_temps, full_prs = measure_thermodynamics(model, tok, prompts, device)

    # FIX: Zero out attention output BEFORE residual addition
    print("\n--- Attention Zeroed Mode (FIXED) ---")
    handles = []
    for li in range(n_layers):
        def make_zero_attn_hook():
            def hook(module, args, output):
                # output is (attn_output, attn_weights, past_kv)
                if isinstance(output, tuple):
                    zero = torch.zeros_like(output[0])
                    return (zero,) + output[1:]
                return torch.zeros_like(output)
            return hook
        h = model.model.layers[li].self_attn.register_forward_hook(make_zero_attn_hook())
        handles.append(h)

    zero_temps, zero_prs = measure_thermodynamics(model, tok, prompts, device)
    for h in handles:
        h.remove()

    # FFN zeroed (keep attention, remove FFN)
    print("\n--- FFN Zeroed Mode ---")
    handles = []
    for li in range(n_layers):
        def make_zero_ffn_hook():
            def hook(module, args, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return hook
        h = model.model.layers[li].mlp.register_forward_hook(make_zero_ffn_hook())
        handles.append(h)

    ffn_zero_temps, ffn_zero_prs = measure_thermodynamics(model, tok, prompts, device)
    for h in handles:
        h.remove()

    # Fit cooling laws
    layers = np.arange(1, len(full_temps) + 1).astype(float)
    fits = {}
    for name, temps in [('Full', full_temps), ('No-Attn', zero_temps), ('No-FFN', ffn_zero_temps)]:
        try:
            popt, _ = curve_fit(power_law, layers, temps, p0=[temps[0], -0.5], maxfev=5000)
            fits[name] = popt[1]
        except Exception:
            fits[name] = 0.0
        print(f"  {name}: alpha = {fits[name]:.4f}")

    # Noether conservation
    full_prt = full_prs[1:] * full_temps[1:]
    zero_prt = zero_prs[1:] * zero_temps[1:]
    ffn_prt = ffn_zero_prs[1:] * ffn_zero_temps[1:]

    full_cv = np.std(full_prt) / (np.mean(full_prt) + 1e-10) * 100
    zero_cv = np.std(zero_prt) / (np.mean(zero_prt) + 1e-10) * 100
    ffn_cv = np.std(ffn_prt) / (np.mean(ffn_prt) + 1e-10) * 100

    print(f"\n  PR*T Conservation:")
    print(f"    Full:     mean={np.mean(full_prt):.1f}, CV={full_cv:.1f}%")
    print(f"    No-Attn:  mean={np.mean(zero_prt):.1f}, CV={zero_cv:.1f}%")
    print(f"    No-FFN:   mean={np.mean(ffn_prt):.1f}, CV={ffn_cv:.1f}%")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(layers, full_temps, 'o-', color='#e74c3c', label=f'Full (a={fits["Full"]:.3f})', ms=4)
    ax.plot(layers, zero_temps, 's-', color='#3498db', label=f'No-Attn (a={fits["No-Attn"]:.3f})', ms=4)
    ax.plot(layers, ffn_zero_temps, '^-', color='#2ecc71', label=f'No-FFN (a={fits["No-FFN"]:.3f})', ms=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Temperature (L2 norm)')
    ax.set_title('(a) Cooling Law: T ~ l^alpha')
    ax.legend()

    ax = axes[1]
    layer_range = np.arange(1, len(full_prt) + 1)
    ax.plot(layer_range, full_prt, 'o-', color='#e74c3c', label=f'Full (CV={full_cv:.1f}%)', ms=4)
    ax.plot(layer_range, zero_prt, 's-', color='#3498db', label=f'No-Attn (CV={zero_cv:.1f}%)', ms=4)
    ax.plot(layer_range, ffn_prt, '^-', color='#2ecc71', label=f'No-FFN (CV={ffn_cv:.1f}%)', ms=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('PR x T')
    ax.set_title('(b) Noether Conservation')
    ax.legend()

    ax = axes[2]
    ax.plot(range(len(full_prs)), full_prs, 'o-', color='#e74c3c', label='Full', ms=4)
    ax.plot(range(len(zero_prs)), zero_prs, 's-', color='#3498db', label='No-Attn', ms=4)
    ax.plot(range(len(ffn_zero_prs)), ffn_zero_prs, '^-', color='#2ecc71', label='No-FFN', ms=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('(c) Participation Ratio')
    ax.legend()

    fig.suptitle(
        f"Phase 6b: Architecture Comparison (FIXED)\n"
        f"Full: a={fits['Full']:.3f} | No-Attn: a={fits['No-Attn']:.3f} | No-FFN: a={fits['No-FFN']:.3f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase6b_architecture_fixed")
    plt.close()

    alpha_diff = abs(fits['Full'] - fits['No-Attn'])
    if alpha_diff > 0.1:
        verdict = (f"ATTENTION IS ESSENTIAL: alpha changes {fits['Full']:.3f} -> {fits['No-Attn']:.3f} "
                   f"(diff={alpha_diff:.3f}). Attention drives the cooling law.")
    else:
        verdict = (f"ARCHITECTURE-INDEPENDENT: alpha similar ({alpha_diff:.3f}). "
                   f"The cooling law is geometric.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 6b: Architecture Comparison (FIXED)',
        'summary': {'verdict': verdict,
                    'alpha_full': fits['Full'], 'alpha_no_attn': fits['No-Attn'],
                    'alpha_no_ffn': fits['No-FFN'], 'alpha_diff': alpha_diff,
                    'prt_full_cv': full_cv, 'prt_no_attn_cv': zero_cv},
    }
    save_results("phase6b_architecture_fixed", result)
    return result


if __name__ == '__main__':
    main()
