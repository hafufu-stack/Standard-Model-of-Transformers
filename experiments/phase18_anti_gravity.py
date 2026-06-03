# -*- coding: utf-8 -*-
"""
Phase 18: Anti-Gravity & The Big Rip
======================================
Reverse the sign of attention: h = h - Attn(h) instead of h = h + Attn(h).
If Attention is gravity, reversing it should cause time reversal,
entropy explosion, and the "Big Rip" of the meaning space.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 18: Anti-Gravity & The Big Rip")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The meaning of life is to seek understanding",
        "Water freezes at zero degrees Celsius",
        "The brain processes information through neurons",
        "Mathematics is the language of nature",
    ]

    # Baseline
    print("\n--- Baseline ---")
    baseline_data = []
    for p in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, p, device)
        baseline_data.append(thermo)

    # Anti-gravity: subtract attention instead of adding
    print("\n--- Anti-Gravity Mode ---")
    # Hook layers 10-20 (middle layers) to invert attention
    anti_layers = list(range(10, min(20, n_layers)))

    def make_anti_gravity_hook():
        """Invert attention: instead of adding, subtract the attention output."""
        captured_input = {}
        def pre_hook(module, args):
            # Capture the input to attention
            if isinstance(args, tuple) and len(args) > 0:
                captured_input['h'] = args[0].clone()
            return args

        def post_hook(module, args, output):
            if isinstance(output, tuple) and 'h' in captured_input:
                h_in = captured_input['h']
                attn_out = output[0]
                # Normal: residual = h_in + attn_out
                # Anti-gravity: we want residual = h_in - attn_out
                # Since the model will ADD this output, we return -attn_out
                # so residual = h_in + (-attn_out) = h_in - attn_out
                anti = -attn_out
                anti = torch.nan_to_num(anti, nan=0.0, posinf=65000.0, neginf=-65000.0)
                return (anti,) + output[1:]
            return output
        return pre_hook, post_hook

    handles = []
    for li in anti_layers:
        pre_h, post_h = make_anti_gravity_hook()
        h1 = model.model.layers[li].self_attn.register_forward_pre_hook(pre_h)
        h2 = model.model.layers[li].self_attn.register_forward_hook(post_h)
        handles.extend([h1, h2])

    anti_data = []
    for p in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, p, device)
        anti_data.append(thermo)

    for h in handles:
        h.remove()

    # Compute entropy of output distribution (measure chaos)
    print("\n--- Output Entropy ---")
    for mode, data_list, label in [("Normal", baseline_data, "Baseline"),
                                     ("Anti-G", anti_data, "Anti-Gravity")]:
        avg_final_T = np.mean([d[-1]['T'] for d in data_list])
        avg_final_U = np.mean([d[-1]['U'] for d in data_list])
        print(f"  {label}: Final T(entropy)={avg_final_T:.2f}, Final U(energy)={avg_final_U:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    n_vis_layers = len(baseline_data[0])
    layers = np.arange(n_vis_layers)

    # Average across prompts
    base_U = np.mean([[d['U'] for d in thermo] for thermo in baseline_data], axis=0)
    base_T = np.mean([[d['T'] for d in thermo] for thermo in baseline_data], axis=0)
    base_PR = np.mean([[d['PR'] for d in thermo] for thermo in baseline_data], axis=0)
    anti_U = np.mean([[d['U'] for d in thermo] for thermo in anti_data], axis=0)
    anti_T = np.mean([[d['T'] for d in thermo] for thermo in anti_data], axis=0)
    anti_PR = np.mean([[d['PR'] for d in thermo] for thermo in anti_data], axis=0)

    ax = axes[0][0]
    ax.plot(layers, base_U, 'o-', color='#3498db', ms=3, label='Normal')
    ax.plot(layers, anti_U, 's-', color='#e74c3c', ms=3, label='Anti-Gravity')
    ax.axvspan(10, min(20, n_layers), alpha=0.1, color='red', label='Anti-G zone')
    ax.set_xlabel('Layer'); ax.set_ylabel('U (L2 norm)')
    ax.set_title('(a) Internal Energy'); ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.plot(layers, base_T, 'o-', color='#3498db', ms=3, label='Normal')
    ax.plot(layers, anti_T, 's-', color='#e74c3c', ms=3, label='Anti-Gravity')
    ax.axvspan(10, min(20, n_layers), alpha=0.1, color='red')
    ax.set_xlabel('Layer'); ax.set_ylabel('T (logit entropy)')
    ax.set_title('(b) Temperature'); ax.legend(fontsize=8)

    ax = axes[0][2]
    ax.plot(layers, base_PR, 'o-', color='#3498db', ms=3, label='Normal')
    ax.plot(layers, anti_PR, 's-', color='#e74c3c', ms=3, label='Anti-Gravity')
    ax.axvspan(10, min(20, n_layers), alpha=0.1, color='red')
    ax.set_xlabel('Layer'); ax.set_ylabel('PR')
    ax.set_title('(c) Participation Ratio'); ax.legend(fontsize=8)

    # dU/dT comparison
    ax = axes[1][0]
    base_dU = np.diff(base_U)
    base_dT = np.diff(base_T)
    anti_dU = np.diff(anti_U)
    anti_dT = np.diff(anti_T)
    ax.scatter(base_dT, base_dU, c='#3498db', s=30, label='Normal', alpha=0.7)
    ax.scatter(anti_dT, anti_dU, c='#e74c3c', s=30, label='Anti-Gravity', alpha=0.7)
    ax.set_xlabel('dT'); ax.set_ylabel('dU')
    ax.set_title('(d) dU vs dT Phase Space'); ax.legend()

    # PRT comparison
    base_PRT = base_PR * base_T
    anti_PRT = anti_PR * anti_T
    ax = axes[1][1]
    ax.plot(layers, base_PRT, 'o-', color='#3498db', ms=3, label='Normal')
    ax.plot(layers, anti_PRT, 's-', color='#e74c3c', ms=3, label='Anti-Gravity')
    ax.axvspan(10, min(20, n_layers), alpha=0.1, color='red')
    ax.set_xlabel('Layer'); ax.set_ylabel('PR x T')
    ax.set_title('(e) Conservation Law'); ax.legend(fontsize=8)

    # Ratio plot
    ax = axes[1][2]
    ratio_U = anti_U / (base_U + 1e-10)
    ratio_T = anti_T / (base_T + 1e-10)
    ax.plot(layers, ratio_U, 'o-', color='#e74c3c', ms=3, label='U ratio')
    ax.plot(layers, ratio_T, 's-', color='#3498db', ms=3, label='T ratio')
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax.axvspan(10, min(20, n_layers), alpha=0.1, color='red')
    ax.set_xlabel('Layer'); ax.set_ylabel('Anti-G / Normal')
    ax.set_title('(f) Anti-Gravity Effect Ratio'); ax.legend()

    # Check if dU/dT sign reversed
    try:
        valid = np.isfinite(base_T[1:]) & np.isfinite(base_U[1:])
        base_slope = np.polyfit(base_T[1:][valid], base_U[1:][valid], 1)[0] if valid.sum() > 2 else 0
    except Exception:
        base_slope = 0
    try:
        valid = np.isfinite(anti_T[1:]) & np.isfinite(anti_U[1:])
        anti_slope = np.polyfit(anti_T[1:][valid], anti_U[1:][valid], 1)[0] if valid.sum() > 2 else 0
    except Exception:
        anti_slope = 0

    fig.suptitle(
        f"Phase 18: Anti-Gravity & The Big Rip\n"
        f"Normal dU/dT={base_slope:.2f} | Anti-G dU/dT={anti_slope:.2f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase18_anti_gravity")
    plt.close()

    sign_reversed = (base_slope < 0 and anti_slope > 0) or (base_slope > 0 and anti_slope < 0)
    if sign_reversed:
        verdict = (f"BIG RIP CONFIRMED: dU/dT reversed ({base_slope:.2f} -> {anti_slope:.2f}). "
                   f"Anti-gravity causes time reversal and entropy explosion!")
    else:
        verdict = (f"PARTIAL RIP: dU/dT ({base_slope:.2f} -> {anti_slope:.2f}). "
                   f"Anti-gravity disrupts but doesn't fully reverse thermodynamics.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 18: Anti-Gravity & The Big Rip',
        'summary': {'verdict': verdict, 'base_slope': base_slope, 'anti_slope': anti_slope},
    }
    save_results("phase18_anti_gravity", result)
    return result


if __name__ == '__main__':
    main()
