# -*- coding: utf-8 -*-
"""
Phase 19: Thermodynamic Phase Forcing
=======================================
Force a poetry prompt into the "arithmetic phase" (PR*T~1176)
by directly manipulating hidden state thermodynamic quantities.
Can we switch the model's "cognitive mode" through physics alone?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 19: Thermodynamic Phase Forcing")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    # Targets from Phase 15
    TARGET_PRT = 1200.0  # "Arithmetic phase"
    poetry_prompts = [
        "Roses are red violets are blue the sun shines",
        "The moonlight dances across the silver lake gently",
        "In the garden of dreams flowers bloom eternally",
        "Whispers of wind carry songs through the forest",
    ]
    arith_prompts = [
        "Two plus three equals five and five plus two equals",
        "Calculate the sum of twelve and fifteen which is",
        "The product of seven and eight is fifty six and",
        "If x equals three then two x plus one equals",
    ]

    # Measure baseline
    print("\n--- Baseline Measurements ---")
    poetry_base = []
    for p in poetry_prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, p, device)
        prt_final = thermo[-1]['PRT']
        poetry_base.append({'thermo': thermo, 'prt': prt_final})
        print(f"  Poetry: PR*T = {prt_final:.1f}")

    arith_base = []
    for p in arith_prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, p, device)
        prt_final = thermo[-1]['PRT']
        arith_base.append({'thermo': thermo, 'prt': prt_final})
        print(f"  Arithmetic: PR*T = {prt_final:.1f}")

    avg_poetry_prt = np.mean([d['prt'] for d in poetry_base])
    avg_arith_prt = np.mean([d['prt'] for d in arith_base])
    print(f"\n  Avg Poetry PR*T = {avg_poetry_prt:.1f}")
    print(f"  Avg Arith PR*T = {avg_arith_prt:.1f}")

    # Phase Forcing: scale hidden states to achieve target PR*T
    print(f"\n--- Phase Forcing: targeting PR*T={TARGET_PRT:.0f} ---")

    def make_phase_force_hook(target_prt, lm_head):
        """Scale hidden state to force PR*T toward target."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].float()
                # Measure current PR*T
                last = h[:, -1:, :]
                with torch.no_grad():
                    logits = lm_head(last.to(next(lm_head.parameters()).dtype)).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                h_vec = last.squeeze().float()
                h_sq = h_vec ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                PR = 1.0 / (h_prob ** 2).sum().item()
                current_prt = PR * T

                if current_prt > 0:
                    scale = (target_prt / current_prt) ** 0.25  # gentle correction
                    scale = max(0.5, min(2.0, scale))  # clamp
                    h_scaled = h * scale
                    h_scaled = torch.nan_to_num(h_scaled, nan=0.0, posinf=65000.0, neginf=-65000.0)
                    return (h_scaled.to(output[0].dtype),) + output[1:]
            return output
        return hook

    # Apply forcing hooks to layers 5-25
    handles = []
    force_layers = list(range(5, min(25, n_layers)))
    for li in force_layers:
        h = model.model.layers[li].register_forward_hook(
            make_phase_force_hook(TARGET_PRT, model.lm_head))
        handles.append(h)

    forced_data = []
    forced_tokens = []
    for p in poetry_prompts:
        thermo, out = measure_full_thermodynamics(model, tok, p, device)
        prt_final = thermo[-1]['PRT']
        forced_data.append({'thermo': thermo, 'prt': prt_final})
        # Get top tokens
        logits = out.logits[0, -1, :].float()
        top5 = torch.topk(logits, 5)
        top_words = tok.decode(top5.indices.tolist())
        forced_tokens.append(top_words)
        print(f"  Forced Poetry: PR*T = {prt_final:.1f}, top tokens: {top_words}")

    for h in handles:
        h.remove()

    avg_forced_prt = np.mean([d['prt'] for d in forced_data])

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    x = ['Poetry\n(baseline)', 'Arithmetic\n(baseline)', f'Poetry\n(forced)']
    vals = [avg_poetry_prt, avg_arith_prt, avg_forced_prt]
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(x, vals, color=colors, alpha=0.8)
    ax.axhline(y=TARGET_PRT, color='red', ls='--', label=f'Target ({TARGET_PRT})')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{v:.0f}', ha='center', fontsize=10)
    ax.set_ylabel('PR x T')
    ax.set_title('(a) Phase Comparison')
    ax.legend()

    # Layer-by-layer comparison
    ax = axes[1]
    n_vis = len(poetry_base[0]['thermo'])
    base_prt_layers = np.mean([[d['PRT'] for d in t['thermo']] for t in poetry_base], axis=0)
    forced_prt_layers = np.mean([[d['PRT'] for d in t['thermo']] for t in forced_data], axis=0)
    arith_prt_layers = np.mean([[d['PRT'] for d in t['thermo']] for t in arith_base], axis=0)
    ax.plot(range(n_vis), base_prt_layers, 'o-', color='#3498db', ms=3, label='Poetry')
    ax.plot(range(n_vis), arith_prt_layers, 's-', color='#e74c3c', ms=3, label='Arithmetic')
    ax.plot(range(n_vis), forced_prt_layers, '^-', color='#9b59b6', ms=3, label='Forced')
    ax.set_xlabel('Layer'); ax.set_ylabel('PR x T')
    ax.set_title('(b) PR*T Profile per Layer')
    ax.legend()

    # Temperature comparison
    ax = axes[2]
    base_T = np.mean([[d['T'] for d in t['thermo']] for t in poetry_base], axis=0)
    forced_T = np.mean([[d['T'] for d in t['thermo']] for t in forced_data], axis=0)
    arith_T = np.mean([[d['T'] for d in t['thermo']] for t in arith_base], axis=0)
    ax.plot(range(n_vis), base_T, 'o-', color='#3498db', ms=3, label='Poetry')
    ax.plot(range(n_vis), arith_T, 's-', color='#e74c3c', ms=3, label='Arithmetic')
    ax.plot(range(n_vis), forced_T, '^-', color='#9b59b6', ms=3, label='Forced')
    ax.set_xlabel('Layer'); ax.set_ylabel('T (logit entropy)')
    ax.set_title('(c) Temperature Profile')
    ax.legend()

    shift = avg_forced_prt - avg_poetry_prt
    target_shift = TARGET_PRT - avg_poetry_prt
    efficiency = shift / (target_shift + 1e-10) * 100

    fig.suptitle(
        f"Phase 19: Thermodynamic Phase Forcing\n"
        f"Poetry {avg_poetry_prt:.0f} -> Forced {avg_forced_prt:.0f} "
        f"(target {TARGET_PRT:.0f}, efficiency {efficiency:.0f}%)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase19_phase_forcing")
    plt.close()

    if abs(avg_forced_prt - TARGET_PRT) < abs(avg_poetry_prt - TARGET_PRT) * 0.5:
        verdict = (f"PHASE FORCING SUCCESS: Poetry PR*T {avg_poetry_prt:.0f} -> {avg_forced_prt:.0f} "
                   f"(target {TARGET_PRT:.0f}, {efficiency:.0f}% efficient). "
                   f"Cognitive mode switched via physics!")
    else:
        verdict = (f"PARTIAL FORCING: Poetry PR*T {avg_poetry_prt:.0f} -> {avg_forced_prt:.0f} "
                   f"(target {TARGET_PRT:.0f}). System resists phase change.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 19: Thermodynamic Phase Forcing',
        'summary': {'verdict': verdict, 'efficiency': efficiency,
                    'forced_tokens': forced_tokens},
    }
    save_results("phase19_phase_forcing", result)
    return result


if __name__ == '__main__':
    main()
