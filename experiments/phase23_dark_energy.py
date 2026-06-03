# -*- coding: utf-8 -*-
"""
Phase 23: Dark Energy Detection (Opus Original)
=================================================
Track the "Hubble parameter" of the hidden space: how fast
the effective dimensionality (PR) expands per layer.
If Attention is gravity (contracts), what force drives expansion?
Is there a "dark energy" in the FFN layers?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 23: Dark Energy Detection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The fundamental forces of nature include gravity electromagnetism",
        "Machine learning algorithms process training data to learn patterns",
        "The periodic table organizes chemical elements by atomic number",
        "Stars generate energy through hydrogen fusion in their cores",
        "Neural network architectures transform input representations layer by layer",
        "The ocean currents distribute heat energy across the planet surface",
    ]

    # Decompose each layer into Attention contribution and FFN contribution
    all_attn_effect = []
    all_ffn_effect = []
    all_pr_profile = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Capture attention output and FFN output separately
        attn_outputs = []
        ffn_outputs = []
        pre_attn_states = []

        def make_attn_hook(storage):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    storage.append(output[0][0, -1, :].float().detach().cpu())
                return output
            return hook

        def make_ffn_hook(storage):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    storage.append(output[0][0, -1, :].float().detach().cpu())
                else:
                    storage.append(output[0, -1, :].float().detach().cpu())
                return output
            return hook

        handles = []
        for li in range(n_layers):
            a_store = []
            f_store = []
            attn_outputs.append(a_store)
            ffn_outputs.append(f_store)
            h1 = model.model.layers[li].self_attn.register_forward_hook(make_attn_hook(a_store))
            h2 = model.model.layers[li].mlp.register_forward_hook(make_ffn_hook(f_store))
            handles.extend([h1, h2])

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in handles:
            h.remove()

        # Hidden state PR at each layer
        pr_profile = []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float().cpu()
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()
            pr_profile.append(PR)

        # Attn effect on PR (does it increase or decrease PR?)
        attn_pr_change = []
        ffn_pr_change = []
        for li in range(n_layers):
            if attn_outputs[li] and ffn_outputs[li]:
                attn_vec = attn_outputs[li][0]
                ffn_vec = ffn_outputs[li][0]
                # Norm as "force magnitude"
                attn_force = attn_vec.norm().item()
                ffn_force = ffn_vec.norm().item()
                # PR change across this layer
                if li + 1 < len(pr_profile):
                    dPR = pr_profile[li + 1] - pr_profile[li]
                else:
                    dPR = 0
                attn_pr_change.append(attn_force)
                ffn_pr_change.append(ffn_force)
            else:
                attn_pr_change.append(0)
                ffn_pr_change.append(0)

        all_attn_effect.append(attn_pr_change)
        all_ffn_effect.append(ffn_pr_change)
        all_pr_profile.append(pr_profile)

    # Average
    min_len_eff = min(len(a) for a in all_attn_effect)
    avg_attn = np.mean([a[:min_len_eff] for a in all_attn_effect], axis=0)
    avg_ffn = np.mean([f[:min_len_eff] for f in all_ffn_effect], axis=0)
    min_len_pr = min(len(p) for p in all_pr_profile)
    avg_pr = np.mean([p[:min_len_pr] for p in all_pr_profile], axis=0)

    # "Dark energy fraction" = FFN_force / (Attn_force + FFN_force)
    dark_frac = avg_ffn / (avg_attn + avg_ffn + 1e-10)

    print("\n--- Force Decomposition ---")
    for li in range(0, min_len_eff, 4):
        print(f"  L{li}: Attn force={avg_attn[li]:.2f}, FFN force={avg_ffn[li]:.2f}, "
              f"Dark energy frac={dark_frac[li]:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    layers = np.arange(min_len_eff)
    ax.plot(layers, avg_attn, 'o-', color='#3498db', ms=4, label='Attention (gravity)')
    ax.plot(layers, avg_ffn, 's-', color='#e74c3c', ms=4, label='FFN (dark energy)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Force Magnitude (L2 norm)')
    ax.set_title('(a) Gravity vs Dark Energy')
    ax.legend()

    ax = axes[1]
    ax.fill_between(layers, 0, 1-dark_frac, alpha=0.3, color='#3498db', label='Gravity')
    ax.fill_between(layers, 1-dark_frac, 1, alpha=0.3, color='#e74c3c', label='Dark Energy')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction')
    ax.set_title('(b) Energy Budget')
    ax.legend()

    ax = axes[2]
    # PR expansion rate (Hubble)
    dPR = np.diff(avg_pr[:min_len_eff+1])
    hubble = dPR / (avg_pr[:min_len_eff] + 1e-10)
    ax.plot(layers, hubble, 'o-', color='#9b59b6', ms=4)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('dPR/PR (Hubble parameter)')
    ax.set_title('(c) Expansion Rate')

    mean_dark = np.mean(dark_frac)
    fig.suptitle(
        f"Phase 23: Dark Energy Detection\n"
        f"Mean dark energy fraction = {mean_dark:.3f} "
        f"({mean_dark*100:.1f}% of total force is FFN)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase23_dark_energy")
    plt.close()

    if mean_dark > 0.5:
        verdict = (f"FFN DOMINATES: Dark energy={mean_dark:.3f}. "
                   f"FFN contributes {mean_dark*100:.0f}% of the total force. "
                   f"The 'expansion' of representation is driven by FFN!")
    else:
        verdict = (f"GRAVITY DOMINATES: Dark energy={mean_dark:.3f}. "
                   f"Attention contributes {(1-mean_dark)*100:.0f}% of force. "
                   f"But FFN provides crucial {mean_dark*100:.0f}% expansion.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 23: Dark Energy Detection',
        'summary': {'verdict': verdict, 'dark_energy_fraction': float(mean_dark)},
    }
    save_results("phase23_dark_energy", result)
    return result


if __name__ == '__main__':
    main()
