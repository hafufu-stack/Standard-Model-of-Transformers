# -*- coding: utf-8 -*-
"""
Phase 20: Autoregressive Cosmological Expansion
=================================================
Track PR*T (energy density) during 200-token autoregressive generation.
As KV cache grows (space expands), does attention gravity dilute?
When does the "heat death" (hallucination) begin?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 20: Autoregressive Cosmological Expansion")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    max_new_tokens = 200

    seed_prompts = [
        "The fundamental theorem of calculus states that",
        "In the beginning the universe was extremely hot and dense",
        "The human immune system consists of",
    ]

    all_results = []

    for seed in seed_prompts:
        print(f"\n--- Generating from: '{seed[:50]}...' ---")
        inp = tok(seed, return_tensors='pt').to(device)
        input_ids = inp['input_ids']

        step_data = []
        past_kv = None

        for t in range(max_new_tokens):
            if past_kv is None:
                curr_input = input_ids
            else:
                curr_input = next_token_id

            with torch.no_grad():
                out = model(
                    input_ids=curr_input,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )

            past_kv = out.past_key_values
            logits = out.logits[0, -1, :].float()

            # Thermodynamic measurements
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            PR = 1.0 / (probs ** 2).sum().item()
            PRT = PR * T

            # Hidden state energy
            h_last = out.hidden_states[-1][0, -1, :].float()
            U = h_last.norm().item()

            # Top-1 probability (confidence)
            top_prob = probs.max().item()

            # KV cache size
            try:
                kv_size = past_kv.get_seq_length()
            except AttributeError:
                try:
                    kv_size = past_kv[0][0].shape[2]
                except Exception:
                    kv_size = t + len(input_ids[0])

            step_data.append({
                'step': t, 'T': T, 'PR': PR, 'PRT': PRT,
                'U': U, 'top_prob': top_prob, 'kv_size': kv_size,
            })

            # Sample next token (safe: handle zero probabilities)
            probs_safe = probs.clamp(min=1e-10)
            probs_safe = probs_safe / probs_safe.sum()
            next_token_id = torch.multinomial(probs_safe, 1).unsqueeze(0)

            if t % 50 == 0:
                print(f"  t={t}: PR*T={PRT:.1f}, T={T:.2f}, U={U:.1f}, "
                      f"kv={kv_size}, top_p={top_prob:.3f}")

        all_results.append({'seed': seed, 'steps': step_data})

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    cmap = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    for idx, res in enumerate(all_results):
        steps = [d['step'] for d in res['steps']]
        label = res['seed'][:30] + '...'

        ax = axes[0][0]
        ax.plot(steps, [d['PRT'] for d in res['steps']], '-', color=cmap[idx],
                lw=1, label=label, alpha=0.8)
        ax.set_ylabel('PR x T'); ax.set_title('(a) Energy Density over Time')

        ax = axes[0][1]
        ax.plot(steps, [d['T'] for d in res['steps']], '-', color=cmap[idx], lw=1, alpha=0.8)
        ax.set_ylabel('T (entropy)'); ax.set_title('(b) Temperature Evolution')

        ax = axes[0][2]
        ax.plot(steps, [d['U'] for d in res['steps']], '-', color=cmap[idx], lw=1, alpha=0.8)
        ax.set_ylabel('U (L2 norm)'); ax.set_title('(c) Internal Energy')

        ax = axes[1][0]
        ax.plot(steps, [d['top_prob'] for d in res['steps']], '-', color=cmap[idx],
                lw=1, alpha=0.8)
        ax.set_ylabel('Top-1 Probability'); ax.set_title('(d) Confidence (gravity strength)')

        ax = axes[1][1]
        ax.plot(steps, [d['kv_size'] for d in res['steps']], '-', color=cmap[idx],
                lw=1, alpha=0.8)
        ax.set_ylabel('KV Cache Size'); ax.set_title('(e) Space Expansion')

    axes[0][0].legend(fontsize=6)

    # Correlation: PRT vs KV size (all prompts)
    ax = axes[1][2]
    all_prt = []
    all_kv = []
    for res in all_results:
        for d in res['steps']:
            all_prt.append(d['PRT'])
            all_kv.append(d['kv_size'])
    ax.scatter(all_kv, all_prt, s=5, alpha=0.3, c='#e74c3c')
    try:
        z = np.polyfit(all_kv, all_prt, 1)
        p = np.poly1d(z)
        kv_range = np.linspace(min(all_kv), max(all_kv), 100)
        ax.plot(kv_range, p(kv_range), '--', color='black', label=f'slope={z[0]:.2f}')
    except Exception:
        pass
    ax.set_xlabel('KV Cache Size')
    ax.set_ylabel('PR x T')
    ax.set_title('(f) Energy Density vs Space Size')
    ax.legend()

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Generation Step')

    # Compute Hubble-like constant: d(PRT)/dt / PRT
    hubble_values = []
    for res in all_results:
        prt_series = [d['PRT'] for d in res['steps']]
        for i in range(1, len(prt_series)):
            if prt_series[i-1] > 0:
                h = (prt_series[i] - prt_series[i-1]) / prt_series[i-1]
                hubble_values.append(h)
    mean_hubble = np.mean(hubble_values) if hubble_values else 0

    fig.suptitle(
        f"Phase 20: Cosmological Expansion\n"
        f"Hubble constant (dPRT/PRT per step) = {mean_hubble:.4f} | "
        f"{'EXPANDING' if mean_hubble > 0 else 'CONTRACTING'}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase20_cosmological_expansion")
    plt.close()

    if mean_hubble < -0.01:
        verdict = (f"HEAT DEATH: Hubble={mean_hubble:.4f}. "
                   f"Energy density declines as space expands. "
                   f"Hallucination = cosmological heat death.")
    elif mean_hubble > 0.01:
        verdict = (f"INFLATION: Hubble={mean_hubble:.4f}. "
                   f"Energy density INCREASES. "
                   f"Dark energy dominates attention gravity!")
    else:
        verdict = (f"STEADY STATE: Hubble={mean_hubble:.4f}. "
                   f"Energy density is roughly constant despite expansion.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 20: Cosmological Expansion',
        'summary': {'verdict': verdict, 'hubble': mean_hubble},
    }
    save_results("phase20_cosmological_expansion", result)
    return result


if __name__ == '__main__':
    main()
