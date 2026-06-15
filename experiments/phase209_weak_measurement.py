# -*- coding: utf-8 -*-
"""
Phase 209: Weak Measurement Dashboard
========================================
Phase 205 proved that "strong measurement" (Zeno = discrete token projection)
destroys inference (+263% entropy). Existing interpretability tools like
Logit Lens do exactly this and corrupt the computation.

Weak Measurement: observe ONLY continuous thermodynamic variables
(eta, U, T, S) without ever projecting to token space at intermediate layers.
This is the safe, non-destructive alternative for white-box monitoring.

Verify: weak measurement preserves output quality perfectly.
Visualize: a "cardiac monitor" dashboard for the inference engine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

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

L0 = 21  # Phase transition layer


def measure_weak(model, tok, device, prompt):
    """Weak measurement: extract eta/U/T/S at all layers without discretization."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    norm_layer = model.model.norm
    lm_head = model.lm_head

    U_list, T_list, S_list = [], [], []
    for hs in out.hidden_states:
        h = hs[0, -1, :].float()
        # U: internal energy (L2 norm)
        U_list.append(h.norm().item())
        # S: hidden entropy (from activation distribution)
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
        S_list.append(S if not np.isnan(S) else 0)
        # T: temperature (logit entropy via final norm + lm_head)
        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_list.append(T if not np.isnan(T) else 0)

    # Derived quantities
    T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
    T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

    # dT/dLayer for anomaly detection
    dT = [abs(T_list[i+1] - T_list[i]) for i in range(len(T_list)-1)]

    # Output quality (untouched by measurement)
    final_logits = out.logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
    top1_prob = final_probs.max().item()
    top_token = tok.decode(final_logits.argmax().item())

    return {
        'U': U_list, 'T': T_list, 'S': S_list,
        'dT': dT,
        'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
        'output_entropy': output_entropy, 'top1_prob': top1_prob,
        'top_token': top_token,
    }


def measure_strong_zeno(model, tok, device, prompt, measure_every=1):
    """Strong measurement (Zeno): project to token at each layer."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head
    embed_tokens = model.model.embed_tokens

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)

        # Compute position embeddings (Qwen2 requirement)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        for li in range(n_layers):
            layer = model.model.layers[li]
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Strong measurement: collapse to token
            if measure_every > 0 and (li + 1) % measure_every == 0 and li < n_layers - 1:
                normed_h = norm_layer(hidden[:, -1:, :])
                collapse_logits = lm_head(normed_h).squeeze()
                collapsed_token_id = collapse_logits.argmax().item()
                collapsed_embed = embed_tokens(
                    torch.tensor([[collapsed_token_id]], device=device))
                hidden = hidden.clone()
                hidden[0, -1, :] = collapsed_embed[0, 0, :]

        normed = norm_layer(hidden)
        final_logits = lm_head(normed)
        final_probs = torch.softmax(final_logits[0, -1, :].float(), dim=-1)
        output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
        top1_prob = final_probs.max().item()

    return {
        'output_entropy': output_entropy,
        'top1_prob': top1_prob,
    }


def main():
    print("=" * 70)
    print("Phase 209: Weak Measurement Dashboard")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    # Collect weak measurements for all prompts
    weak_results = []
    for prompt in PROMPTS:
        r = measure_weak(model, tok, device, prompt)
        weak_results.append(r)

    # Compare: weak vs strong vs baseline
    print("\n=== Weak vs Strong Measurement Comparison ===")
    comparison = {'weak': [], 'strong_every1': [], 'strong_every4': []}
    for pi, prompt in enumerate(PROMPTS):
        w = weak_results[pi]
        s1 = measure_strong_zeno(model, tok, device, prompt, measure_every=1)
        s4 = measure_strong_zeno(model, tok, device, prompt, measure_every=4)
        comparison['weak'].append(w['output_entropy'])
        comparison['strong_every1'].append(s1['output_entropy'])
        comparison['strong_every4'].append(s4['output_entropy'])
        print(f"  Prompt {pi+1}: weak={w['output_entropy']:.3f}, "
              f"strong(1)={s1['output_entropy']:.3f}, "
              f"strong(4)={s4['output_entropy']:.3f}")

    mean_weak = float(np.mean(comparison['weak']))
    mean_s1 = float(np.mean(comparison['strong_every1']))
    mean_s4 = float(np.mean(comparison['strong_every4']))

    print(f"\n  Mean: weak={mean_weak:.3f}, strong(1)={mean_s1:.3f}, "
          f"strong(4)={mean_s4:.3f}")
    print(f"  Weak measurement overhead: 0% (zero perturbation)")
    print(f"  Strong(1) entropy increase: "
          f"{(mean_s1 - mean_weak) / mean_weak * 100:.1f}%")

    # === Dashboard Visualization ===
    # Use first prompt as the showcase
    r = weak_results[0]
    n_layers = len(r['U'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) U vs Layer - Internal Energy Monitor
    axes[0, 0].plot(range(n_layers), r['U'], '-', color='#e74c3c', lw=2)
    axes[0, 0].fill_between(range(n_layers), r['U'], alpha=0.1, color='#e74c3c')
    axes[0, 0].axvline(x=L0, color='#f39c12', linestyle='--', lw=2, label=f'L0={L0}')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Internal Energy U')
    axes[0, 0].set_title('(a) Energy Monitor')
    axes[0, 0].legend(fontsize=8)

    # (b) T vs Layer - Temperature Monitor
    axes[0, 1].plot(range(n_layers), r['T'], '-', color='#3498db', lw=2)
    axes[0, 1].fill_between(range(n_layers), r['T'], alpha=0.1, color='#3498db')
    axes[0, 1].axvline(x=L0, color='#f39c12', linestyle='--', lw=2)
    # Mark T_hot and T_cold
    T_hot_idx = r['T'].index(r['T_hot']) if r['T_hot'] in r['T'] else 0
    T_cold_idx = r['T'][1:].index(r['T_cold']) + 1 if r['T_cold'] in r['T'][1:] else 0
    axes[0, 1].annotate('T_hot', xy=(T_hot_idx, r['T_hot']),
                        fontsize=8, color='red', fontweight='bold')
    axes[0, 1].annotate('T_cold', xy=(T_cold_idx, r['T_cold']),
                        fontsize=8, color='blue', fontweight='bold')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Temperature T (nats)')
    axes[0, 1].set_title('(b) Temperature Monitor')

    # (c) S vs Layer - Entropy Monitor
    axes[0, 2].plot(range(n_layers), r['S'], '-', color='#2ecc71', lw=2)
    axes[0, 2].fill_between(range(n_layers), r['S'], alpha=0.1, color='#2ecc71')
    axes[0, 2].axvline(x=L0, color='#f39c12', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Hidden Entropy S')
    axes[0, 2].set_title('(c) Entropy Monitor')

    # (d) dT/dLayer - Anomaly Detection
    dT = r['dT']
    colors_dT = ['#e74c3c' if d > np.mean(dT) + 2 * np.std(dT)
                 else '#95a5a6' for d in dT]
    axes[1, 0].bar(range(len(dT)), dT, color=colors_dT, alpha=0.7)
    axes[1, 0].axhline(y=np.mean(dT) + 2 * np.std(dT), color='red',
                       linestyle='--', label='anomaly threshold')
    axes[1, 0].set_xlabel('Layer transition')
    axes[1, 0].set_ylabel('|dT/dLayer|')
    axes[1, 0].set_title('(d) Phase Transition Detector')
    axes[1, 0].legend(fontsize=8)

    # (e) Phase Portrait: U vs T
    axes[1, 1].plot(r['T'], r['U'], '-o', color='#9b59b6', markersize=4, lw=1.5,
                    alpha=0.7)
    axes[1, 1].plot(r['T'][0], r['U'][0], 'o', color='green', markersize=10,
                    label='start', zorder=5)
    axes[1, 1].plot(r['T'][-1], r['U'][-1], 's', color='red', markersize=10,
                    label='end', zorder=5)
    axes[1, 1].set_xlabel('Temperature T')
    axes[1, 1].set_ylabel('Internal Energy U')
    axes[1, 1].set_title('(e) Phase Portrait (U vs T)')
    axes[1, 1].legend(fontsize=8)

    # (f) Weak vs Strong comparison
    methods = ['Weak\n(continuous)', 'Strong\n(every 4)', 'Strong\n(every 1)']
    means = [mean_weak, mean_s4, mean_s1]
    bar_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = axes[1, 2].bar(methods, means, color=bar_colors, alpha=0.7)
    for bar, val in zip(bars, means):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    axes[1, 2].set_ylabel('Output Entropy (nats)')
    axes[1, 2].set_title('(f) Measurement Impact')

    fig.suptitle("Phase 209: Weak Measurement Dashboard",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase209_weak_measurement')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Weak measurement: {mean_weak:.3f} (ZERO perturbation)")
    print(f"Strong (every 1): {mean_s1:.3f} (+{(mean_s1-mean_weak)/mean_weak*100:.1f}%)")
    print(f"Conclusion: Weak measurement is SAFE for white-box monitoring")
    print(f"{'=' * 70}")

    # Aggregate stats for all prompts
    all_eta = [r['eta'] for r in weak_results]
    all_T_hot = [r['T_hot'] for r in weak_results]
    all_T_cold = [r['T_cold'] for r in weak_results]

    save_results('phase209_weak_measurement', {
        'experiment': 'Weak Measurement Dashboard',
        'comparison': {
            'weak_entropy_mean': mean_weak,
            'strong_every1_mean': mean_s1,
            'strong_every4_mean': mean_s4,
            'weak_perturbation': 0.0,
            'strong1_entropy_increase_pct': (mean_s1 - mean_weak) / mean_weak * 100,
        },
        'thermodynamic_stats': {
            'eta_mean': float(np.mean(all_eta)),
            'T_hot_mean': float(np.mean(all_T_hot)),
            'T_cold_mean': float(np.mean(all_T_cold)),
        },
        'example_profiles': {
            'U': [float(x) for x in weak_results[0]['U']],
            'T': [float(x) for x in weak_results[0]['T']],
            'S': [float(x) for x in weak_results[0]['S']],
        },
    })


if __name__ == '__main__':
    main()
