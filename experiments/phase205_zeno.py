# -*- coding: utf-8 -*-
"""
Phase 205: Quantum Zeno Freezing
==================================
Quantum Zeno effect: frequent measurement halts state evolution.

Force "observation" (RMSNorm + LM_Head projection) at every layer.
Feed the argmax token embedding back as the hidden state.

Does this "collapse the wavefunction" at each layer and freeze the
computation, producing nonsense?
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
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]

L0 = 21
# Measurement frequencies: every N layers (1 = every layer = max Zeno)
MEASURE_INTERVALS = [0, 1, 2, 4, 7, 14, 28]  # 0 = no measurement (baseline)


def run_with_zeno(model, tok, device, prompt, measure_every=0):
    """Run forward pass, optionally forcing observation at every N layers.
    
    'Observation' = RMSNorm + LM_Head + argmax + re-embed.
    This collapses the continuous hidden state to a discrete token.
    """
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head
    embed_tokens = model.model.embed_tokens

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']

    with torch.no_grad():
        # Embedding
        hidden = model.model.embed_tokens(input_ids)

        U_list = [hidden[0, -1, :].float().norm().item()]
        T_list = []
        S_list = []
        observed_tokens = []
        n_collapses = 0

        # Initial T
        normed = norm_layer(hidden[:, -1:, :])
        logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_list.append(T_val if not np.isnan(T_val) else 0)

        h = hidden[0, -1, :].float()
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S_list.append(-(h_prob * torch.log(h_prob + 1e-10)).sum().item())

        # Compute position embeddings once (needed by Qwen2 layers)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        for li in range(n_layers):
            layer = model.model.layers[li]
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Zeno measurement: force observation
            if measure_every > 0 and (li + 1) % measure_every == 0 and li < n_layers - 1:
                # "Collapse" - project to token space and re-embed
                normed_h = norm_layer(hidden[:, -1:, :])
                collapse_logits = lm_head(normed_h).squeeze()
                collapsed_token_id = collapse_logits.argmax().item()
                observed_tokens.append(tok.decode(collapsed_token_id))

                # Re-embed: replace last position with the embedding of collapsed token
                collapsed_embed = embed_tokens(
                    torch.tensor([[collapsed_token_id]], device=device)
                )
                # Replace only the last-token hidden state
                hidden = hidden.clone()
                hidden[0, -1, :] = collapsed_embed[0, 0, :]
                n_collapses += 1

            # Measure thermodynamics
            h = hidden[0, -1, :].float()
            U_list.append(h.norm().item())

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_list.append(-(h_prob * torch.log(h_prob + 1e-10)).sum().item())

            normed = norm_layer(hidden[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T_val if not np.isnan(T_val) else 0)

        # Final prediction
        normed = norm_layer(hidden)
        final_logits = lm_head(normed)
        final_probs = torch.softmax(final_logits[0, -1, :].float(), dim=-1)
        output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
        top_token = tok.decode(final_logits[0, -1, :].argmax().item())
        top1_prob = final_probs.max().item()

    T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
    T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0
    dS = S_list[-1] - S_list[0]

    return {
        'U': U_list, 'T': T_list, 'S': S_list,
        'eta': eta, 'dS': dS,
        'output_entropy': output_entropy,
        'top_token': top_token, 'top1_prob': top1_prob,
        'n_collapses': n_collapses,
        'observed_tokens': observed_tokens,
    }


def main():
    print("=" * 70)
    print("Phase 205: Quantum Zeno Freezing")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    results_by_interval = {}
    for mi, interval in enumerate(MEASURE_INTERVALS):
        label = f"every_{interval}" if interval > 0 else "baseline"
        print(f"\n[{mi+1}/{len(MEASURE_INTERVALS)}] "
              f"{'Baseline (no measurement)' if interval == 0 else f'Measure every {interval} layers'}")

        all_eta = []
        all_ent = []
        all_dS = []
        all_top1 = []
        all_tokens = []

        for prompt in PROMPTS:
            r = run_with_zeno(model, tok, device, prompt, measure_every=interval)
            all_eta.append(r['eta'])
            all_ent.append(r['output_entropy'])
            all_dS.append(r['dS'])
            all_top1.append(r['top1_prob'])
            all_tokens.append(r['top_token'])

        results_by_interval[label] = {
            'interval': interval,
            'eta_mean': float(np.mean(all_eta)),
            'eta_std': float(np.std(all_eta)),
            'entropy_mean': float(np.mean(all_ent)),
            'dS_mean': float(np.mean(all_dS)),
            'top1_mean': float(np.mean(all_top1)),
            'top_tokens': all_tokens,
        }
        print(f"  eta={np.mean(all_eta):.4f}, entropy={np.mean(all_ent):.3f}, "
              f"top1={np.mean(all_top1):.4f}")

    # === Get profile examples for plotting ===
    example_profiles = {}
    for interval in [0, 1, 4, 14]:
        r = run_with_zeno(model, tok, device, PROMPTS[0], measure_every=interval)
        label = f"every_{interval}" if interval > 0 else "baseline"
        example_profiles[label] = r

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    intervals_plot = [r['interval'] for r in results_by_interval.values()]
    intervals_labels = ['None'] + [str(i) for i in intervals_plot[1:]]

    # (a) eta vs measurement frequency
    etas = [r['eta_mean'] for r in results_by_interval.values()]
    axes[0, 0].plot(range(len(etas)), etas, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[0, 0].set_xticks(range(len(intervals_labels)))
    axes[0, 0].set_xticklabels(intervals_labels, fontsize=8)
    axes[0, 0].set_xlabel('Measurement Interval (layers)')
    axes[0, 0].set_ylabel('Carnot Efficiency eta')
    axes[0, 0].set_title('(a) Zeno Effect on Efficiency')

    # (b) Output entropy vs measurement frequency
    ents = [r['entropy_mean'] for r in results_by_interval.values()]
    axes[0, 1].plot(range(len(ents)), ents, 's-', color='#3498db', markersize=8, linewidth=2)
    axes[0, 1].set_xticks(range(len(intervals_labels)))
    axes[0, 1].set_xticklabels(intervals_labels, fontsize=8)
    axes[0, 1].set_xlabel('Measurement Interval (layers)')
    axes[0, 1].set_ylabel('Output Entropy (nats)')
    axes[0, 1].set_title('(b) Zeno Effect on Output Quality')

    # (c) Top-1 probability vs measurement frequency
    top1s = [r['top1_mean'] for r in results_by_interval.values()]
    axes[0, 2].plot(range(len(top1s)), top1s, 'D-', color='#2ecc71', markersize=8, linewidth=2)
    axes[0, 2].set_xticks(range(len(intervals_labels)))
    axes[0, 2].set_xticklabels(intervals_labels, fontsize=8)
    axes[0, 2].set_xlabel('Measurement Interval (layers)')
    axes[0, 2].set_ylabel('Top-1 Token Probability')
    axes[0, 2].set_title('(c) Confidence under Zeno')

    # (d) T profiles for different measurement rates
    profile_colors = {'baseline': '#2ecc71', 'every_1': '#e74c3c',
                      'every_4': '#3498db', 'every_14': '#9b59b6'}
    for label, r in example_profiles.items():
        axes[1, 0].plot(range(len(r['T'])), r['T'], '-', color=profile_colors[label],
                        linewidth=1.5, label=label, alpha=0.8)
    axes[1, 0].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Temperature T')
    axes[1, 0].set_title('(d) Temperature Profiles')
    axes[1, 0].legend(fontsize=7)

    # (e) S profiles
    for label, r in example_profiles.items():
        axes[1, 1].plot(range(len(r['S'])), r['S'], '-', color=profile_colors[label],
                        linewidth=1.5, label=label, alpha=0.8)
    axes[1, 1].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Hidden Entropy S')
    axes[1, 1].set_title('(e) Entropy Profiles')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    baseline = results_by_interval['baseline']
    zeno_max = results_by_interval['every_1']
    summary_text = (
        f"Quantum Zeno Freezing\n\n"
        f"Baseline (no measurement):\n"
        f"  eta = {baseline['eta_mean']:.4f}\n"
        f"  entropy = {baseline['entropy_mean']:.3f}\n"
        f"  top1 = {baseline['top1_mean']:.4f}\n\n"
        f"Max Zeno (every layer):\n"
        f"  eta = {zeno_max['eta_mean']:.4f}\n"
        f"  entropy = {zeno_max['entropy_mean']:.3f}\n"
        f"  top1 = {zeno_max['top1_mean']:.4f}\n\n"
        f"Entropy increase:\n"
        f"  {(zeno_max['entropy_mean']-baseline['entropy_mean'])/baseline['entropy_mean']*100:.1f}%\n"
        f"Zeno freezes thought:\n"
        f"  {'YES' if zeno_max['entropy_mean'] > baseline['entropy_mean'] * 1.5 else 'PARTIAL'}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 205: Quantum Zeno Freezing", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase205_zeno')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Baseline eta: {baseline['eta_mean']:.4f}")
    print(f"Zeno (every layer) eta: {zeno_max['eta_mean']:.4f}")
    print(f"Entropy increase: "
          f"{(zeno_max['entropy_mean']-baseline['entropy_mean'])/baseline['entropy_mean']*100:.1f}%")
    print(f"{'=' * 70}")

    save_results('phase205_zeno', {
        'experiment': 'Quantum Zeno Freezing',
        'results': results_by_interval,
        'summary': {
            'baseline_eta': baseline['eta_mean'],
            'zeno_eta': zeno_max['eta_mean'],
            'baseline_entropy': baseline['entropy_mean'],
            'zeno_entropy': zeno_max['entropy_mean'],
            'entropy_increase_pct': (zeno_max['entropy_mean'] - baseline['entropy_mean']) / baseline['entropy_mean'] * 100,
        }
    })


if __name__ == '__main__':
    main()
