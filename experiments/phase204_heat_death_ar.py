# -*- coding: utf-8 -*-
"""
Phase 204: Autoregressive Heat Death
======================================
Track thermodynamic observables during long-form autoregressive generation.
As KV cache grows and context lengthens, does the model approach "heat death"
(eta -> 0, repetition collapse)?

Measure eta_t, L0_t, and S_t at each generation step t.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

SEED_PROMPTS = [
    "In the beginning, the universe was a singularity of infinite density. Over billions of years,",
    "The history of artificial intelligence began with a simple question about whether machines could think. Early pioneers",
]

MAX_NEW_TOKENS = 512  # Long enough to see degradation
MEASURE_EVERY = 4     # Measure thermodynamics every N tokens
L0 = 21


def measure_thermo_at_step(model, input_ids, device):
    """Measure thermodynamics for current input_ids (full context)."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    U_list, T_list, S_list = [], [], []
    for li, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        U_list.append(h.norm().item())

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S_list.append(-(h_prob * torch.log(h_prob + 1e-10)).sum().item())

        normed = norm_layer(hs[:, -1:, :])
        logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_list.append(T_val if not np.isnan(T_val) else 0)

    T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
    T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

    # Output distribution stats
    final_logits = out.logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
    top1_prob = final_probs.max().item()

    return {
        'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
        'U_mean': np.mean(U_list),
        'S_mean': np.mean(S_list),
        'T_mean': np.mean(T_list),
        'output_entropy': output_entropy,
        'top1_prob': top1_prob,
    }


def generate_with_tracking(model, tok, device, prompt):
    """Generate tokens one-by-one, tracking thermodynamics."""
    input_ids = tok(prompt, return_tensors='pt').input_ids.to(device)
    initial_len = input_ids.shape[1]

    trajectory = []
    generated_tokens = []
    repetition_counts = []

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(input_ids)
        next_logits = out.logits[0, -1, :]

        # Greedy decode
        next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        token_text = tok.decode(next_token[0].item())
        generated_tokens.append(token_text)

        # Count recent repetitions (last 20 tokens)
        recent = generated_tokens[-20:]
        if len(recent) >= 4:
            # Check if last 4 tokens repeat
            rep_count = 0
            pattern = recent[-4:]
            for i in range(0, len(recent) - 4):
                if recent[i:i+4] == pattern:
                    rep_count += 1
            repetition_counts.append(rep_count)
        else:
            repetition_counts.append(0)

        # Measure thermodynamics periodically
        if step % MEASURE_EVERY == 0:
            thermo = measure_thermo_at_step(model, input_ids, device)
            thermo['step'] = step
            thermo['context_length'] = input_ids.shape[1]
            thermo['repetition_score'] = repetition_counts[-1]
            trajectory.append(thermo)
            if step % 50 == 0:
                print(f"    Step {step}: eta={thermo['eta']:.3f}, "
                      f"S={thermo['S_mean']:.2f}, ent={thermo['output_entropy']:.2f}, "
                      f"rep={repetition_counts[-1]}")

        # Early stop if stuck in loop
        if len(repetition_counts) > 20 and all(r > 2 for r in repetition_counts[-10:]):
            print(f"    Repetition collapse detected at step {step}!")
            break

    generated_text = tok.decode(input_ids[0, initial_len:].cpu())
    return trajectory, generated_text, repetition_counts


def main():
    print("=" * 70)
    print("Phase 204: Autoregressive Heat Death")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    all_trajectories = []
    all_texts = []

    for pi, prompt in enumerate(SEED_PROMPTS):
        print(f"\n[{pi+1}/{len(SEED_PROMPTS)}] Generating from: '{prompt[:50]}...'")
        traj, text, rep_counts = generate_with_tracking(model, tok, device, prompt)
        all_trajectories.append(traj)
        all_texts.append(text[:200])  # Save first 200 chars
        print(f"  Generated {len(text)} chars, {len(traj)} measurements")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['#e74c3c', '#3498db']

    for ti, traj in enumerate(all_trajectories):
        steps = [t['step'] for t in traj]
        etas = [t['eta'] for t in traj]
        output_ents = [t['output_entropy'] for t in traj]
        S_means = [t['S_mean'] for t in traj]
        U_means = [t['U_mean'] for t in traj]
        top1s = [t['top1_prob'] for t in traj]
        reps = [t['repetition_score'] for t in traj]

        # (a) eta over time
        axes[0, 0].plot(steps, etas, '-', color=colors[ti], linewidth=1.5,
                        label=f'Prompt {ti+1}', alpha=0.8)

        # (b) Output entropy over time
        axes[0, 1].plot(steps, output_ents, '-', color=colors[ti], linewidth=1.5,
                        label=f'Prompt {ti+1}', alpha=0.8)

        # (c) Top-1 probability over time
        axes[0, 2].plot(steps, top1s, '-', color=colors[ti], linewidth=1.5,
                        label=f'Prompt {ti+1}', alpha=0.8)

        # (d) Mean S over time
        axes[1, 0].plot(steps, S_means, '-', color=colors[ti], linewidth=1.5,
                        label=f'Prompt {ti+1}', alpha=0.8)

        # (e) Repetition score
        axes[1, 1].plot(steps, reps, '-', color=colors[ti], linewidth=1.5,
                        label=f'Prompt {ti+1}', alpha=0.8)

    axes[0, 0].set_xlabel('Generation Step')
    axes[0, 0].set_ylabel('Carnot Efficiency eta')
    axes[0, 0].set_title('(a) Efficiency Over Time')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('Output Entropy (nats)')
    axes[0, 1].set_title('(b) Output Entropy Over Time')
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].set_xlabel('Generation Step')
    axes[0, 2].set_ylabel('Top-1 Token Probability')
    axes[0, 2].set_title('(c) Confidence Over Time')
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].set_xlabel('Generation Step')
    axes[1, 0].set_ylabel('Mean Hidden Entropy S')
    axes[1, 0].set_title('(d) Internal Entropy Over Time')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel('Generation Step')
    axes[1, 1].set_ylabel('Repetition Score')
    axes[1, 1].set_title('(e) Repetition Detection')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    # Compare first vs last measurement
    for ti, traj in enumerate(all_trajectories):
        if len(traj) >= 2:
            eta_start = traj[0]['eta']
            eta_end = traj[-1]['eta']
            ent_start = traj[0]['output_entropy']
            ent_end = traj[-1]['output_entropy']

    summary_text = (
        f"Autoregressive Heat Death\n\n"
        f"Prompt 1:\n"
        f"  eta: {all_trajectories[0][0]['eta']:.3f} -> "
        f"{all_trajectories[0][-1]['eta']:.3f}\n"
        f"  entropy: {all_trajectories[0][0]['output_entropy']:.2f} -> "
        f"{all_trajectories[0][-1]['output_entropy']:.2f}\n"
        f"  steps: {len(all_trajectories[0])}\n\n"
        f"Prompt 2:\n"
        f"  eta: {all_trajectories[1][0]['eta']:.3f} -> "
        f"{all_trajectories[1][-1]['eta']:.3f}\n"
        f"  entropy: {all_trajectories[1][0]['output_entropy']:.2f} -> "
        f"{all_trajectories[1][-1]['output_entropy']:.2f}\n"
        f"  steps: {len(all_trajectories[1])}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 204: Autoregressive Heat Death", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase204_heat_death')
    plt.close()

    print(f"\n{'=' * 70}")
    for ti, traj in enumerate(all_trajectories):
        print(f"Prompt {ti+1}: eta {traj[0]['eta']:.3f} -> {traj[-1]['eta']:.3f}")
    print(f"{'=' * 70}")

    save_results('phase204_heat_death', {
        'experiment': 'Autoregressive Heat Death',
        'trajectories': [[{k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in t.items()} for t in traj]
                        for traj in all_trajectories],
        'generated_texts': all_texts,
        'summary': {
            'eta_start_mean': np.mean([t[0]['eta'] for t in all_trajectories]),
            'eta_end_mean': np.mean([t[-1]['eta'] for t in all_trajectories]),
            'entropy_start_mean': np.mean([t[0]['output_entropy'] for t in all_trajectories]),
            'entropy_end_mean': np.mean([t[-1]['output_entropy'] for t in all_trajectories]),
        }
    })


if __name__ == '__main__':
    main()
