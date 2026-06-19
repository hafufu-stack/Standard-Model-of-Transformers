# -*- coding: utf-8 -*-
"""
Phase 272: Bose-Einstein Condensation of Tokens
==================================================
When a Bose gas is cooled below T_c, particles collapse into the
ground state. In LLMs, repetition collapse (T->0) may be the analogue.

Measure cosine similarity between ALL token hidden states as T->0.
If BEC occurs, all vectors converge to a single macrostate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

SEED_PROMPT = "The quick brown fox jumps over the lazy dog"
N_ITERATIONS = 15  # iterative feeding rounds to push T->0


def iterative_collapse(model, tok, prompt, device, n_iter=N_ITERATIONS):
    """Feed output back as input repeatedly to induce T->0 collapse."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    iteration_data = []
    current_text = prompt

    for it in range(n_iter):
        inp = tok(current_text, return_tensors='pt', truncation=True,
                  max_length=512).to(device)
        seq_len = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get final layer hidden states for all positions
        hs_final = out.hidden_states[-1][0]  # (seq, hidden)

        # Cosine similarity matrix
        hs_normed = hs_final / (hs_final.norm(dim=-1, keepdim=True) + 1e-10)
        cos_sim = (hs_normed @ hs_normed.T).cpu().float().numpy()

        # Mean pairwise cosine similarity (upper triangle)
        n = cos_sim.shape[0]
        if n > 1:
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            mean_cos = float(cos_sim[mask].mean())
            std_cos = float(cos_sim[mask].std())
        else:
            mean_cos, std_cos = 1.0, 0.0

        # Temperature at final layer
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Participation ratio of hidden state distribution
        hs_sq = (hs_final.float() ** 2).mean(dim=0)  # mean over positions
        hs_prob = hs_sq / (hs_sq.sum() + 1e-10)
        pr = 1.0 / (hs_prob ** 2).sum().item()

        # Phase coherence: std of vector angles relative to mean vector
        mean_vec = hs_final.float().mean(dim=0)
        angles = []
        for pos in range(n):
            cos_angle = torch.dot(hs_final[pos].float(), mean_vec) / (
                hs_final[pos].float().norm() * mean_vec.norm() + 1e-10)
            angles.append(cos_angle.item())
        phase_coherence = float(np.mean(angles))

        iteration_data.append({
            'iteration': it,
            'seq_len': seq_len,
            'T': round(t_val, 4),
            'P1': round(p1, 4),
            'mean_cos_sim': round(mean_cos, 6),
            'std_cos_sim': round(std_cos, 6),
            'PR': round(pr, 2),
            'phase_coherence': round(phase_coherence, 6),
            'text_preview': current_text[:100],
        })

        print(f"  Iter {it}: T={t_val:.3f}, P1={p1:.3f}, "
              f"cos_sim={mean_cos:.4f}, coherence={phase_coherence:.4f}")

        # Generate next tokens and feed back
        with torch.no_grad():
            gen = model.generate(inp['input_ids'], max_new_tokens=30,
                                do_sample=False)
        new_text = tok.decode(gen[0], skip_special_tokens=True)
        current_text = new_text

    return iteration_data


def main():
    print("=" * 70)
    print("Phase 272: Bose-Einstein Condensation of Tokens")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        data = iterative_collapse(model, tok, SEED_PROMPT, device)
        all_results[size] = data
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        c = colors[size]
        iters = [d['iteration'] for d in data]

        # (a) Temperature collapse
        axes[0, 0].plot(iters, [d['T'] for d in data], '-o', color=c,
                       lw=2, markersize=5, label=size)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Temperature Collapse', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Cosine similarity increase
    for size, data in all_results.items():
        c = colors[size]
        iters = [d['iteration'] for d in data]
        axes[0, 1].plot(iters, [d['mean_cos_sim'] for d in data], '-o', color=c,
                       lw=2, markersize=5, label=size)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Mean Cosine Similarity')
    axes[0, 1].set_title('(b) Token Vector Condensation', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Phase coherence
    for size, data in all_results.items():
        c = colors[size]
        iters = [d['iteration'] for d in data]
        axes[0, 2].plot(iters, [d['phase_coherence'] for d in data], '-o', color=c,
                       lw=2, markersize=5, label=size)
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Phase Coherence')
    axes[0, 2].set_title('(c) Phase Coherence (BEC Order Parameter)', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) T vs cos_sim (phase diagram)
    for size, data in all_results.items():
        c = colors[size]
        axes[1, 0].scatter([d['T'] for d in data],
                          [d['mean_cos_sim'] for d in data],
                          c=c, s=60, label=size, zorder=5)
        # Connect with line
        axes[1, 0].plot([d['T'] for d in data],
                       [d['mean_cos_sim'] for d in data],
                       '-', color=c, alpha=0.5)
    axes[1, 0].set_xlabel('Temperature T')
    axes[1, 0].set_ylabel('Mean Cosine Similarity')
    axes[1, 0].set_title('(d) BEC Phase Diagram', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) P1 trajectory
    for size, data in all_results.items():
        c = colors[size]
        iters = [d['iteration'] for d in data]
        axes[1, 1].plot(iters, [d['P1'] for d in data], '-o', color=c,
                       lw=2, markersize=5, label=size)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('P1 (max probability)')
    axes[1, 1].set_title('(e) P1 Trajectory During Collapse', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "BOSE-EINSTEIN CONDENSATION\n\n"
    for size, data in all_results.items():
        t0, tf = data[0]['T'], data[-1]['T']
        c0, cf = data[0]['mean_cos_sim'], data[-1]['mean_cos_sim']
        summary += f"{size}:\n"
        summary += f"  T: {t0:.2f} -> {tf:.2f}\n"
        summary += f"  cos_sim: {c0:.4f} -> {cf:.4f}\n"
        summary += f"  BEC: {'YES' if cf > 0.8 else 'PARTIAL' if cf > 0.5 else 'NO'}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 272: Bose-Einstein Condensation of Tokens",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase272_bec')
    plt.close()

    save_results('phase272_bec', {
        'experiment': 'Bose-Einstein Condensation of Tokens',
        'seed_prompt': SEED_PROMPT,
        'n_iterations': N_ITERATIONS,
        'results': all_results,
    })


if __name__ == '__main__':
    main()
