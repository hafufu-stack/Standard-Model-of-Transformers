# -*- coding: utf-8 -*-
"""
Phase 22: Gravitational Lensing (Opus Original)
=================================================
If Attention is gravity, then information "near" a highly-attended
token should be "bent" - deflected toward that token's representation.
Measure the angular deflection of hidden states as they pass
through layers where high-attention tokens act as gravitational lenses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 22: Gravitational Lensing of Information")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    # Test prompts with clear "gravitational mass" tokens
    prompts = [
        "The president of the United States signed the new bill into law",
        "Einstein discovered that energy equals mass times the speed of light squared",
        "The cat sat on the mat and the dog ran in the park yesterday",
        "Python is a programming language used for machine learning tasks",
    ]

    all_lensing = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        n_tokens = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get hidden states for ALL tokens at each layer
        hs = []
        for layer_idx in range(len(out.hidden_states)):
            h = out.hidden_states[layer_idx][0, :, :].float().cpu()  # (seq, hidden)
            hs.append(h)

        # Find the "gravitationally heavy" token: highest L2 norm at final layer
        final_norms = hs[-1].norm(dim=1)
        heavy_idx = torch.argmax(final_norms).item()
        heavy_token = tok.decode(inp['input_ids'][0, heavy_idx].item())

        print(f"\n  Prompt: '{prompt[:50]}...'")
        print(f"  Heavy token: '{heavy_token}' (idx={heavy_idx})")

        # Measure "lensing": how much each token's trajectory bends
        # toward the heavy token across layers
        lensing_per_layer = []
        for l in range(1, len(hs)):
            cos_sims = []
            for t in range(n_tokens):
                if t == heavy_idx:
                    continue
                # Direction of movement
                delta = hs[l][t] - hs[l-1][t]
                # Direction toward heavy token
                toward_heavy = hs[l][heavy_idx] - hs[l][t]
                # Cosine between movement and "toward heavy"
                d_norm = delta.norm()
                t_norm = toward_heavy.norm()
                if d_norm > 1e-6 and t_norm > 1e-6:
                    cos = torch.dot(delta, toward_heavy) / (d_norm * t_norm)
                    cos_sims.append(cos.item())
            avg_cos = np.mean(cos_sims) if cos_sims else 0
            lensing_per_layer.append(avg_cos)

        all_lensing.append({
            'prompt': prompt[:50],
            'heavy_token': heavy_token,
            'heavy_idx': heavy_idx,
            'lensing': lensing_per_layer,
        })

        # Summary
        avg_lens = np.mean(lensing_per_layer)
        print(f"  Avg lensing (cos toward heavy): {avg_lens:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    for idx, res in enumerate(all_lensing):
        ax.plot(range(len(res['lensing'])), res['lensing'], 'o-', ms=3,
                label=f"'{res['heavy_token']}'")
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cos(movement, toward heavy)')
    ax.set_title('(a) Gravitational Lensing per Layer')
    ax.legend(fontsize=7)

    # Average lensing profile
    ax = axes[1]
    min_len = min(len(r['lensing']) for r in all_lensing)
    avg_profile = np.mean([r['lensing'][:min_len] for r in all_lensing], axis=0)
    ax.fill_between(range(min_len), avg_profile, alpha=0.3, color='#e74c3c')
    ax.plot(range(min_len), avg_profile, 'o-', color='#e74c3c', ms=4)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Lensing')
    ax.set_title('(b) Mean Gravitational Lensing Profile')

    # Lensing vs distance from heavy token
    ax = axes[2]
    for res in all_lensing:
        prompt_tokens = len(res['lensing']) + 1
        heavy = res['heavy_idx']
        # At final layer, check if closer tokens are more lensed
        # (we don't have per-token lensing here, so use layer-averaged)
        pass
    # Instead: show heavy token norm growth
    for prompt in prompts[:2]:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        final_norms = out.hidden_states[-1][0, :, :].float().cpu().norm(dim=1)
        heavy_idx = torch.argmax(final_norms).item()
        norms_across_layers = []
        for l in range(len(out.hidden_states)):
            n = out.hidden_states[l][0, heavy_idx, :].float().cpu().norm().item()
            norms_across_layers.append(n)
        ax.plot(range(len(norms_across_layers)), norms_across_layers, 'o-', ms=3,
                label=tok.decode(inp['input_ids'][0, heavy_idx].item()))
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm of Heavy Token')
    ax.set_title('(c) Gravitational Mass Growth')
    ax.legend(fontsize=7)

    overall_lens = np.mean(avg_profile)
    fig.suptitle(
        f"Phase 22: Gravitational Lensing\n"
        f"Mean lensing = {overall_lens:.4f} | "
        f"{'POSITIVE (attractive)' if overall_lens > 0 else 'NEGATIVE (repulsive)'}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase22_gravitational_lensing")
    plt.close()

    if overall_lens > 0.05:
        verdict = (f"GRAVITATIONAL LENSING CONFIRMED: mean lensing={overall_lens:.4f}. "
                   f"Information trajectories bend toward heavy tokens!")
    elif overall_lens > 0:
        verdict = (f"WEAK LENSING: mean={overall_lens:.4f}. "
                   f"Slight attraction toward massive tokens.")
    else:
        verdict = (f"NO LENSING: mean={overall_lens:.4f}. "
                   f"Information does not bend toward heavy tokens.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 22: Gravitational Lensing',
        'summary': {'verdict': verdict, 'mean_lensing': float(overall_lens)},
    }
    save_results("phase22_gravitational_lensing", result)
    return result


if __name__ == '__main__':
    main()
