# -*- coding: utf-8 -*-
"""
Phase 268: Token Position Independence of P1*T
=================================================
All previous measurements used the LAST token position.
Question: Does P1*T conservation hold at EVERY token position,
or is it specific to the prediction position?

This tests the universality of P1*T across the sequence dimension.
If it holds everywhere, the law applies to the model itself.
If it holds only at the last position, it's about autoregressive prediction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPT = "The fundamental theorem of calculus states that differentiation and integration are inverse operations"


def measure_all_positions(model, tok, device, model_name):
    """Measure P1*T at every token position across all layers."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(PROMPT, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states)

    # PRT[layer][position]
    PRT_map = np.zeros((n_layers, seq_len))
    T_map = np.zeros((n_layers, seq_len))
    P1_map = np.zeros((n_layers, seq_len))

    for li, hs in enumerate(out.hidden_states):
        for pos in range(seq_len):
            with torch.no_grad():
                normed = norm_layer(hs[:, pos:pos+1, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            p1 = float(probs.max().item())
            t_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(t_sm): t_sm = 0

            P1_map[li, pos] = p1
            T_map[li, pos] = t_sm
            PRT_map[li, pos] = p1 * t_sm

    # CV per position (across layers, skipping layer 0)
    cv_per_pos = []
    for pos in range(seq_len):
        profile = PRT_map[1:, pos]
        cv = float(np.std(profile) / (np.mean(profile) + 1e-10))
        cv_per_pos.append(cv)

    # CV per layer (across positions)
    cv_per_layer = []
    for li in range(n_layers):
        profile = PRT_map[li, :]
        cv = float(np.std(profile) / (np.mean(profile) + 1e-10))
        cv_per_layer.append(cv)

    # Mean PRT per position
    mean_prt_per_pos = PRT_map[1:, :].mean(axis=0).tolist()

    # Arrow per position
    arrow_per_pos = []
    for pos in range(seq_len):
        rho, _ = stats.spearmanr(range(n_layers), T_map[:, pos])
        arrow_per_pos.append(float(rho))

    tokens = tok.convert_ids_to_tokens(inp['input_ids'][0].tolist())

    return {
        'model': model_name,
        'seq_len': seq_len,
        'n_layers': n_layers,
        'cv_per_pos': cv_per_pos,
        'cv_per_layer': cv_per_layer,
        'mean_prt_per_pos': mean_prt_per_pos,
        'arrow_per_pos': arrow_per_pos,
        'PRT_map': PRT_map,
        'T_map': T_map,
        'tokens': tokens,
        'last_pos_cv': cv_per_pos[-1],
        'mean_pos_cv': round(float(np.mean(cv_per_pos)), 4),
        'median_pos_cv': round(float(np.median(cv_per_pos)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 268: Token Position Independence of P1*T")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_all_positions(model, tok, device, size)
        results[size] = r
        print(f"  Seq len = {r['seq_len']}, Layers = {r['n_layers']}")
        print(f"  Last-pos CV = {r['last_pos_cv']:.4f}")
        print(f"  Mean-pos CV = {r['mean_pos_cv']:.4f}")
        print(f"  Median-pos CV = {r['median_pos_cv']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        c = colors[size]

        # (a) PRT heatmap (just first model)
        if size == '0.5B':
            im = axes[0, 0].imshow(r['PRT_map'][1:], aspect='auto', cmap='viridis',
                                   interpolation='nearest')
            axes[0, 0].set_xlabel('Token Position')
            axes[0, 0].set_ylabel('Layer')
            axes[0, 0].set_title('(a) P1*T Heatmap (0.5B)', fontweight='bold')
            plt.colorbar(im, ax=axes[0, 0], label='P1*T')

        # (b) CV per position
        axes[0, 1].plot(range(r['seq_len']), r['cv_per_pos'], '-', color=c, lw=2,
                       label=f'{size} (mean={r["mean_pos_cv"]:.3f})')
        axes[0, 1].axhline(r['last_pos_cv'], color=c, ls='--', alpha=0.5,
                          label=f'Last pos ({size})')

    axes[0, 1].set_xlabel('Token Position')
    axes[0, 1].set_ylabel('CV(P1*T) across layers')
    axes[0, 1].set_title('(b) Conservation at Each Position', fontweight='bold')
    axes[0, 1].legend(fontsize=6); axes[0, 1].grid(alpha=0.3)

    # (c) Mean PRT per position
    for size, r in results.items():
        c = colors[size]
        axes[0, 2].plot(range(r['seq_len']), r['mean_prt_per_pos'], '-', color=c,
                       lw=2, label=size)
    axes[0, 2].set_xlabel('Token Position')
    axes[0, 2].set_ylabel('Mean P1*T')
    axes[0, 2].set_title('(c) P1*T Value at Each Position', fontweight='bold')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    # (d) Arrow per position
    for size, r in results.items():
        c = colors[size]
        axes[1, 0].plot(range(r['seq_len']), r['arrow_per_pos'], '-', color=c,
                       lw=2, label=size)
    axes[1, 0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('Token Position')
    axes[1, 0].set_ylabel('Arrow of Time (rho)')
    axes[1, 0].set_title('(d) Arrow at Each Position', fontweight='bold')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    # (e) T heatmap (0.5B)
    if '0.5B' in results:
        r = results['0.5B']
        im2 = axes[1, 1].imshow(r['T_map'][1:], aspect='auto', cmap='coolwarm',
                                interpolation='nearest')
        axes[1, 1].set_xlabel('Token Position')
        axes[1, 1].set_ylabel('Layer')
        axes[1, 1].set_title('(e) Temperature Heatmap (0.5B)', fontweight='bold')
        plt.colorbar(im2, ax=axes[1, 1], label='T_sm')

    # (f) Summary
    summary = "TOKEN POSITION INDEPENDENCE\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Last-pos CV: {r['last_pos_cv']:.4f}\n"
        summary += f"  Mean-pos CV: {r['mean_pos_cv']:.4f}\n"
        summary += f"  Median-pos CV: {r['median_pos_cv']:.4f}\n\n"

    position_independent = all(
        r['mean_pos_cv'] < 0.3 for r in results.values()
    )
    summary += f"Position independent: {'YES' if position_independent else 'NO'}"

    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 268: Is P1*T Conserved at Every Token Position?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase268_token_position')
    plt.close()

    save_results('phase268_token_position', {
        'experiment': 'Token Position Independence',
        'results': {k: {kk: vv for kk, vv in v.items()
                       if kk not in ('PRT_map', 'T_map', 'cv_per_layer')}
                   for k, v in results.items()},
    })


if __name__ == '__main__':
    main()
