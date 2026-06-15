# -*- coding: utf-8 -*-
"""
Phase 240: Token Position Thermodynamics
==========================================
Measure thermodynamics at ALL token positions simultaneously.
The "thermodynamic field" across the sequence reveals how information
flows and crystallizes at different positions.
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
    "The cat sat on the mat and looked at the window",
    "Quantum physics explains the behavior of particles at atomic scales",
    "In nineteen sixty nine humans first walked on the surface of the moon",
    "Machine learning algorithms discover hidden patterns in large datasets",
    "The quick brown fox jumps over the lazy dog near the river",
]


def position_field(model, tok, device, model_name):
    """Measure T, P1, U at every (layer, position) pair."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_fields = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        tokens = [tok.decode(inp['input_ids'][0, i]) for i in range(seq_len)]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure at every (layer, position)
        T_field = np.zeros((len(out.hidden_states), seq_len))
        P1_field = np.zeros((len(out.hidden_states), seq_len))
        U_field = np.zeros((len(out.hidden_states), seq_len))

        for li, hs in enumerate(out.hidden_states):
            for pi in range(seq_len):
                h = hs[0, pi, :].float()
                U_field[li, pi] = h.norm().item()
                with torch.no_grad():
                    normed = norm_layer(hs[:, pi:pi+1, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                P1_field[li, pi] = float(probs.max().item())
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_field[li, pi] = float(S) if not np.isnan(S) else 0

        all_fields.append({
            'prompt': prompt[:50],
            'tokens': tokens,
            'seq_len': seq_len,
            'T_field': T_field.tolist(),
            'P1_field': P1_field.tolist(),
            'U_field': U_field.tolist(),
        })

    return {
        'model': model_name,
        'n_layers': n_layers,
        'fields': all_fields,
    }


def main():
    print("=" * 70)
    print("Phase 240: Token Position Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = position_field(model, tok, device, size)
        results[size] = r
        for field in r['fields']:
            T_arr = np.array(field['T_field'])
            print(f"  '{field['prompt'][:30]}...' [{field['seq_len']}tok x {T_arr.shape[0]}L]")
            print(f"    T range: {T_arr.min():.2f} - {T_arr.max():.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    # Use first prompt for detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for si, (size, r) in enumerate(results.items()):
        field = r['fields'][0]  # First prompt
        T_arr = np.array(field['T_field'])
        P1_arr = np.array(field['P1_field'])
        tokens = field['tokens']

        # (a,b) T heatmap
        ax = axes[0, si]
        im = ax.imshow(T_arr, aspect='auto', cmap='hot', origin='lower')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title(f'({chr(97+si)}) T Field ({size})')
        # Token labels on x-axis
        if len(tokens) <= 15:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, fontsize=5, rotation=45, ha='right')
        fig.colorbar(im, ax=ax, shrink=0.7)

    # (c) T at last position vs all positions (1.5B)
    field15 = results[list(results.keys())[-1]]['fields'][0]
    T_arr = np.array(field15['T_field'])
    ax = axes[0, 2]
    # Average T across all positions
    T_mean_pos = T_arr.mean(axis=1)
    T_last_pos = T_arr[:, -1]
    T_first_pos = T_arr[:, 0]
    ax.plot(range(len(T_mean_pos)), T_mean_pos, '-', color='steelblue', lw=2, label='Mean(all pos)')
    ax.plot(range(len(T_last_pos)), T_last_pos, '-', color='coral', lw=2, label='Last pos')
    ax.plot(range(len(T_first_pos)), T_first_pos, '-', color='green', lw=2, label='First pos')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Temperature')
    ax.set_title('(c) T by Position')
    ax.legend(fontsize=7)

    # (d) P1 heatmap (1.5B, first prompt)
    P1_arr = np.array(field15['P1_field'])
    im = axes[1, 0].imshow(P1_arr, aspect='auto', cmap='viridis', origin='lower')
    axes[1, 0].set_xlabel('Token Position')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_title('(d) P1 Field (1.5B)')
    fig.colorbar(im, ax=axes[1, 0], shrink=0.7)

    # (e) Position-averaged profiles across prompts
    for pi, field in enumerate(results[list(results.keys())[-1]]['fields']):
        T_arr = np.array(field['T_field'])
        T_last = T_arr[:, -1]
        axes[1, 1].plot(range(len(T_last)), T_last, '-', alpha=0.5, lw=1,
                       label=field['prompt'][:15] if pi < 3 else None)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('T (last position)')
    axes[1, 1].set_title('(e) Last-Position T Across Prompts')
    axes[1, 1].legend(fontsize=6)

    # (f) Positional gradient: dT/dpos at each layer
    T_arr = np.array(field15['T_field'])
    dT_dpos = np.diff(T_arr, axis=1)  # (layers, seq_len-1)
    mean_grad = dT_dpos.mean(axis=1)
    axes[1, 2].plot(range(len(mean_grad)), mean_grad, '-o', color='purple', lw=2, markersize=3)
    axes[1, 2].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 2].set_xlabel('Layer')
    axes[1, 2].set_ylabel('Mean dT/dpos')
    axes[1, 2].set_title('(f) Positional Gradient')

    fig.suptitle("Phase 240: Token Position Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase240_position')
    plt.close()
    save_results('phase240_position', {
        'experiment': 'Token Position Thermodynamics',
        'results': results,
    })


if __name__ == '__main__':
    main()
