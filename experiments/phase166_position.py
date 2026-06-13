# -*- coding: utf-8 -*-
"""
Phase 166: Token Position Thermodynamics
How does the thermodynamic profile change depending on
which token position we measure (first, middle, last)?
Are there "hot spots" and "cold spots" within a sequence?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects derivatives and integrals together",
    "Quantum mechanics describes how particles behave at the atomic scale precisely",
    "Neural networks learn complex patterns through gradient descent optimization methods",
    "Black holes form when massive stars undergo gravitational collapse completely",
    "The periodic table organizes all known chemical elements by atomic number",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules",
]


def main():
    print("=" * 70)
    print("Phase 166: Token Position Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # For each prompt, measure S at multiple token positions
    position_profiles = {'first_quarter': [], 'mid': [], 'three_quarter': [], 'last': []}
    position_S_maps = []  # For heatmap

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Positions to measure
        positions = {
            'first_quarter': max(1, seq_len // 4),
            'mid': seq_len // 2,
            'three_quarter': 3 * seq_len // 4,
            'last': seq_len - 1,
        }

        S_map = np.zeros((n_layers, len(positions)))

        for pos_i, (pos_name, pos_idx) in enumerate(positions.items()):
            S_profile = []
            for li in range(n_layers):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed = model.model.norm(hs[:, pos_idx:pos_idx+1, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                S = S if not np.isnan(S) else 0
                S_profile.append(S)
                S_map[li, pos_i] = S

            position_profiles[pos_name].append(S_profile)

        position_S_maps.append(S_map)

    # Average across prompts
    avg_profiles = {}
    for pos_name, profiles in position_profiles.items():
        avg_profiles[pos_name] = np.mean(profiles, axis=0)

    # Full heatmap: all positions in one prompt
    # Use the first prompt for detailed position analysis
    prompt = "The fundamental theorem of calculus connects derivatives and integrals together in a beautiful way"
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    full_heatmap = np.zeros((n_layers, seq_len))
    for pos in range(seq_len):
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, pos:pos+1, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            full_heatmap[li, pos] = S if not np.isnan(S) else 0

    tokens = [tok.decode([t]) for t in inp['input_ids'][0]]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = np.arange(n_layers)
    colors = {'first_quarter': '#e74c3c', 'mid': '#f39c12',
              'three_quarter': '#27ae60', 'last': '#2980b9'}

    # (a) S profiles by position
    for pos_name, profile in avg_profiles.items():
        axes[0,0].plot(layers, profile, 'o-', color=colors[pos_name],
                      markersize=3, linewidth=2, label=pos_name)
    axes[0,0].axvline(x=21.7, color='gray', linewidth=1, linestyle='--')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) S by Token Position')
    axes[0,0].legend(fontsize=8)

    # (b) Full heatmap
    im = axes[0,1].imshow(full_heatmap, aspect='auto', cmap='hot', origin='lower')
    axes[0,1].axhline(y=21.7, color='cyan', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Token Position')
    axes[0,1].set_ylabel('Layer')
    axes[0,1].set_title('(b) S Heatmap (Layer x Position)')
    plt.colorbar(im, ax=axes[0,1], label='$S$')

    # (c) S at last layer vs position
    axes[0,2].plot(range(seq_len), full_heatmap[-1, :], 'o-', color='#8e44ad',
                  markersize=4, linewidth=2)
    axes[0,2].set_xlabel('Token Position')
    axes[0,2].set_ylabel('$S_{final}$')
    axes[0,2].set_title('(c) Final Layer Entropy')
    # Label tokens
    for i in range(min(seq_len, 15)):
        safe_tok = tokens[i].encode('ascii', errors='replace').decode('ascii')
        axes[0,2].annotate(safe_tok, (i, full_heatmap[-1, i]),
                          fontsize=5, rotation=45, ha='left')

    # (d) Average S across positions at each layer
    avg_over_pos = np.mean(full_heatmap, axis=1)
    std_over_pos = np.std(full_heatmap, axis=1)
    axes[1,0].plot(layers, avg_over_pos, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[1,0].fill_between(layers, avg_over_pos - std_over_pos,
                           avg_over_pos + std_over_pos, alpha=0.2, color='#2980b9')
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Mean $S$ (across positions)')
    axes[1,0].set_title('(d) Position-Averaged Entropy')

    # (e) Position variance at each layer
    axes[1,1].plot(layers, std_over_pos, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Std $S$ (across positions)')
    axes[1,1].set_title('(e) Position Diversity')

    # (f) Summary
    final_first = avg_profiles['first_quarter'][-1]
    final_last = avg_profiles['last'][-1]
    summary = (
        f"Token Position Thermodynamics\n\n"
        f"Final S by position:\n"
        + "\n".join(f"  {pos}: S={avg_profiles[pos][-1]:.2f}"
                    for pos in avg_profiles)
        + f"\n\nFirst/Last ratio: {final_first/final_last:.2f}x\n\n"
        f"Position {'MATTERS' if abs(final_first - final_last) > 0.5 else 'irrelevant'}\n"
        f"for final entropy"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 166: Token Position Thermodynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase166_position')
    plt.close()

    print(f"\n{'='*70}")
    for pos in avg_profiles:
        print(f"  {pos}: S_final={avg_profiles[pos][-1]:.2f}")
    print(f"{'='*70}")

    save_results('phase166_position', {
        'experiment': 'Token Position Thermodynamics',
        'summary': {pos: float(avg_profiles[pos][-1]) for pos in avg_profiles},
    })


if __name__ == '__main__':
    main()
