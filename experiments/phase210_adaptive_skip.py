# -*- coding: utf-8 -*-
"""
Phase 210: Adaptive Layer Skip (Physics-Informed Dynamic Pruning)
================================================================
Phase 203 showed fixed skip (layers 10-20) saves 39% FLOPs with only 29%
quality loss. But which layers are TRULY adiabatic (dT/dLayer ~ 0)?

Instead of fixed skip, dynamically identify "adiabatic" layers where
temperature barely changes, and skip only those. This is physics-informed
adaptive layer pruning.

Added by Opus: Phase 203's fixed-skip result begs the question of whether
an adaptive approach can do even better.
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

# dT thresholds: layers with |dT| < threshold are skippable
DT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
# Fixed skip configs from Phase 203 for comparison
FIXED_SKIPS = [
    None,                    # Full
    (10, 15),                # Skip 10-15
    (10, 20),                # Skip 10-20
    (5, 20),                 # Skip 5-20
]


def profile_layer_temperatures(model, tok, device, prompt):
    """Full forward pass to measure T at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    norm_layer = model.model.norm
    lm_head = model.lm_head
    T_list = []
    for hs in out.hidden_states:
        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_list.append(T if not np.isnan(T) else 0)
    return T_list


def identify_adiabatic_layers(T_list, threshold):
    """Find layers where |dT| < threshold (adiabatic = safe to skip)."""
    skip_layers = []
    for i in range(1, len(T_list) - 1):  # Never skip first or last
        dT = abs(T_list[i+1] - T_list[i])
        if dT < threshold:
            # Layer index is i (0-indexed transformer layer)
            # T_list[0] = embedding, T_list[1] = after layer 0, etc.
            layer_idx = i - 1  # Adjust: T_list[i] = after layer (i-1)
            if 0 < layer_idx < 27:  # Never skip first or last transformer layer
                skip_layers.append(layer_idx)
    return skip_layers


def run_with_skip(model, tok, device, prompt, skip_set):
    """Manual forward pass skipping specified layers."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)

        # Compute position embeddings (Qwen2 requirement)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        layers_run = 0
        for li in range(n_layers):
            if li in skip_set:
                continue
            layer = model.model.layers[li]
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
            layers_run += 1

        normed = norm_layer(hidden)
        final_logits = lm_head(normed)

    final_probs = torch.softmax(final_logits[0, -1, :].float(), dim=-1)
    output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
    top1_prob = final_probs.max().item()
    top_token = tok.decode(final_logits[0, -1, :].argmax().item())

    return {
        'output_entropy': output_entropy,
        'top1_prob': top1_prob,
        'top_token': top_token,
        'layers_run': layers_run,
        'layers_skipped': n_layers - layers_run,
        'flops_saved_pct': (n_layers - layers_run) / n_layers * 100,
    }


def main():
    print("=" * 70)
    print("Phase 210: Adaptive Layer Skip")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    # Step 1: Profile temperature for all prompts
    print("\n=== Profiling layer temperatures ===")
    all_T_profiles = []
    for prompt in PROMPTS:
        T = profile_layer_temperatures(model, tok, device, prompt)
        all_T_profiles.append(T)

    # Average dT profile
    n_layers_p1 = len(all_T_profiles[0])
    mean_dT = []
    for i in range(n_layers_p1 - 1):
        dTs = [abs(p[i+1] - p[i]) for p in all_T_profiles]
        mean_dT.append(float(np.mean(dTs)))
    print(f"  Mean |dT| range: {min(mean_dT):.3f} to {max(mean_dT):.3f}")

    # Step 2: Adaptive skip at various thresholds
    print("\n=== Adaptive Layer Skip ===")
    adaptive_results = {}
    for threshold in DT_THRESHOLDS:
        all_ent, all_top1, all_skipped = [], [], []
        for pi, prompt in enumerate(PROMPTS):
            T_profile = all_T_profiles[pi]
            skip_layers = identify_adiabatic_layers(T_profile, threshold)
            skip_set = set(skip_layers)
            r = run_with_skip(model, tok, device, prompt, skip_set)
            all_ent.append(r['output_entropy'])
            all_top1.append(r['top1_prob'])
            all_skipped.append(r['layers_skipped'])

        label = f"threshold_{threshold}"
        adaptive_results[label] = {
            'threshold': threshold,
            'entropy_mean': float(np.mean(all_ent)),
            'top1_mean': float(np.mean(all_top1)),
            'layers_skipped_mean': float(np.mean(all_skipped)),
            'flops_saved_pct': float(np.mean(all_skipped)) / 28 * 100,
        }
        print(f"  threshold={threshold:.1f}: entropy={np.mean(all_ent):.3f}, "
              f"skipped={np.mean(all_skipped):.1f}/28, "
              f"FLOPs saved={np.mean(all_skipped)/28*100:.1f}%")

    # Step 3: Fixed skip comparison
    print("\n=== Fixed Skip (Phase 203 comparison) ===")
    fixed_results = {}
    for skip_range in FIXED_SKIPS:
        if skip_range is None:
            skip_set = set()
            label = "full"
        else:
            skip_set = set(range(skip_range[0], skip_range[1]))
            label = f"skip_{skip_range[0]}_{skip_range[1]}"

        all_ent, all_top1 = [], []
        for prompt in PROMPTS:
            r = run_with_skip(model, tok, device, prompt, skip_set)
            all_ent.append(r['output_entropy'])
            all_top1.append(r['top1_prob'])

        fixed_results[label] = {
            'entropy_mean': float(np.mean(all_ent)),
            'top1_mean': float(np.mean(all_top1)),
            'layers_skipped': len(skip_set),
            'flops_saved_pct': len(skip_set) / 28 * 100,
        }
        print(f"  {label}: entropy={np.mean(all_ent):.3f}, "
              f"FLOPs saved={len(skip_set)/28*100:.1f}%")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) dT profile (mean across prompts)
    axes[0, 0].bar(range(len(mean_dT)), mean_dT, color='#e67e22', alpha=0.7)
    axes[0, 0].axhline(y=0.3, color='red', linestyle='--',
                       label='threshold=0.3')
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('Mean |dT/dLayer|')
    axes[0, 0].set_title('(a) Temperature Gradient Profile')
    axes[0, 0].legend(fontsize=8)

    # (b) Entropy vs FLOPs saved (adaptive)
    flops = [r['flops_saved_pct'] for r in adaptive_results.values()]
    ents = [r['entropy_mean'] for r in adaptive_results.values()]
    axes[0, 1].plot(flops, ents, 'o-', color='#2ecc71', markersize=8, lw=2,
                    label='Adaptive')
    # Add fixed skip points
    for label, r in fixed_results.items():
        marker = 'x' if label != 'full' else '*'
        axes[0, 1].plot(r['flops_saved_pct'], r['entropy_mean'], marker,
                        color='#e74c3c', markersize=12, markeredgewidth=2)
    axes[0, 1].set_xlabel('FLOPs Saved (%)')
    axes[0, 1].set_ylabel('Output Entropy (nats)')
    axes[0, 1].set_title('(b) Quality vs Efficiency Frontier')
    axes[0, 1].legend(fontsize=8)

    # (c) Layers skipped vs threshold
    thresholds = [r['threshold'] for r in adaptive_results.values()]
    skipped = [r['layers_skipped_mean'] for r in adaptive_results.values()]
    axes[0, 2].plot(thresholds, skipped, 's-', color='#3498db', markersize=8, lw=2)
    axes[0, 2].set_xlabel('dT Threshold')
    axes[0, 2].set_ylabel('Layers Skipped (mean)')
    axes[0, 2].set_title('(c) Skip Rate vs Threshold')

    # (d) Top-1 prob vs threshold
    top1s = [r['top1_mean'] for r in adaptive_results.values()]
    axes[1, 0].plot(thresholds, top1s, 'D-', color='#9b59b6', markersize=8, lw=2)
    axes[1, 0].set_xlabel('dT Threshold')
    axes[1, 0].set_ylabel('Top-1 Probability')
    axes[1, 0].set_title('(d) Confidence vs Threshold')

    # (e) Adaptive vs Fixed comparison at ~40% FLOPs saved
    # Find adaptive threshold closest to 40% savings
    target = 40
    closest_idx = min(range(len(flops)),
                      key=lambda i: abs(flops[i] - target))
    closest_adaptive = list(adaptive_results.values())[closest_idx]
    fixed_1020 = fixed_results.get('skip_10_20', {})

    methods = ['Full', 'Fixed\n(10-20)', f'Adaptive\n(t={closest_adaptive["threshold"]})']
    ent_vals = [
        fixed_results.get('full', {}).get('entropy_mean', 0),
        fixed_1020.get('entropy_mean', 0),
        closest_adaptive['entropy_mean'],
    ]
    flops_vals = [0, fixed_1020.get('flops_saved_pct', 0),
                  closest_adaptive['flops_saved_pct']]

    x = np.arange(len(methods))
    w = 0.35
    b1 = axes[1, 1].bar(x - w/2, ent_vals, w, label='Entropy', color='#e74c3c', alpha=0.7)
    ax2 = axes[1, 1].twinx()
    b2 = ax2.bar(x + w/2, flops_vals, w, label='FLOPs saved %', color='#3498db', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, fontsize=8)
    axes[1, 1].set_ylabel('Output Entropy')
    ax2.set_ylabel('FLOPs Saved (%)')
    axes[1, 1].set_title('(e) Adaptive vs Fixed Skip')
    axes[1, 1].legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)

    # (f) Summary
    full_ent = fixed_results.get('full', {}).get('entropy_mean', 0)
    summary_text = (
        f"Adaptive Layer Skip\n\n"
        f"Full (28 layers):\n"
        f"  entropy = {full_ent:.3f}\n\n"
        f"Fixed skip (10-20):\n"
        f"  entropy = {fixed_1020.get('entropy_mean', 0):.3f}\n"
        f"  FLOPs saved = {fixed_1020.get('flops_saved_pct', 0):.1f}%\n\n"
        f"Adaptive (t={closest_adaptive['threshold']}):\n"
        f"  entropy = {closest_adaptive['entropy_mean']:.3f}\n"
        f"  FLOPs saved = {closest_adaptive['flops_saved_pct']:.1f}%\n"
        f"  layers skipped = {closest_adaptive['layers_skipped_mean']:.1f}\n"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 210: Adaptive Layer Skip",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase210_adaptive_skip')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Full: entropy={full_ent:.3f}")
    print(f"Best adaptive: entropy={closest_adaptive['entropy_mean']:.3f}, "
          f"FLOPs saved={closest_adaptive['flops_saved_pct']:.1f}%")
    print(f"{'=' * 70}")

    save_results('phase210_adaptive_skip', {
        'experiment': 'Adaptive Layer Skip',
        'adaptive': adaptive_results,
        'fixed': fixed_results,
        'mean_dT_profile': [float(x) for x in mean_dT],
    })


if __name__ == '__main__':
    main()
