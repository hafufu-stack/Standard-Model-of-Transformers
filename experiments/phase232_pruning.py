# -*- coding: utf-8 -*-
"""
Phase 232: Thermodynamic Pruning
==================================
Use thermodynamic metrics to identify redundant layers.
A layer is "thermodynamically redundant" if:
  1. dT/dl ~ 0 (no temperature change)
  2. cos(h_l, h_{l+1}) ~ 1 (no information change)
  3. dP1/dl ~ 0 (no ordering change)

Skip these layers and measure quality loss.
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
]

EVAL_PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The square root of sixteen is",
    "Gravity pulls objects toward the",
]


def measure_redundancy(model, tok, device, model_name):
    """Measure layer redundancy using thermodynamic metrics."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_dT, all_cos, all_dP1 = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_l, P1_l, cos_l = [], [], []
        prev_h = None
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)

            if prev_h is not None:
                cos_l.append(float(torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item()))
            else:
                cos_l.append(0)
            prev_h = h.clone()

        dT = [abs(T_l[i+1] - T_l[i]) for i in range(len(T_l)-1)]
        dP1 = [abs(P1_l[i+1] - P1_l[i]) for i in range(len(P1_l)-1)]
        all_dT.append(dT)
        all_cos.append(cos_l[1:])  # Skip first (no prev)
        all_dP1.append(dP1)

    n_trans = min(len(d) for d in all_dT)
    mean_dT = [float(np.mean([all_dT[p][l] for p in range(len(PROMPTS))])) for l in range(n_trans)]
    mean_cos = [float(np.mean([all_cos[p][l] for p in range(len(PROMPTS))])) for l in range(n_trans)]
    mean_dP1 = [float(np.mean([all_dP1[p][l] for p in range(len(PROMPTS))])) for l in range(n_trans)]

    # Redundancy score: high cos + low dT + low dP1
    redundancy = []
    for l in range(n_trans):
        score = mean_cos[l] * (1 - min(mean_dT[l], 1)) * (1 - min(mean_dP1[l], 1))
        redundancy.append(float(score))

    # Rank layers by redundancy
    layer_ranking = sorted(range(n_trans), key=lambda i: redundancy[i], reverse=True)

    # Test: skip top-k most redundant layers
    # Baseline: full model perplexity-like metric
    def eval_quality(model, tok, device, skip_layers=None):
        """Measure output quality with optional layer skipping."""
        total_entropy = 0
        total_p1 = 0
        n = 0
        for prompt in EVAL_PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            total_entropy += S if not np.isnan(S) else 0
            total_p1 += float(probs.max().item())
            n += 1
        return {'entropy': total_entropy / n, 'p1': total_p1 / n}

    baseline = eval_quality(model, tok, device)

    # Test with layer skipping via hooks
    skip_results = {}
    for n_skip in [1, 2, 3, 4, 5]:
        skip_set = set(layer_ranking[:n_skip])  # Most redundant layers
        # Skip by zeroing residual
        handles = []
        for sl in skip_set:
            if sl < n_layers:
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        # Return input unchanged (skip this layer)
                        if isinstance(output, tuple):
                            return (input[0],) + output[1:]
                        return input[0]
                    return hook
                h = model.model.layers[sl].register_forward_hook(make_hook(sl))
                handles.append(h)

        result = eval_quality(model, tok, device)
        for h in handles:
            h.remove()

        result['n_skip'] = n_skip
        result['skip_layers'] = sorted(skip_set)
        result['entropy_change'] = result['entropy'] - baseline['entropy']
        result['p1_change'] = result['p1'] - baseline['p1']
        skip_results[n_skip] = result

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_dT': mean_dT,
        'mean_cos': mean_cos,
        'mean_dP1': mean_dP1,
        'redundancy': redundancy,
        'layer_ranking': layer_ranking,
        'baseline': baseline,
        'skip_results': skip_results,
    }


def main():
    print("=" * 70)
    print("Phase 232: Thermodynamic Pruning")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_redundancy(model, tok, device, size)
        results[size] = r
        print(f"  Baseline: entropy={r['baseline']['entropy']:.3f}, P1={r['baseline']['p1']:.4f}")
        print(f"  Most redundant layers: {r['layer_ranking'][:5]}")
        for n_skip, sr in r['skip_results'].items():
            print(f"  Skip {n_skip}: dEntropy={sr['entropy_change']:+.3f}, dP1={sr['p1_change']:+.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Redundancy score
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['redundancy'])), r['redundancy'],
                       '-o', color=colors[size], lw=1.5, markersize=4, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Redundancy Score')
    axes[0, 0].set_title('(a) Layer Redundancy')
    axes[0, 0].legend(fontsize=8)

    # (b) cos similarity
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['mean_cos'])), r['mean_cos'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('cos(h_l, h_{l+1})')
    axes[0, 1].set_title('(b) Layer Alignment')
    axes[0, 1].legend(fontsize=8)

    # (c) |dT/dl|
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['mean_dT'])), r['mean_dT'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('|dT/dl|')
    axes[0, 2].set_title('(c) Temperature Change')
    axes[0, 2].legend(fontsize=8)

    # (d) Skip test: entropy change
    for size, r in results.items():
        n_skips = sorted(r['skip_results'].keys())
        d_entropy = [r['skip_results'][k]['entropy_change'] for k in n_skips]
        axes[1, 0].plot(n_skips, d_entropy, '-o', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 0].set_xlabel('Layers Skipped'); axes[1, 0].set_ylabel('Entropy Change')
    axes[1, 0].set_title('(d) Quality vs Pruning')
    axes[1, 0].legend(fontsize=8)

    # (e) Skip test: P1 change
    for size, r in results.items():
        n_skips = sorted(r['skip_results'].keys())
        d_p1 = [r['skip_results'][k]['p1_change'] for k in n_skips]
        axes[1, 1].plot(n_skips, d_p1, '-o', color=colors[size], lw=2, label=size)
    axes[1, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 1].set_xlabel('Layers Skipped'); axes[1, 1].set_ylabel('P1 Change')
    axes[1, 1].set_title('(e) Confidence vs Pruning')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Thermodynamic Pruning\n\n"
    for size, r in results.items():
        summary += f"{size} ({r['n_layers']}L):\n"
        summary += f"  Redundant: L{r['layer_ranking'][:3]}\n"
        sr3 = r['skip_results'].get(3, {})
        summary += f"  Skip 3: dS={sr3.get('entropy_change',0):+.2f}\n"
        summary += f"          dP1={sr3.get('p1_change',0):+.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 232: Thermodynamic Pruning", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase232_pruning')
    plt.close()
    save_results('phase232_pruning', {'experiment': 'Thermodynamic Pruning', 'results': results})


if __name__ == '__main__':
    main()
