# -*- coding: utf-8 -*-
"""
Phase 215: Order Parameter Identification
============================================
A phase transition requires an order parameter Psi that changes
discontinuously (or with a kink) at L0.

Test 5 candidates:
1. Top-1 probability P1
2. Hidden state alignment cos(h_l, h_{l+1})
3. Participation ratio PR
4. Logit entropy T (temperature)
5. Hidden entropy S
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
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
]

MODEL_SIZES = ['0.5B', '1.5B']


def measure_order_parameters(model, tok, device, prompts):
    """Measure all order parameter candidates at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_hs = None

    all_P1, all_cos, all_PR, all_T, all_S = [], [], [], [], []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        if n_hs is None:
            n_hs = len(out.hidden_states)

        P1_list, cos_list, PR_list, T_list, S_list = [], [], [], [], []
        prev_h = None

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()

            # 1. Top-1 probability
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_list.append(probs.max().item())

            # 2. Cosine alignment with previous layer
            if prev_h is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item()
            else:
                cos_sim = 0
            cos_list.append(cos_sim)
            prev_h = h.clone()

            # 3. Participation ratio
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / ((h_prob ** 2).sum().item() + 1e-10)
            PR_list.append(PR)

            # 4. Logit entropy (Temperature)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)

            # 5. Hidden entropy
            S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
            S_list.append(S if not np.isnan(S) else 0)

        all_P1.append(P1_list)
        all_cos.append(cos_list)
        all_PR.append(PR_list)
        all_T.append(T_list)
        all_S.append(S_list)

    # Average across prompts
    def avg_profiles(all_data):
        n = min(len(d) for d in all_data)
        return [float(np.mean([d[i] for d in all_data])) for i in range(n)]

    return {
        'P1': avg_profiles(all_P1),
        'cos': avg_profiles(all_cos),
        'PR': avg_profiles(all_PR),
        'T': avg_profiles(all_T),
        'S': avg_profiles(all_S),
        'n_hs': n_hs,
    }


def score_order_parameter(profile, name):
    """Score how well a profile acts as an order parameter.
    Good OP: large discontinuity (max |dPsi/dLayer|) relative to range."""
    vals = np.array(profile)
    if len(vals) < 3:
        return {'name': name, 'score': 0, 'L0': 0, 'jump': 0, 'range': 0}

    dvals = np.abs(np.diff(vals))
    L0 = int(np.argmax(dvals))
    jump = float(dvals[L0])
    val_range = float(vals.max() - vals.min())
    score = jump / (val_range + 1e-10)

    return {
        'name': name,
        'score': float(score),
        'L0': L0,
        'jump': jump,
        'range': val_range,
        'max_val': float(vals.max()),
        'min_val': float(vals.min()),
    }


def main():
    print("=" * 70)
    print("Phase 215: Order Parameter Identification")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_by_model = {}
    for size in MODEL_SIZES:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        op = measure_order_parameters(model, tok, device, PROMPTS)

        scores = {}
        for name in ['P1', 'cos', 'PR', 'T', 'S']:
            s = score_order_parameter(op[name], name)
            scores[name] = s
            print(f"  {name}: score={s['score']:.4f}, L0={s['L0']}, "
                  f"jump={s['jump']:.4f}")

        # Best order parameter
        best = max(scores.values(), key=lambda x: x['score'])
        print(f"  Best OP: {best['name']} (score={best['score']:.4f})")

        results_by_model[size] = {
            'profiles': {k: v for k, v in op.items() if k != 'n_hs'},
            'n_hs': op['n_hs'],
            'scores': scores,
            'best_op': best['name'],
        }

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    op_names = ['P1', 'cos', 'PR', 'T', 'S']
    op_colors = {'P1': '#e74c3c', 'cos': '#3498db', 'PR': '#2ecc71',
                 'T': '#f39c12', 'S': '#9b59b6'}
    model_styles = {'0.5B': '--', '1.5B': '-'}

    # (a-e) Each OP profile, both models normalized
    for oi, op_name in enumerate(op_names):
        row, col = oi // 3, oi % 3
        for size in MODEL_SIZES:
            profile = results_by_model[size]['profiles'][op_name]
            x_norm = np.linspace(0, 1, len(profile))
            # Normalize to [0, 1]
            p_arr = np.array(profile)
            p_norm = (p_arr - p_arr.min()) / (p_arr.max() - p_arr.min() + 1e-10)
            axes[row, col].plot(x_norm, p_norm, model_styles[size],
                                color=op_colors[op_name], lw=2,
                                label=f'{size}', alpha=0.8)
            # Mark L0
            L0 = results_by_model[size]['scores'][op_name]['L0']
            L0_norm = L0 / len(profile)
            axes[row, col].axvline(x=L0_norm, color=op_colors[op_name],
                                   linestyle=':', alpha=0.3)
        axes[row, col].set_xlabel('Normalized Layer')
        axes[row, col].set_ylabel(f'{op_name} (normalized)')
        score_05 = results_by_model['0.5B']['scores'][op_name]['score']
        score_15 = results_by_model['1.5B']['scores'][op_name]['score']
        axes[row, col].set_title(
            f'({chr(97+oi)}) {op_name} (scores: {score_05:.2f}/{score_15:.2f})')
        axes[row, col].legend(fontsize=7)

    # (f) Summary: bar chart of scores
    x = np.arange(len(op_names))
    w = 0.35
    scores_05 = [results_by_model['0.5B']['scores'][n]['score'] for n in op_names]
    scores_15 = [results_by_model['1.5B']['scores'][n]['score'] for n in op_names]
    axes[1, 2].bar(x - w/2, scores_05, w, label='0.5B', color='#3498db', alpha=0.7)
    axes[1, 2].bar(x + w/2, scores_15, w, label='1.5B', color='#e74c3c', alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(op_names)
    axes[1, 2].set_ylabel('Discontinuity Score')
    axes[1, 2].set_title('(f) Order Parameter Ranking')
    axes[1, 2].legend(fontsize=8)

    # Add best OP annotation
    best_05 = results_by_model['0.5B']['best_op']
    best_15 = results_by_model['1.5B']['best_op']
    universal = "SAME" if best_05 == best_15 else "DIFFERENT"
    axes[1, 2].text(0.5, 0.95, f'Best: 0.5B={best_05}, 1.5B={best_15} ({universal})',
                    ha='center', va='top', transform=axes[1, 2].transAxes,
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle("Phase 215: Order Parameter Identification",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase215_order_parameter')
    plt.close()

    save_results('phase215_order_parameter', {
        'experiment': 'Order Parameter Identification',
        'results': results_by_model,
    })


if __name__ == '__main__':
    main()
