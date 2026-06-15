# -*- coding: utf-8 -*-
"""
Phase 189: Thermodynamic Depth (Bennett's Logical Depth)
==========================================================
Bennett (1988): Logical depth = computational time to generate an object
from its most compressed description.

In transformers, "depth" = how many layers of irreversible work are needed
to transform the input distribution into the output distribution.

KEY QUESTION: Is the semantic complexity of a prompt correlated with
              the thermodynamic depth of its processing?

We define thermodynamic depth as the integral of |dissipated work| dW
along the layer path. Simple prompts should have shallow depth
(most layers idle). Complex prompts should have deep processing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

# Prompts graded by expected complexity
PROMPTS_GRADED = {
    'trivial': [
        "Yes",
        "Hello",
        "The",
    ],
    'simple': [
        "The sky is blue because",
        "Water freezes at zero degrees",
        "Cats are common pets that",
    ],
    'moderate': [
        "The fundamental theorem of calculus connects",
        "Quantum mechanics describes particles at atomic scale",
        "Neural networks learn through gradient descent",
    ],
    'complex': [
        "The renormalization group explains how physical theories change across energy scales by",
        "Non-equilibrium statistical mechanics extends thermodynamics beyond the assumption of detailed balance through",
        "The holographic principle suggests that gravitational dynamics in a volume can be encoded on its boundary via",
    ],
    'deep': [
        "If consciousness is an emergent property of information integration as proposed by integrated information theory then the relationship between phenomenal experience and physical computation can be formalized by",
        "The connection between the cosmological constant problem and the hierarchy problem in particle physics suggests that our understanding of vacuum energy and symmetry breaking requires fundamentally new theoretical frameworks such as",
        "Given that Godel's incompleteness theorems impose fundamental limits on formal axiomatic systems and that Turing's halting problem establishes uncomputability boundaries the question of whether mathematical truth is discovered or invented becomes particularly acute when considering",
    ],
}


def measure_thermodynamic_depth(model, tok, prompt, device):
    """Measure thermodynamic depth and related quantities."""
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
    n_tokens = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states)
    U_vals, T_vals, S_vals = [], [], []

    for li in range(n_layers):
        hs = out.hidden_states[li]
        h = hs[0, -1, :].float()
        U = h.norm().item()

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        U_vals.append(U if not np.isnan(U) else 0)
        T_vals.append(T if not np.isnan(T) else 0)
        S_vals.append(S if not np.isnan(S) else 0)

    # Thermodynamic depth: integral of |dissipated work|
    W_diss = []
    active_layers = 0
    for i in range(n_layers - 1):
        dU = U_vals[i + 1] - U_vals[i]
        dS = S_vals[i + 1] - S_vals[i]
        T_avg = (T_vals[i] + T_vals[i + 1]) / 2 + 1e-10
        Q = T_avg * dS
        W = Q - dU  # Work by the layer
        W_diss.append(abs(W) if not np.isnan(W) else 0)
        if abs(W) > 0.1:  # Count "active" layers
            active_layers += 1

    depth = sum(W_diss)

    # Participation depth: effective number of active layers (like PR but for work)
    W_arr = np.array(W_diss)
    W_prob = W_arr / (W_arr.sum() + 1e-10)
    part_depth = np.exp(-(W_prob * np.log(W_prob + 1e-20)).sum())

    # Confidence
    final_logits = out.logits[0, -1, :].float()
    probs_final = torch.softmax(final_logits, dim=-1)
    conf = probs_final.max().item()

    # Final entropy
    S_final = T_vals[-1]

    # eta
    T_hot = np.mean(T_vals[:3])
    T_cold = np.mean(T_vals[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    return {
        'depth': float(depth),
        'part_depth': float(part_depth),
        'active_layers': int(active_layers),
        'n_layers': n_layers,
        'eta': float(eta),
        'conf': float(conf),
        'S_final': float(S_final),
        'n_tokens': int(n_tokens),
        'W_diss': [float(w) for w in W_diss],
    }


def main():
    print("=" * 70)
    print("Phase 189: Thermodynamic Depth")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    all_results = {}
    for grade, prompts in PROMPTS_GRADED.items():
        print(f"\n--- {grade.upper()} ---")
        all_results[grade] = []
        for prompt in prompts:
            r = measure_thermodynamic_depth(model, tok, prompt, device)
            r['grade'] = grade
            r['prompt_preview'] = prompt[:50].encode('ascii', errors='replace').decode('ascii')
            all_results[grade].append(r)
            print(f"  [{r['n_tokens']:3d} tok] depth={r['depth']:.2f}, "
                  f"part={r['part_depth']:.1f}, active={r['active_layers']}, "
                  f"conf={r['conf']:.4f}")

    # === Analysis ===
    grades = list(PROMPTS_GRADED.keys())
    grade_depths = {g: np.mean([r['depth'] for r in all_results[g]]) for g in grades}
    grade_parts = {g: np.mean([r['part_depth'] for r in all_results[g]]) for g in grades}
    grade_confs = {g: np.mean([r['conf'] for r in all_results[g]]) for g in grades}
    grade_etas = {g: np.mean([r['eta'] for r in all_results[g]]) for g in grades}
    grade_tokens = {g: np.mean([r['n_tokens'] for r in all_results[g]]) for g in grades}

    # Correlation: depth vs token count
    all_depths = [r['depth'] for g in grades for r in all_results[g]]
    all_tokens = [r['n_tokens'] for g in grades for r in all_results[g]]
    all_confs = [r['conf'] for g in grades for r in all_results[g]]

    from scipy.stats import pearsonr
    corr_depth_tokens, p_dt = pearsonr(all_depths, all_tokens)
    corr_depth_conf, p_dc = pearsonr(all_depths, all_confs) if len(all_depths) > 3 else (0, 1)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    grade_colors = {'trivial': '#bdc3c7', 'simple': '#3498db', 'moderate': '#2ecc71',
                    'complex': '#f39c12', 'deep': '#e74c3c'}

    # (a) Depth by complexity grade
    x = np.arange(len(grades))
    bars = axes[0, 0].bar(x, [grade_depths[g] for g in grades],
                          color=[grade_colors[g] for g in grades], edgecolor='black', alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(grades, fontsize=9)
    axes[0, 0].set_ylabel('Thermodynamic Depth')
    axes[0, 0].set_title('(a) Depth vs Semantic Complexity')

    # (b) Participation depth by grade
    axes[0, 1].bar(x, [grade_parts[g] for g in grades],
                   color=[grade_colors[g] for g in grades], edgecolor='black', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(grades, fontsize=9)
    axes[0, 1].set_ylabel('Participation Depth')
    axes[0, 1].set_title('(b) Active Layers vs Complexity')

    # (c) Depth vs token count scatter
    for g in grades:
        d = [r['depth'] for r in all_results[g]]
        t = [r['n_tokens'] for r in all_results[g]]
        axes[0, 2].scatter(t, d, c=grade_colors[g], s=80, edgecolors='black', label=g, zorder=5)
    axes[0, 2].set_xlabel('Prompt Length (tokens)')
    axes[0, 2].set_ylabel('Thermodynamic Depth')
    axes[0, 2].set_title(f'(c) Depth vs Length (r={corr_depth_tokens:.3f})')
    axes[0, 2].legend(fontsize=7)

    # (d) Work profile comparison
    for g in ['trivial', 'moderate', 'deep']:
        if all_results[g]:
            W = all_results[g][0]['W_diss']
            axes[1, 0].plot(np.arange(len(W)) + 0.5, W, 'o-', markersize=3,
                            color=grade_colors[g], linewidth=2, label=g)
    axes[1, 0].axvline(x=21, color='black', linestyle='--', alpha=0.3, label='$L_0$')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$|W_{diss}|$')
    axes[1, 0].set_title('(d) Work Profile by Complexity')
    axes[1, 0].legend(fontsize=7)

    # (e) eta by complexity
    axes[1, 1].bar(x, [grade_etas[g] for g in grades],
                   color=[grade_colors[g] for g in grades], edgecolor='black', alpha=0.8)
    axes[1, 1].axhline(y=0.813, color='black', linestyle='--', alpha=0.3, label='$\\eta=0.813$')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(grades, fontsize=9)
    axes[1, 1].set_ylabel('$\\eta$')
    axes[1, 1].set_title('(e) Efficiency by Complexity')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Thermodynamic Depth\n(Bennett's Logical Depth)\n\n"
        f"Depth by grade:\n"
    )
    for g in grades:
        summary += f"  {g}: {grade_depths[g]:.2f}\n"
    summary += (
        f"\nCorrelations:\n"
        f"  depth vs tokens: r={corr_depth_tokens:.3f}\n"
        f"  depth vs conf: r={corr_depth_conf:.3f}\n\n"
        f"Deeper prompts use\n"
        f"{'MORE' if grade_depths['deep'] > grade_depths['trivial'] else 'SAME'}"
        f" computational work"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 189: Thermodynamic Depth (Bennett's Logical Depth)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase189_depth')
    plt.close()

    print(f"\n{'=' * 70}")
    for g in grades:
        print(f"  {g}: depth={grade_depths[g]:.2f}, part={grade_parts[g]:.1f}, conf={grade_confs[g]:.4f}")
    print(f"  depth vs tokens: r={corr_depth_tokens:.3f}")
    print(f"  depth vs conf:   r={corr_depth_conf:.3f}")
    print(f"{'=' * 70}")

    save_results('phase189_depth', {
        'experiment': "Thermodynamic Depth (Bennett's Logical Depth)",
        'by_grade': {g: {'depth': float(grade_depths[g]), 'part_depth': float(grade_parts[g]),
                          'conf': float(grade_confs[g]), 'eta': float(grade_etas[g])}
                     for g in grades},
        'correlations': {
            'depth_vs_tokens': float(corr_depth_tokens),
            'depth_vs_conf': float(corr_depth_conf),
        },
    })


if __name__ == '__main__':
    main()
