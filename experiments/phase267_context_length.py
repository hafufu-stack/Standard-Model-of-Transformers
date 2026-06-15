# -*- coding: utf-8 -*-
"""
Phase 267: Context Length Thermodynamics
=========================================
Question: Does input length change the thermodynamic state?
  - More tokens = more energy?
  - More tokens = lower temperature (more context = less uncertainty)?
  - Does P1*T constant depend on context length?

Tests with context lengths from 5 to 100 tokens.
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

# Prompts of varying lengths (we'll truncate/extend)
BASE_TEXT = (
    "The fundamental theorem of calculus states that differentiation and "
    "integration are inverse operations. This remarkable result connects "
    "the concept of a derivative with the concept of an integral. "
    "It was independently discovered by Newton and Leibniz in the "
    "seventeenth century. The theorem has two main parts that together "
    "form the backbone of mathematical analysis. The first part states "
    "that an antiderivative can be computed using a definite integral. "
    "The second part allows definite integrals to be evaluated using "
    "antiderivatives. This has profound implications for physics and "
    "engineering where continuous change must be measured and controlled."
)

# Target token counts
TARGET_LENGTHS = [5, 10, 20, 40, 60, 80, 100]


def measure_at_length(model, tok, device, n_tokens):
    """Measure thermodynamic state for a given context length."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Tokenize full text and truncate
    full_ids = tok.encode(BASE_TEXT)
    if n_tokens > len(full_ids):
        n_tokens = len(full_ids)

    input_ids = torch.tensor([full_ids[:n_tokens]]).to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    n_layers = len(out.hidden_states)
    T_l, P1_l, U_l, PRT_l = [], [], [], []

    for hs in out.hidden_states:
        h = hs[0, -1, :].float()
        U = float(h.norm().item())

        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        P1 = float(probs.max().item())
        T_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(T_sm): T_sm = 0

        T_l.append(T_sm)
        P1_l.append(P1)
        U_l.append(U)
        PRT_l.append(P1 * T_sm)

    cv = float(np.std(PRT_l[1:]) / (np.mean(PRT_l[1:]) + 1e-10))
    rho_T, _ = stats.spearmanr(range(n_layers), T_l)

    return {
        'n_tokens': n_tokens,
        'T_final': round(T_l[-1], 4),
        'P1_final': round(P1_l[-1], 4),
        'U_final': round(U_l[-1], 4),
        'PRT_final': round(PRT_l[-1], 4),
        'PRT_mean': round(float(np.mean(PRT_l[1:])), 4),
        'PRT_cv': round(cv, 4),
        'T_profile': T_l,
        'P1_profile': P1_l,
        'PRT_profile': PRT_l,
        'arrow_rho': round(float(rho_T), 4),
    }


def main():
    print("=" * 70)
    print("Phase 267: Context Length Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)

        length_data = []
        for n in TARGET_LENGTHS:
            r = measure_at_length(model, tok, device, n)
            length_data.append(r)
            print(f"  n={r['n_tokens']:3d}: T_final={r['T_final']:.3f}, "
                  f"P1={r['P1_final']:.3f}, P1*T={r['PRT_mean']:.3f}, "
                  f"CV={r['PRT_cv']:.3f}")

        all_results[size] = length_data
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Analysis ===
    for size, data in all_results.items():
        ns = [d['n_tokens'] for d in data]
        Ts = [d['T_final'] for d in data]
        PRTs = [d['PRT_mean'] for d in data]
        r_T, p_T = stats.pearsonr(ns, Ts)
        r_PRT, p_PRT = stats.pearsonr(ns, PRTs)
        print(f"\n  {size}: r(n, T_final) = {r_T:.4f} (p={p_T:.4f})")
        print(f"  {size}: r(n, PRT_mean) = {r_PRT:.4f} (p={p_PRT:.4f})")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) T_final vs context length
    for size, data in all_results.items():
        c = colors[size]
        ns = [d['n_tokens'] for d in data]
        Ts = [d['T_final'] for d in data]
        axes[0, 0].plot(ns, Ts, '-o', color=c, lw=2, markersize=6, label=size)
    axes[0, 0].set_xlabel('Context Length (tokens)')
    axes[0, 0].set_ylabel('T_final')
    axes[0, 0].set_title('(a) Final Temperature vs Context', fontweight='bold')
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # (b) P1*T mean vs context length
    for size, data in all_results.items():
        c = colors[size]
        ns = [d['n_tokens'] for d in data]
        PRTs = [d['PRT_mean'] for d in data]
        axes[0, 1].plot(ns, PRTs, '-o', color=c, lw=2, markersize=6, label=size)
    axes[0, 1].set_xlabel('Context Length (tokens)')
    axes[0, 1].set_ylabel('Mean P1*T')
    axes[0, 1].set_title('(b) P1*T Constant vs Context', fontweight='bold')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # (c) P1*T CV vs context length
    for size, data in all_results.items():
        c = colors[size]
        ns = [d['n_tokens'] for d in data]
        CVs = [d['PRT_cv'] for d in data]
        axes[0, 2].plot(ns, CVs, '-o', color=c, lw=2, markersize=6, label=size)
    axes[0, 2].set_xlabel('Context Length (tokens)')
    axes[0, 2].set_ylabel('CV(P1*T)')
    axes[0, 2].set_title('(c) Conservation Quality vs Context', fontweight='bold')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    # (d) P1 and T profiles for different lengths (0.5B)
    if '0.5B' in all_results:
        data = all_results['0.5B']
        cmap = plt.cm.viridis
        for i, d in enumerate(data):
            c = cmap(i / max(len(data) - 1, 1))
            x = np.linspace(0, 1, len(d['T_profile']))
            axes[1, 0].plot(x, d['T_profile'], '-', color=c, lw=1.5,
                           label=f'n={d["n_tokens"]}')
    axes[1, 0].set_xlabel('Normalized Depth')
    axes[1, 0].set_ylabel('T_sm')
    axes[1, 0].set_title('(d) T Profiles by Length (0.5B)', fontweight='bold')
    axes[1, 0].legend(fontsize=6, ncol=2); axes[1, 0].grid(alpha=0.3)

    # (e) P1*T profiles for different lengths (0.5B)
    if '0.5B' in all_results:
        data = all_results['0.5B']
        for i, d in enumerate(data):
            c = cmap(i / max(len(data) - 1, 1))
            x = np.linspace(0, 1, len(d['PRT_profile']))
            axes[1, 1].plot(x, d['PRT_profile'], '-', color=c, lw=1.5,
                           label=f'n={d["n_tokens"]}')
    axes[1, 1].set_xlabel('Normalized Depth')
    axes[1, 1].set_ylabel('P1 x T')
    axes[1, 1].set_title('(e) P1*T Profiles by Length (0.5B)', fontweight='bold')
    axes[1, 1].legend(fontsize=6, ncol=2); axes[1, 1].grid(alpha=0.3)

    # (f) Arrow of time vs context
    for size, data in all_results.items():
        c = colors[size]
        ns = [d['n_tokens'] for d in data]
        arrows = [d['arrow_rho'] for d in data]
        axes[1, 2].plot(ns, arrows, '-o', color=c, lw=2, markersize=6, label=size)
    axes[1, 2].axhline(0, color='gray', ls='--', lw=0.5)
    axes[1, 2].set_xlabel('Context Length (tokens)')
    axes[1, 2].set_ylabel('Arrow of Time (rho)')
    axes[1, 2].set_title('(f) Arrow of Time vs Context', fontweight='bold')
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(alpha=0.3)

    fig.suptitle("Phase 267: Context Length Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase267_context_length')
    plt.close()

    save_results('phase267_context_length', {
        'experiment': 'Context Length Thermodynamics',
        'results': {size: [{k: v for k, v in d.items()
                           if k not in ('T_profile', 'P1_profile', 'PRT_profile')}
                          for d in data]
                   for size, data in all_results.items()},
    })


if __name__ == '__main__':
    main()
