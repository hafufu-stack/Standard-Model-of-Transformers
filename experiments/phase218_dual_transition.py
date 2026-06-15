# -*- coding: utf-8 -*-
"""
Phase 218: Dual Phase Transition
===================================
Season 15 revealed L0=0 (max dT/dLayer), but the crystallization at ~L21
is still real. There are TWO transitions:
  1. L0=0: embedding->transformer ignition (max gradient)
  2. Lc~21: high-T -> low-T condensation (crystallization)

Rigorously characterize both transitions.
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


def full_profile(model, tok, device):
    """Comprehensive profiling: T, U, P1, PR, cos at every layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_T, all_U, all_P1, all_PR, all_cos = [], [], [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, U_l, P1_l, PR_l, cos_l = [], [], [], [], []
        prev_h = None
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            # Hidden PR
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR_l.append(1.0 / ((h_prob ** 2).sum().item() + 1e-10))
            # Cosine with previous
            if prev_h is not None:
                cos_l.append(torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item())
            else:
                cos_l.append(0)
            prev_h = h.clone()
            # Logit-based T and P1
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(probs.max().item())
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(T if not np.isnan(T) else 0)
        all_T.append(T_l); all_U.append(U_l); all_P1.append(P1_l)
        all_PR.append(PR_l); all_cos.append(cos_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_T, mean_U, mean_P1, mean_PR, mean_cos = (
        avg(all_T), avg(all_U), avg(all_P1), avg(all_PR), avg(all_cos))

    # Derivatives
    dT = [mean_T[i+1] - mean_T[i] for i in range(n-1)]
    dU = [mean_U[i+1] - mean_U[i] for i in range(n-1)]
    d2T = [dT[i+1] - dT[i] for i in range(len(dT)-1)]

    # Transition points
    abs_dT = [abs(x) for x in dT]
    L_ignition = int(np.argmax(abs_dT))  # Max |dT/dl|

    # Crystallization: where T drops below threshold (e.g., median of non-zero T)
    T_threshold = np.median([t for t in mean_T[1:] if t > 0])
    L_crystal = 0
    for i in range(len(mean_T)-1, 0, -1):
        if mean_T[i] > T_threshold:
            L_crystal = i
            break

    # Second transition: where d2T has maximum negative value (concavity change)
    neg_d2T = [x for x in d2T]
    L_inflection = int(np.argmin(neg_d2T)) if neg_d2T else 0

    return {
        'mean_T': mean_T, 'mean_U': mean_U, 'mean_P1': mean_P1,
        'mean_PR': mean_PR, 'mean_cos': mean_cos,
        'dT': [float(x) for x in dT],
        'dU': [float(x) for x in dU],
        'd2T': [float(x) for x in d2T],
        'L_ignition': L_ignition,
        'L_crystal': L_crystal,
        'L_inflection': L_inflection,
        'n_layers': len(model.model.layers),
    }


def main():
    print("=" * 70)
    print("Phase 218: Dual Phase Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        p = full_profile(model, tok, device)
        results[size] = p
        print(f"  L_ignition={p['L_ignition']}, L_crystal={p['L_crystal']}, "
              f"L_inflection={p['L_inflection']}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Temperature profile with dual transitions
    for size, p in results.items():
        x = range(len(p['mean_T']))
        axes[0, 0].plot(x, p['mean_T'], '-', color=colors[size], lw=2, label=size)
        axes[0, 0].axvline(x=p['L_ignition'], color=colors[size], ls=':', alpha=0.5)
        axes[0, 0].axvline(x=p['L_crystal'], color=colors[size], ls='--', alpha=0.5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) T profile: ignition(:) & crystal(--)')
    axes[0, 0].legend(fontsize=8)

    # (b) dT/dl with transition markers
    for size, p in results.items():
        axes[0, 1].plot(range(len(p['dT'])), p['dT'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('dT/dLayer')
    axes[0, 1].set_title('(b) Temperature Gradient')
    axes[0, 1].legend(fontsize=8)

    # (c) P1 profile
    for size, p in results.items():
        axes[0, 2].plot(range(len(p['mean_P1'])), p['mean_P1'], '-', color=colors[size], lw=2, label=size)
        axes[0, 2].axvline(x=p['L_crystal'], color=colors[size], ls='--', alpha=0.5)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Top-1 Probability')
    axes[0, 2].set_title('(c) Order Parameter P1')
    axes[0, 2].legend(fontsize=8)

    # (d) d2T/dl2 (curvature)
    for size, p in results.items():
        axes[1, 0].plot(range(len(p['d2T'])), p['d2T'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('d2T/dLayer2')
    axes[1, 0].set_title('(d) Temperature Curvature')
    axes[1, 0].legend(fontsize=8)

    # (e) Cosine alignment
    for size, p in results.items():
        axes[1, 1].plot(range(len(p['mean_cos'])), p['mean_cos'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('cos(h_l, h_{l+1})')
    axes[1, 1].set_title('(e) Layer Alignment')
    axes[1, 1].legend(fontsize=8)

    # (f) Phase diagram: T vs U
    for size, p in results.items():
        axes[1, 2].scatter(p['mean_U'], p['mean_T'], c=range(len(p['mean_T'])),
                          cmap='coolwarm', s=30, alpha=0.7, label=size)
        # Connect with line
        axes[1, 2].plot(p['mean_U'], p['mean_T'], '-', color=colors[size], alpha=0.3)
    axes[1, 2].set_xlabel('Internal Energy U')
    axes[1, 2].set_ylabel('Temperature T')
    axes[1, 2].set_title('(f) Phase Diagram (T vs U)')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Phase 218: Dual Phase Transition", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase218_dual_transition')
    plt.close()
    save_results('phase218_dual_transition', {'experiment': 'Dual Phase Transition', 'results': results})


if __name__ == '__main__':
    main()
