# -*- coding: utf-8 -*-
"""
Phase 185: Efficiency Landscape
Map eta and L0 across prompt complexity (length x type) space.
Create 2D heatmap of the efficiency landscape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

# Prompts organized by complexity type
PROMPT_TYPES = {
    'factual': [
        "Water",
        "The capital of France is Paris and",
        "The periodic table organizes chemical elements by their atomic number and electron configuration",
        "In chemistry, the periodic table of elements arranges all known chemical substances by increasing atomic number, grouping elements with similar electron configurations and chemical properties into columns called groups, while rows represent increasing energy levels",
    ],
    'reasoning': [
        "Because",
        "If we assume that gravity affects all",
        "The logical consequence of combining thermodynamics with information theory suggests that entropy",
        "Given that neural networks approximate functions through iterative gradient descent optimization, and considering the universal approximation theorem which states that sufficiently wide networks can represent any continuous function, we can deduce that",
    ],
    'creative': [
        "Once",
        "In the dream, the ancient forest whispered",
        "The last astronaut stood at the edge of the observable universe, contemplating the nature of existence",
        "In a world where consciousness could be transferred between biological and digital substrates, the philosopher queen of the seventh dimension pondered whether the copy of her mind running on quantum processors in the asteroid belt was truly",
    ],
    'technical': [
        "def",
        "The Fourier transform converts signals from time",
        "In quantum field theory, the path integral formulation sums over all possible field configurations weighted",
        "The renormalization group flow equation describes how coupling constants evolve as a function of the energy scale, with fixed points corresponding to scale-invariant theories that exhibit conformal symmetry in the infrared or ultraviolet limits",
    ],
}


def measure_eta_and_L0(model, tok, prompt, device):
    """Measure eta and approximate L0 for a prompt."""
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
    n_tokens = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states)
    T_vals = []
    U_vals = []

    for li in range(n_layers):
        hs = out.hidden_states[li]
        h = hs[0, -1, :].float()
        U = h.norm().item()
        U_vals.append(U if not np.isnan(U) else 0)

        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_vals.append(T if not np.isnan(T) else 0)

    # eta
    T_hot = np.mean(T_vals[:3])
    T_cold = np.mean(T_vals[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    # L0: layer of maximum dU/dl
    dU = np.diff(U_vals)
    L0 = np.argmax(np.abs(dU)) + 1
    L0_ratio = L0 / n_layers

    # Confidence
    final_logits = out.logits[0, -1, :].float()
    probs = torch.softmax(final_logits, dim=-1)
    conf = probs.max().item()

    return {
        'eta': float(eta), 'L0': int(L0), 'L0_ratio': float(L0_ratio),
        'T_hot': float(T_hot), 'T_cold': float(T_cold),
        'confidence': float(conf), 'n_tokens': int(n_tokens),
    }


def main():
    print("=" * 70)
    print("Phase 185: Efficiency Landscape")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Measure all prompts
    results_by_type = {}
    all_etas = []
    all_L0s = []
    all_tokens = []
    all_types = []

    for ptype, prompts in PROMPT_TYPES.items():
        print(f"\n--- {ptype} ---")
        results_by_type[ptype] = []
        for prompt in prompts:
            r = measure_eta_and_L0(model, tok, prompt, device)
            r['type'] = ptype
            r['prompt_preview'] = prompt[:40]
            results_by_type[ptype].append(r)
            all_etas.append(r['eta'])
            all_L0s.append(r['L0_ratio'])
            all_tokens.append(r['n_tokens'])
            all_types.append(ptype)
            print(f"  [{r['n_tokens']:3d} tok] eta={r['eta']:.3f}, L0/L={r['L0_ratio']:.3f}, conf={r['confidence']:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    type_colors = {'factual': '#3498db', 'reasoning': '#e74c3c', 'creative': '#2ecc71', 'technical': '#f39c12'}

    # (a) eta vs token count, colored by type
    for ptype, results in results_by_type.items():
        toks = [r['n_tokens'] for r in results]
        etas = [r['eta'] for r in results]
        axes[0, 0].scatter(toks, etas, c=type_colors[ptype], s=80, edgecolors='black',
                           label=ptype, zorder=5)
        axes[0, 0].plot(toks, etas, '-', color=type_colors[ptype], alpha=0.5, linewidth=1)
    axes[0, 0].axhline(y=0.813, color='black', linestyle='--', alpha=0.3, label='$\\eta=0.813$')
    axes[0, 0].set_xlabel('Prompt Length (tokens)')
    axes[0, 0].set_ylabel('$\\eta$')
    axes[0, 0].set_title('(a) Efficiency vs Prompt Length')
    axes[0, 0].legend(fontsize=7)

    # (b) L0/L vs token count
    for ptype, results in results_by_type.items():
        toks = [r['n_tokens'] for r in results]
        l0s = [r['L0_ratio'] for r in results]
        axes[0, 1].scatter(toks, l0s, c=type_colors[ptype], s=80, edgecolors='black',
                           label=ptype, zorder=5)
        axes[0, 1].plot(toks, l0s, '-', color=type_colors[ptype], alpha=0.5, linewidth=1)
    axes[0, 1].axhline(y=0.75, color='black', linestyle='--', alpha=0.3, label='$L_0/L=0.75$')
    axes[0, 1].set_xlabel('Prompt Length (tokens)')
    axes[0, 1].set_ylabel('$L_0 / L$')
    axes[0, 1].set_title('(b) Critical Layer vs Prompt Length')
    axes[0, 1].legend(fontsize=7)

    # (c) eta by type (box plot)
    type_names = list(results_by_type.keys())
    eta_by_type = [[r['eta'] for r in results_by_type[t]] for t in type_names]
    bp = axes[0, 2].boxplot(eta_by_type, labels=type_names, patch_artist=True)
    for patch, t in zip(bp['boxes'], type_names):
        patch.set_facecolor(type_colors[t])
        patch.set_alpha(0.6)
    axes[0, 2].axhline(y=0.813, color='black', linestyle='--', alpha=0.3)
    axes[0, 2].set_ylabel('$\\eta$')
    axes[0, 2].set_title('(c) Efficiency by Prompt Type')

    # (d) 2D heatmap: type x length bin
    length_bins = ['Short (1-5)', 'Medium (5-20)', 'Long (20-50)', 'Very Long (50+)']
    heatmap = np.zeros((len(type_names), 4))
    for ti, t in enumerate(type_names):
        for ri, r in enumerate(results_by_type[t]):
            heatmap[ti, ri] = r['eta']
    im = axes[1, 0].imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels(length_bins, fontsize=7, rotation=15)
    axes[1, 0].set_yticks(range(len(type_names)))
    axes[1, 0].set_yticklabels(type_names)
    for i in range(len(type_names)):
        for j in range(4):
            axes[1, 0].text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=axes[1, 0], label='$\\eta$')
    axes[1, 0].set_title('(d) Efficiency Landscape')

    # (e) T_hot and T_cold scatter
    for ptype, results in results_by_type.items():
        t_hot = [r['T_hot'] for r in results]
        t_cold = [r['T_cold'] for r in results]
        axes[1, 1].scatter(t_hot, t_cold, c=type_colors[ptype], s=80, edgecolors='black', label=ptype)
    axes[1, 1].set_xlabel('$T_{hot}$')
    axes[1, 1].set_ylabel('$T_{cold}$')
    axes[1, 1].set_title('(e) Temperature Operating Points')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    eta_mean = np.mean(all_etas)
    eta_std = np.std(all_etas)
    L0_mean = np.mean(all_L0s)
    L0_std = np.std(all_L0s)
    summary = (
        f"Efficiency Landscape\n\n"
        f"Overall:\n"
        f"  eta = {eta_mean:.3f} +/- {eta_std:.3f}\n"
        f"  L0/L = {L0_mean:.3f} +/- {L0_std:.3f}\n\n"
        f"By Type:\n"
    )
    for t in type_names:
        etas_t = [r['eta'] for r in results_by_type[t]]
        summary += f"  {t}: eta={np.mean(etas_t):.3f}\n"

    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 185: Efficiency Landscape', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase185_efficiency_landscape')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Overall: eta={eta_mean:.3f}+/-{eta_std:.3f}, L0/L={L0_mean:.3f}+/-{L0_std:.3f}")
    for t in type_names:
        etas_t = [r['eta'] for r in results_by_type[t]]
        print(f"  {t}: eta={np.mean(etas_t):.3f}")
    print(f"{'=' * 70}")

    save_results('phase185_efficiency_landscape', {
        'experiment': 'Efficiency Landscape',
        'overall': {'eta_mean': float(eta_mean), 'eta_std': float(eta_std),
                     'L0_mean': float(L0_mean), 'L0_std': float(L0_std)},
        'by_type': {t: [r for r in results_by_type[t]] for t in type_names},
    })


if __name__ == '__main__':
    main()
