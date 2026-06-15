# -*- coding: utf-8 -*-
"""
Phase 223: Thermodynamic Arrow of Depth
==========================================
Does the transformer have a well-defined thermodynamic arrow?
Test: Is the "computational arrow" (layer order) the same as the
"thermodynamic arrow" (entropy decrease direction)?

Key tests:
1. Time-reversal: swap layer order and measure information loss
2. Arrow alignment: does S decrease monotonically?
3. Information-theoretic irreversibility: KL(forward || reverse)
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
    "Superconductors have zero electrical resistance",
    "The brain contains billions of neurons",
    "Algorithms determine computational complexity",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
]


def measure_arrow(model, tok, device, model_name):
    """Measure thermodynamic arrow of depth."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_T, all_U, all_S, all_P1 = [], [], [], []
    all_probs = []  # For KL computation

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_l, U_l, S_l, P1_l, probs_l = [], [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            top_probs = probs.topk(100).values.cpu().numpy()
            probs_l.append(top_probs)
            P1_l.append(float(probs.max().item()))
            S = float(-(probs * torch.log(probs + 1e-10)).sum().item())
            T_l.append(S if not np.isnan(S) else 0)
            S_l.append(S if not np.isnan(S) else 0)
        all_T.append(T_l); all_U.append(U_l); all_S.append(S_l)
        all_P1.append(P1_l); all_probs.append(probs_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_T, mean_U, mean_S, mean_P1 = avg(all_T), avg(all_U), avg(all_S), avg(all_P1)

    # 1. Monotonicity of S decrease
    dS = [mean_S[i+1] - mean_S[i] for i in range(n-1)]
    n_decreasing = sum(1 for x in dS if x < 0)
    monotonicity = n_decreasing / len(dS) if dS else 0

    # 2. Arrow strength: Spearman correlation of S with layer index
    rho_S, p_S = stats.spearmanr(range(n), mean_S)
    rho_P1, p_P1 = stats.spearmanr(range(n), mean_P1)

    # 3. KL divergence: forward vs reverse
    # KL(p_l || p_{l-1}) vs KL(p_{l-1} || p_l)
    kl_forward = []
    kl_reverse = []
    for l in range(1, n):
        kl_f_list, kl_r_list = [], []
        for pi in range(len(PROMPTS)):
            if l < len(all_probs[pi]) and l-1 < len(all_probs[pi]):
                p_curr = all_probs[pi][l] + 1e-10
                p_prev = all_probs[pi][l-1] + 1e-10
                # Normalize
                p_curr = p_curr / p_curr.sum()
                p_prev = p_prev / p_prev.sum()
                kl_f = float(np.sum(p_curr * np.log(p_curr / p_prev)))
                kl_r = float(np.sum(p_prev * np.log(p_prev / p_curr)))
                if not np.isnan(kl_f): kl_f_list.append(kl_f)
                if not np.isnan(kl_r): kl_r_list.append(kl_r)
        kl_forward.append(float(np.mean(kl_f_list)) if kl_f_list else 0)
        kl_reverse.append(float(np.mean(kl_r_list)) if kl_r_list else 0)

    # Irreversibility: mean |KL_forward - KL_reverse|
    irreversibility = [abs(kl_forward[i] - kl_reverse[i]) for i in range(len(kl_forward))]
    mean_irreversibility = float(np.mean(irreversibility))

    # 4. Arrow direction consistency across prompts
    prompt_arrows = []
    for pi in range(len(PROMPTS)):
        s_vals = all_S[pi][:n]
        rho, _ = stats.spearmanr(range(len(s_vals)), s_vals)
        prompt_arrows.append(rho)
    arrow_consistency = float(np.mean([1 if r < 0 else 0 for r in prompt_arrows]))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_T': mean_T,
        'mean_S': mean_S,
        'mean_P1': mean_P1,
        'dS': [float(x) for x in dS],
        'monotonicity': monotonicity,
        'rho_S': float(rho_S),
        'p_S': float(p_S),
        'rho_P1': float(rho_P1),
        'p_P1': float(p_P1),
        'kl_forward': kl_forward,
        'kl_reverse': kl_reverse,
        'irreversibility': irreversibility,
        'mean_irreversibility': mean_irreversibility,
        'arrow_consistency': arrow_consistency,
        'prompt_arrows': [float(x) for x in prompt_arrows],
    }


def main():
    print("=" * 70)
    print("Phase 223: Thermodynamic Arrow of Depth")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_arrow(model, tok, device, size)
        results[size] = r
        print(f"  Monotonicity: {r['monotonicity']:.2%}")
        print(f"  Arrow (S): rho={r['rho_S']:.4f}, p={r['p_S']:.2e}")
        print(f"  Arrow (P1): rho={r['rho_P1']:.4f}, p={r['p_P1']:.2e}")
        print(f"  Irreversibility: {r['mean_irreversibility']:.4f}")
        print(f"  Consistency: {r['arrow_consistency']:.2%}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) S vs layer with arrow
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['mean_S'])), r['mean_S'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Entropy S')
    axes[0, 0].set_title('(a) Entropy vs Depth')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].annotate('Arrow ->', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)

    # (b) dS/dl
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['dS'])), r['dS'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(y=0, color='red', ls='--', alpha=0.5)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('dS/dLayer')
    axes[0, 1].set_title('(b) Entropy Rate of Change')
    axes[0, 1].legend(fontsize=8)

    # (c) KL forward vs reverse
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['kl_forward'])), r['kl_forward'], '-',
                       color=colors[size], lw=2, label=f'{size} fwd')
        axes[0, 2].plot(range(len(r['kl_reverse'])), r['kl_reverse'], '--',
                       color=colors[size], lw=1.5, alpha=0.6, label=f'{size} rev')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].set_title('(c) Forward vs Reverse KL')
    axes[0, 2].legend(fontsize=7)

    # (d) Irreversibility profile
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['irreversibility'])), r['irreversibility'],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('|KL_fwd - KL_rev|')
    axes[1, 0].set_title('(d) Irreversibility')
    axes[1, 0].legend(fontsize=8)

    # (e) Per-prompt arrows
    for size, r in results.items():
        axes[1, 1].hist(r['prompt_arrows'], bins=10, alpha=0.5, color=colors[size], label=size)
    axes[1, 1].axvline(x=0, color='gray', ls='--')
    axes[1, 1].set_xlabel('Spearman rho (S vs layer)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('(e) Arrow Distribution Across Prompts')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Thermodynamic Arrow\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  S monotonic: {r['monotonicity']:.0%}\n"
        summary += f"  rho(S,l)  = {r['rho_S']:.3f}\n"
        summary += f"  rho(P1,l) = {r['rho_P1']:.3f}\n"
        summary += f"  Irrev     = {r['mean_irreversibility']:.4f}\n"
        summary += f"  Consist   = {r['arrow_consistency']:.0%}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 223: Thermodynamic Arrow of Depth", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase223_arrow')
    plt.close()
    save_results('phase223_arrow', {'experiment': 'Thermodynamic Arrow', 'results': results})


if __name__ == '__main__':
    main()
