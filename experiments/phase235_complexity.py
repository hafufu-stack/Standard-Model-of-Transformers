# -*- coding: utf-8 -*-
"""
Phase 235: Prompt Complexity Thermodynamics
=============================================
How does prompt complexity affect the thermodynamic profile?
Test: simple facts, complex reasoning, mathematical, creative, nonsense.
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

PROMPT_CATEGORIES = {
    'simple': [
        "The sky is blue because",
        "Cats are popular pets because",
        "The sun rises in the east and",
        "Water is essential for life because",
        "Trees produce oxygen through",
    ],
    'complex': [
        "The implications of quantum entanglement for faster-than-light communication remain",
        "Despite advances in neuroscience the hard problem of consciousness continues to",
        "The relationship between dark matter and dark energy in the expanding universe suggests",
        "Emergent properties in complex adaptive systems demonstrate that reductionism fails when",
        "The anthropic principle raises philosophical questions about the fine-tuning of",
    ],
    'math': [
        "The integral of e to the power of negative x squared from negative infinity to infinity equals",
        "The eigenvalues of a symmetric positive definite matrix are always",
        "By the fundamental theorem of algebra every polynomial of degree n has exactly",
        "The Riemann zeta function evaluated at negative even integers gives",
        "The determinant of the product of two matrices equals the product of",
    ],
    'creative': [
        "Once upon a time in a kingdom made entirely of glass there lived",
        "The robot looked at the sunset and for the first time felt",
        "In the year three thousand humanity discovered that dreams were actually",
        "The last tree on Earth whispered to the wind a secret about",
        "When the ocean learned to sing the first melody it chose was",
    ],
    'nonsense': [
        "Purple elephants frantically calculated the square root of",
        "The moon decided to become a professional dancer and",
        "Seven abstract thoughts collided in a vacuum creating",
        "Yesterday tomorrow forgot to remember the color of",
        "Silence tasted exactly like the sound of growing",
    ],
}


def profile_complexity(model, tok, device, model_name):
    """Measure thermodynamic profiles for different prompt categories."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    category_results = {}
    for cat_name, prompts in PROMPT_CATEGORIES.items():
        all_T, all_P1, all_U = [], [], []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            T_l, P1_l, U_l = [], [], []
            for hs in out.hidden_states:
                h = hs[0, -1, :].float()
                U_l.append(h.norm().item())
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                P1_l.append(float(probs.max().item()))
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_l.append(float(S) if not np.isnan(S) else 0)
            all_T.append(T_l); all_P1.append(P1_l); all_U.append(U_l)

        n = min(len(t) for t in all_T)
        avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
        mean_T, mean_P1, mean_U = avg(all_T), avg(all_P1), avg(all_U)

        rho_S, _ = stats.spearmanr(range(n), mean_T)
        rho_P1, _ = stats.spearmanr(range(n), mean_P1)

        category_results[cat_name] = {
            'mean_T': mean_T,
            'mean_P1': mean_P1,
            'mean_U': mean_U,
            'T_final': mean_T[-1],
            'P1_final': mean_P1[-1],
            'rho_S': float(rho_S),
            'rho_P1': float(rho_P1),
            'T_range': max(mean_T) - min(mean_T),
        }

    return {
        'model': model_name,
        'categories': category_results,
    }


def main():
    print("=" * 70)
    print("Phase 235: Prompt Complexity Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = profile_complexity(model, tok, device, size)
        results[size] = r
        for cat, cr in r['categories'].items():
            print(f"  {cat:>10}: T_final={cr['T_final']:.2f}, P1={cr['P1_final']:.3f}, rho_S={cr['rho_S']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    cat_colors = {
        'simple': '#2ecc71', 'complex': '#e74c3c', 'math': '#3498db',
        'creative': '#9b59b6', 'nonsense': '#f39c12',
    }
    cats = list(PROMPT_CATEGORIES.keys())

    # (a) T profiles by category (0.5B)
    r05 = results['0.5B']['categories']
    for cat in cats:
        x = np.linspace(0, 1, len(r05[cat]['mean_T']))
        axes[0, 0].plot(x, r05[cat]['mean_T'], '-', color=cat_colors[cat], lw=2, label=cat)
    axes[0, 0].set_xlabel('Normalized Depth'); axes[0, 0].set_ylabel('T')
    axes[0, 0].set_title('(a) T by Complexity (0.5B)')
    axes[0, 0].legend(fontsize=7)

    # (b) T profiles by category (1.5B)
    r15 = results['1.5B']['categories']
    for cat in cats:
        x = np.linspace(0, 1, len(r15[cat]['mean_T']))
        axes[0, 1].plot(x, r15[cat]['mean_T'], '-', color=cat_colors[cat], lw=2, label=cat)
    axes[0, 1].set_xlabel('Normalized Depth'); axes[0, 1].set_ylabel('T')
    axes[0, 1].set_title('(b) T by Complexity (1.5B)')
    axes[0, 1].legend(fontsize=7)

    # (c) P1 profiles
    for cat in cats:
        x = np.linspace(0, 1, len(r15[cat]['mean_P1']))
        axes[0, 2].plot(x, r15[cat]['mean_P1'], '-', color=cat_colors[cat], lw=2, label=cat)
    axes[0, 2].set_xlabel('Normalized Depth'); axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) P1 by Complexity (1.5B)')
    axes[0, 2].legend(fontsize=7)

    # (d) T_final comparison
    x = np.arange(len(cats))
    width = 0.35
    for si, (size, r) in enumerate(results.items()):
        T_finals = [r['categories'][c]['T_final'] for c in cats]
        axes[1, 0].bar(x + si*width, T_finals, width, label=size,
                       color=['#3498db', '#e74c3c'][si], alpha=0.7)
    axes[1, 0].set_xticks(x + width/2)
    axes[1, 0].set_xticklabels(cats, fontsize=8)
    axes[1, 0].set_ylabel('T_final')
    axes[1, 0].set_title('(d) Final Temperature')
    axes[1, 0].legend(fontsize=8)

    # (e) Arrow strength
    for si, (size, r) in enumerate(results.items()):
        rho_vals = [r['categories'][c]['rho_S'] for c in cats]
        axes[1, 1].bar(x + si*width, rho_vals, width, label=size,
                       color=['#3498db', '#e74c3c'][si], alpha=0.7)
    axes[1, 1].set_xticks(x + width/2)
    axes[1, 1].set_xticklabels(cats, fontsize=8)
    axes[1, 1].set_ylabel('rho(S)')
    axes[1, 1].set_title('(e) Arrow Strength')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)

    # (f) Summary
    summary = "COMPLEXITY DEPENDENCE\n\n"
    for cat in cats:
        t05 = results['0.5B']['categories'][cat]['T_final']
        t15 = results['1.5B']['categories'][cat]['T_final']
        summary += f"{cat:>10}: T_f={t05:.1f}/{t15:.1f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 235: Prompt Complexity Thermodynamics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase235_complexity')
    plt.close()
    save_results('phase235_complexity', {'experiment': 'Prompt Complexity', 'results': results})


if __name__ == '__main__':
    main()
