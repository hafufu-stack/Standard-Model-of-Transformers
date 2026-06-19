# -*- coding: utf-8 -*-
"""
Phase 306: Entanglement Entropy -- Area Law vs Volume Law
==========================================================
In quantum systems:
- Area law: S(A) ~ |boundary(A)| (gapped, short-range entangled)
- Volume law: S(A) ~ |A| (critical, long-range entangled)
- Log correction: S(A) ~ c/3 * log|A| (1+1D CFT)
Measure how information entropy scales with subsystem size.
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
    "The theory of general relativity predicts that a massive object curves the spacetime around it",
    "In quantum mechanics the uncertainty principle states that position and momentum cannot both be",
    "Machine learning models are trained by minimizing a loss function over the training dataset",
    "The speed of light is constant in all reference frames which leads to time dilation effects",
    "Evolution explains the diversity of life on Earth through natural selection and genetic mutation",
    "The laws of thermodynamics govern all energy transformations in physical and chemical processes",
]


def measure_entanglement_entropy(model, tok, prompt, device):
    """Measure entanglement entropy scaling."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Use middle layer for clearest signal
    mid = len(out.hidden_states) // 2
    h = out.hidden_states[mid][0].float()  # (seq, D)

    # Entanglement entropy: for subsystem A = first k tokens
    # Compute SVD of h[:k], entropy from singular values
    subsystem_sizes = []
    entropies = []

    for k in range(2, seq_len - 1):
        h_sub = h[:k]  # (k, D)
        _, s, _ = torch.linalg.svd(h_sub, full_matrices=False)
        s_norm = s / (s.sum() + 1e-10)
        ent = -(s_norm * torch.log(s_norm + 1e-15)).sum().item()
        subsystem_sizes.append(k)
        entropies.append(ent)

    sizes = np.array(subsystem_sizes, dtype=float)
    ents = np.array(entropies)

    # Test area law: S ~ const (boundary is constant in 1D)
    # Test volume law: S ~ k
    # Test log law: S ~ c/3 * log(k) (CFT prediction)

    # Linear fit (volume law)
    slope_vol, int_vol, r_vol, _, _ = stats.linregress(sizes, ents)
    r2_vol = r_vol**2

    # Log fit (CFT)
    log_sizes = np.log(sizes)
    slope_log, int_log, r_log, _, _ = stats.linregress(log_sizes, ents)
    r2_log = r_log**2

    # Central charge from log fit: S = c/3 * log(L) + const
    c_from_ent = 3 * slope_log

    # Area law: S ~ const (fit with constant)
    r2_area = 1 - np.var(ents - np.mean(ents)) / (np.var(ents) + 1e-10)

    return {
        'subsystem_sizes': [int(s) for s in subsystem_sizes],
        'entropies': [round(e, 4) for e in entropies],
        'r2_volume': round(float(r2_vol), 4),
        'r2_log': round(float(r2_log), 4),
        'r2_area': round(float(r2_area), 4),
        'c_from_entropy': round(float(c_from_ent), 4),
        'volume_slope': round(float(slope_vol), 4),
        'log_slope': round(float(slope_log), 4),
        'seq_len': seq_len,
    }


def main():
    print("=" * 70)
    print("Phase 306: Entanglement Entropy Scaling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ee_data = []
        for prompt in PROMPTS:
            ee = measure_entanglement_entropy(model, tok, prompt, device)
            ee_data.append(ee)

        avg_r2_vol = float(np.mean([d['r2_volume'] for d in ee_data]))
        avg_r2_log = float(np.mean([d['r2_log'] for d in ee_data]))
        avg_c = float(np.mean([d['c_from_entropy'] for d in ee_data]))

        # Best fit
        if avg_r2_log > avg_r2_vol:
            best_law = 'Log (CFT)'
        elif avg_r2_vol > 0.9:
            best_law = 'Volume'
        else:
            best_law = 'Area'

        all_results[size] = {
            'avg_r2_volume': round(avg_r2_vol, 4),
            'avg_r2_log': round(avg_r2_log, 4),
            'avg_c_from_entropy': round(avg_c, 4),
            'best_law': best_law,
            'individual': ee_data,
        }
        print(f"  R2 volume: {avg_r2_vol:.4f}")
        print(f"  R2 log (CFT): {avg_r2_log:.4f}")
        print(f"  c from entropy: {avg_c:.4f}")
        print(f"  Best fit: {best_law}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) S vs subsystem size (first prompt)
    for size, data in all_results.items():
        d = data['individual'][0]
        axes[0, 0].plot(d['subsystem_sizes'], d['entropies'], 'o-',
                       color=colors[size], lw=1.5, markersize=3, label=size)
    axes[0, 0].set_xlabel('Subsystem Size k')
    axes[0, 0].set_ylabel('Entanglement Entropy S(k)')
    axes[0, 0].set_title('(a) Entropy vs Subsystem Size', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) S vs log(k) (CFT test)
    for size, data in all_results.items():
        d = data['individual'][0]
        log_k = np.log(d['subsystem_sizes'])
        axes[0, 1].plot(log_k, d['entropies'], 'o-',
                       color=colors[size], lw=1.5, markersize=3, label=size)
    axes[0, 1].set_xlabel('log(k)')
    axes[0, 1].set_ylabel('S(k)')
    axes[0, 1].set_title('(b) S vs log(k) -- CFT Test', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) R2 comparison
    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.35
    axes[0, 2].bar(x - w/2, [all_results[s]['avg_r2_volume'] for s in sizes], w,
                  label='Volume law', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['avg_r2_log'] for s in sizes], w,
                  label='Log law (CFT)', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_ylabel('R2')
    axes[0, 2].set_title('(c) Law Comparison', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Central charge from entropy
    c_vals = [all_results[s]['avg_c_from_entropy'] for s in sizes]
    axes[1, 0].bar(sizes, c_vals, color=[colors[s] for s in sizes])
    axes[1, 0].axhline(1.0, color='gold', ls='--', lw=2, label='c=1')
    axes[1, 0].set_ylabel('Central Charge c')
    axes[1, 0].set_title('(d) c from Entanglement', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e)
    axes[1, 1].axis('off')

    # (f) Summary
    txt = "ENTANGLEMENT ENTROPY\n\n"
    txt += "CFT: S = (c/3) log(k) + const\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Best: {d['best_law']}\n"
        txt += f"  c = {d['avg_c_from_entropy']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 306: Entanglement Entropy -- Area vs Volume vs Log",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase306_entanglement')
    plt.close()

    save_results('phase306_entanglement', {
        'experiment': 'Entanglement Entropy Scaling',
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'individual'} for k, v in all_results.items()},
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
