# -*- coding: utf-8 -*-
"""
Phase 309: Hawking Radiation -- Information at the Boundary
============================================================
In black hole physics, Hawking radiation carries information from the
event horizon. Analogy: the transformer's final layer "radiates"
information to the output logits.
Measure: what fraction of internal information survives to the output?
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_hawking_radiation(model, tok, prompt, device):
    """Measure information radiation from hidden states to output."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    logits = out.logits[0, -1, :].float()  # (vocab,)

    # Output entropy (logit distribution)
    probs = torch.softmax(logits, dim=0)
    S_output = -(probs * torch.log(probs + 1e-15)).sum().item()

    # Information at each layer
    layer_info = []
    layer_mutual_info = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()

        # Hidden state entropy
        h_sq = h ** 2
        h_p = h_sq / (h_sq.sum() + 1e-10)
        S_h = -(h_p * torch.log(h_p + 1e-15)).sum().item()

        # Effective rank
        _, s, _ = torch.linalg.svd(out.hidden_states[li][0].float(), full_matrices=False)
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = torch.exp(-(s_norm * torch.log(s_norm + 1e-15)).sum()).item()

        layer_info.append({
            'S_hidden': float(S_h),
            'eff_rank': float(eff_rank),
        })

    # Radiation efficiency: S_output / S_final_hidden
    S_final = layer_info[-1]['S_hidden']
    radiation_eff = S_output / (S_final + 1e-10)

    # Information retention curve
    retention = [S_output / (li['S_hidden'] + 1e-10) for li in layer_info]

    # Hawking temperature (effective temperature of output distribution)
    # T_Hawking ~ 1/log(1/P1) where P1 is the top probability
    P1 = probs.max().item()
    T_hawking = 1.0 / (np.log(1.0 / (P1 + 1e-10)) + 1e-10)

    return {
        'S_output': round(S_output, 4),
        'S_hidden': [round(li['S_hidden'], 4) for li in layer_info],
        'eff_rank': [round(li['eff_rank'], 4) for li in layer_info],
        'radiation_eff': round(float(radiation_eff), 4),
        'retention': [round(r, 6) for r in retention],
        'T_hawking': round(float(T_hawking), 4),
        'P1_output': round(float(P1), 6),
    }


def main():
    print("=" * 70)
    print("Phase 309: Hawking Radiation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        hr_data = []
        for prompt in PROMPTS:
            hr = measure_hawking_radiation(model, tok, prompt, device)
            hr_data.append(hr)

        n = len(hr_data[0]['S_hidden'])
        avg_S_h = [float(np.mean([h['S_hidden'][i] for h in hr_data])) for i in range(n)]
        avg_eff = [float(np.mean([h['eff_rank'][i] for h in hr_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n - 1,
            'avg_S_output': round(float(np.mean([h['S_output'] for h in hr_data])), 4),
            'avg_S_hidden': [round(s, 4) for s in avg_S_h],
            'avg_eff_rank': [round(e, 4) for e in avg_eff],
            'avg_radiation_eff': round(float(np.mean([h['radiation_eff'] for h in hr_data])), 4),
            'avg_T_hawking': round(float(np.mean([h['T_hawking'] for h in hr_data])), 4),
            'avg_P1': round(float(np.mean([h['P1_output'] for h in hr_data])), 6),
        }
        print(f"  S_output = {all_results[size]['avg_S_output']:.4f}")
        print(f"  Radiation eff = {all_results[size]['avg_radiation_eff']:.4f}")
        print(f"  T_Hawking = {all_results[size]['avg_T_hawking']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Hidden entropy profile
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_S_hidden'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Hidden State Entropy')
    axes[0, 0].set_title('(a) Hidden Entropy Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Effective rank
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_eff_rank'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Effective Rank')
    axes[0, 1].set_title('(b) Effective Rank Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Hawking temperature
    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['avg_T_hawking'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Hawking Temperature')
    axes[0, 2].set_title('(c) T_Hawking', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # (d) Radiation efficiency
    axes[1, 0].bar(sizes, [all_results[s]['avg_radiation_eff'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Radiation Efficiency')
    axes[1, 0].set_title('(d) S_output / S_hidden', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) S_output vs S_hidden_final
    for size, data in all_results.items():
        axes[1, 1].scatter([data['avg_S_hidden'][-1]], [data['avg_S_output']],
                          color=colors[size], s=100, label=size, zorder=5)
    axes[1, 1].plot([0, 10], [0, 10], 'k--', lw=1, label='S_out = S_hid')
    axes[1, 1].set_xlabel('S_hidden (final layer)')
    axes[1, 1].set_ylabel('S_output')
    axes[1, 1].set_title('(e) Output vs Hidden Entropy', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "HAWKING RADIATION\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  S_out = {d['avg_S_output']:.2f}\n"
        txt += f"  T_H = {d['avg_T_hawking']:.3f}\n"
        txt += f"  Rad eff = {d['avg_radiation_eff']:.3f}\n\n"
    txt += "T_H = 1/log(1/P1)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 309: Hawking Radiation -- Information at the Boundary",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase309_hawking')
    plt.close()

    save_results('phase309_hawking', {
        'experiment': 'Hawking Radiation',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
