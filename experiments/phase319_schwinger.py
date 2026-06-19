# -*- coding: utf-8 -*-
"""
Phase 319: Schwinger Effect -- Pair Production from Strong Fields
==================================================================
The Schwinger effect: strong electric fields can create
particle-antiparticle pairs from the vacuum.
In transformers: strong activations at one layer may create
"pairs" of correlated features in subsequent layers.
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


def measure_schwinger(model, tok, prompt, device):
    """Measure Schwinger pair production analogy."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # "Field strength" at each layer = gradient of energy
    layer_E = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        E = float((h ** 2).sum().item())
        layer_E.append(E)

    field_strength = [abs(layer_E[i+1] - layer_E[i]) for i in range(n_layers)]

    # "Pair production": emergence of anti-correlated feature pairs
    # Look for dimensions that flip sign between layers
    pair_counts = []
    for li in range(n_layers):
        h1 = out.hidden_states[li][0, -1, :].float()
        h2 = out.hidden_states[li + 1][0, -1, :].float()
        # Count sign flips
        signs1 = (h1 > 0).float()
        signs2 = (h2 > 0).float()
        flips = (signs1 != signs2).sum().item()
        pair_counts.append(int(flips))

    # Schwinger threshold: pair production rate ~ exp(-pi*m^2/E)
    # Test if pair count correlates with field strength
    if len(field_strength) > 3:
        log_pairs = np.log(np.array(pair_counts, dtype=float) + 1)
        log_field = np.log(np.array(field_strength) + 1e-10)
        slope, _, r, _, _ = stats.linregress(log_field, log_pairs)
        schwinger_r2 = r**2
    else:
        slope = 0
        schwinger_r2 = 0

    # Critical field strength (threshold for pair production > 50%)
    D = model.config.hidden_size
    pair_rate = [pc / D for pc in pair_counts]
    critical_fields = [f for f, pr in zip(field_strength, pair_rate) if pr > 0.5]
    E_critical = float(np.min(critical_fields)) if critical_fields else float(np.max(field_strength))

    return {
        'field_strength': [round(f, 4) for f in field_strength],
        'pair_counts': pair_counts,
        'pair_rate': [round(pr, 4) for pr in pair_rate],
        'schwinger_slope': round(float(slope), 4),
        'schwinger_r2': round(float(schwinger_r2), 4),
        'E_critical': round(float(E_critical), 4),
        'mean_pair_rate': round(float(np.mean(pair_rate)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 319: Schwinger Effect")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        sch_data = []
        for prompt in PROMPTS:
            s = measure_schwinger(model, tok, prompt, device)
            sch_data.append(s)

        n = len(sch_data[0]['field_strength'])
        avg_field = [float(np.mean([s['field_strength'][i] for s in sch_data])) for i in range(n)]
        avg_rate = [float(np.mean([s['pair_rate'][i] for s in sch_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n,
            'avg_field_strength': [round(f, 4) for f in avg_field],
            'avg_pair_rate': [round(r, 4) for r in avg_rate],
            'mean_pair_rate': round(float(np.mean(avg_rate)), 4),
            'E_critical': round(float(np.mean([s['E_critical'] for s in sch_data])), 4),
            'schwinger_r2': round(float(np.mean([s['schwinger_r2'] for s in sch_data])), 4),
        }
        print(f"  Mean pair rate: {all_results[size]['mean_pair_rate']:.4f}")
        print(f"  E_critical: {all_results[size]['E_critical']:.4f}")
        print(f"  Schwinger R2: {all_results[size]['schwinger_r2']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_field_strength'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Field Strength |dE/dl|')
    axes[0, 0].set_title('(a) Field Strength', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_pair_rate'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(0.5, color='gold', ls='--', lw=1, label='50% threshold')
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Pair Rate')
    axes[0, 1].set_title('(b) Pair Production Rate', fontweight='bold')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['E_critical'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('E_critical'); axes[0, 2].set_title('(c) Critical Field', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "SCHWINGER EFFECT\n\n"
    txt += "Strong field -> pair creation\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  pair rate = {d['mean_pair_rate']:.3f}\n"
        txt += f"  E_c = {d['E_critical']:.1f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 319: Schwinger Effect -- Pair Production",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase319_schwinger')
    plt.close()
    save_results('phase319_schwinger', {'experiment': 'Schwinger Effect', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
