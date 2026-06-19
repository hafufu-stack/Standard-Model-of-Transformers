# -*- coding: utf-8 -*-
"""
Phase 315: Quantum Tunneling -- Layer Skip Dynamics
=====================================================
In quantum mechanics, tunneling allows particles to pass through
energy barriers. In transformers:
- Can information "tunnel" through layers?
- Skip connections = tunneling paths?
- Measure non-local correlations between non-adjacent layers.
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


def measure_tunneling(model, tok, prompt, device):
    """Measure quantum tunneling in transformers."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Full layer-to-layer correlation matrix
    h_list = [out.hidden_states[li][0, -1, :].float() for li in range(n_layers + 1)]

    corr_matrix = np.zeros((n_layers + 1, n_layers + 1))
    for i in range(n_layers + 1):
        for j in range(n_layers + 1):
            cos = torch.nn.functional.cosine_similarity(
                h_list[i].unsqueeze(0), h_list[j].unsqueeze(0)).item()
            corr_matrix[i, j] = cos

    # Tunneling = non-local correlations beyond nearest neighbor
    # For each layer i, check if it's more correlated with a distant layer
    # than with its immediate neighbors
    tunneling_events = []
    for i in range(2, n_layers - 1):
        nn_corr = max(corr_matrix[i, i-1], corr_matrix[i, i+1])
        for j in range(n_layers + 1):
            if abs(j - i) > 2:  # distant layer
                if corr_matrix[i, j] > nn_corr * 1.05:  # 5% more than nn
                    tunneling_events.append({
                        'from': i, 'to': j,
                        'tunnel_corr': round(float(corr_matrix[i, j]), 4),
                        'nn_corr': round(float(nn_corr), 4),
                    })

    # Tunneling probability: correlation decay with distance
    distances = []
    correlations = []
    for i in range(n_layers + 1):
        for j in range(i + 1, n_layers + 1):
            distances.append(j - i)
            correlations.append(corr_matrix[i, j])

    d_arr = np.array(distances)
    c_arr = np.array(correlations)

    # Fit exponential decay: C(d) = C0 * exp(-d/xi)
    if len(d_arr) > 3:
        log_c = np.log(np.abs(c_arr) + 1e-15)
        slope, intercept, r, _, _ = stats.linregress(d_arr, log_c)
        decay_length = -1.0 / (slope + 1e-10)  # xi
        r2_decay = r**2
    else:
        decay_length = 0
        r2_decay = 0

    return {
        'n_tunneling_events': len(tunneling_events),
        'tunneling_events': tunneling_events[:5],
        'decay_length': round(float(decay_length), 4),
        'r2_decay': round(float(r2_decay), 4),
        'corr_matrix_diag': [round(float(corr_matrix[i, i+1]), 4) for i in range(n_layers)],
    }


def main():
    print("=" * 70)
    print("Phase 315: Quantum Tunneling -- Layer Skip Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        tun_data = []
        for prompt in PROMPTS:
            t = measure_tunneling(model, tok, prompt, device)
            tun_data.append(t)

        avg_events = float(np.mean([t['n_tunneling_events'] for t in tun_data]))
        avg_decay = float(np.mean([t['decay_length'] for t in tun_data]))

        n = len(tun_data[0]['corr_matrix_diag'])
        avg_diag = [float(np.mean([t['corr_matrix_diag'][i] for t in tun_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n,
            'avg_tunneling_events': round(avg_events, 1),
            'avg_decay_length': round(avg_decay, 4),
            'avg_nn_correlations': [round(c, 4) for c in avg_diag],
            'tunneling_detected': avg_events > 1,
        }
        print(f"  Tunneling events: {avg_events:.1f}")
        print(f"  Decay length: {avg_decay:.4f}")
        print(f"  Tunneling: {'YES' if avg_events > 1 else 'NO'}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_nn_correlations'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('NN Correlation')
    axes[0, 0].set_title('(a) Nearest-Neighbor Correlation', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 1].bar(sizes, [all_results[s]['avg_tunneling_events'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 1].set_ylabel('N Events'); axes[0, 1].set_title('(b) Tunneling Events', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(sizes, [all_results[s]['avg_decay_length'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Decay Length xi'); axes[0, 2].set_title('(c) Correlation Decay Length', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')

    txt = "QUANTUM TUNNELING\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Events: {d['avg_tunneling_events']:.0f}\n"
        txt += f"  xi = {d['avg_decay_length']:.2f}\n\n"
    txt += "xi > 0: info tunnels\n"
    txt += "between distant layers"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 315: Quantum Tunneling -- Layer Skip Dynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase315_tunneling')
    plt.close()
    save_results('phase315_tunneling', {'experiment': 'Quantum Tunneling', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
