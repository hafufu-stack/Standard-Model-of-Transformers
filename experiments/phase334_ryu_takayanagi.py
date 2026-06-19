# -*- coding: utf-8 -*-
"""
Phase 334: Holographic Entanglement -- Ryu-Takayanagi
=====================================================
The Ryu-Takayanagi formula: entanglement entropy of boundary region A
equals the area of the minimal surface in the bulk:
S_A = Area(gamma_A) / (4 G_N)
Test whether hidden state entanglement follows geometric (area) scaling.
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


def measure_ryu_takayanagi(model, tok, prompt, device):
    """Test Ryu-Takayanagi formula."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # For subsystem A = first k dimensions, compute S_A
    # and compare with boundary area
    subsystem_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    layer_results = []

    for li in range(1, n_layers + 1):
        h = hiddens[li]
        s_values = []
        area_values = []
        for frac in subsystem_fracs:
            k = max(1, int(frac * dim))
            # Subsystem A: first k dims
            h_A = h[:k]
            # Entanglement entropy of A (via participation ratio proxy)
            p_A = torch.softmax(h_A, dim=0)
            S_A = float(-torch.sum(p_A * torch.log(p_A + 1e-30)).item())
            s_values.append(S_A)
            # Boundary area: proportional to surface between A and complement
            area = 2.0 * float(frac * (1 - frac))  # Normalized boundary area
            area_values.append(area)

        # RT test: S_A proportional to Area
        if np.std(area_values) > 1e-10:
            r, p = stats.pearsonr(s_values, area_values)
            slope, intercept, _, _, _ = stats.linregress(area_values, s_values)
        else:
            r, p, slope, intercept = 0, 1, 0, 0

        layer_results.append({
            'r_rt': round(float(r), 4),
            'p_rt': round(float(p), 6),
            'slope': round(float(slope), 4),  # G_N proxy
            'intercept': round(float(intercept), 4),
        })

    # Average RT correlation
    avg_r = float(np.mean([lr['r_rt'] for lr in layer_results]))
    avg_slope = float(np.mean([lr['slope'] for lr in layer_results]))

    # Page curve: S_A as function of subsystem size
    # Should be symmetric around 0.5 (Page curve)
    h_mid = hiddens[n_layers // 2]
    page_curve = []
    for frac in subsystem_fracs:
        k = max(1, int(frac * dim))
        h_sub = h_mid[:k]
        p_sub = torch.softmax(h_sub, dim=0)
        S_sub = float(-torch.sum(p_sub * torch.log(p_sub + 1e-30)).item())
        page_curve.append(S_sub)

    # Page curve symmetry: S(f) should equal S(1-f)
    sym_error = 0
    for i in range(len(subsystem_fracs) // 2):
        j = len(subsystem_fracs) - 1 - i
        sym_error += abs(page_curve[i] - page_curve[j])
    sym_error /= max(1, len(subsystem_fracs) // 2)

    return {
        'avg_r_rt': round(float(avg_r), 4),
        'avg_slope': round(float(avg_slope), 4),
        'page_curve': [round(p, 4) for p in page_curve],
        'page_symmetry_error': round(float(sym_error), 4),
        'rt_per_layer': [lr['r_rt'] for lr in layer_results],
        'rt_holds': avg_r > 0.5,
    }


def main():
    print("=" * 70)
    print("Phase 334: Ryu-Takayanagi")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        rt_data = []
        for prompt in PROMPTS:
            r = measure_ryu_takayanagi(model, tok, prompt, device)
            rt_data.append(r)

        n = len(rt_data[0]['rt_per_layer'])
        all_results[size] = {
            'avg_r_rt': round(float(np.mean([r['avg_r_rt'] for r in rt_data])), 4),
            'avg_slope': round(float(np.mean([r['avg_slope'] for r in rt_data])), 4),
            'page_symmetry_error': round(float(np.mean([r['page_symmetry_error'] for r in rt_data])), 4),
            'rt_per_layer': [round(float(np.mean([r['rt_per_layer'][i] for r in rt_data])), 4)
                           for i in range(n)],
            'page_curve': [round(float(np.mean([r['page_curve'][i] for r in rt_data])), 4)
                          for i in range(len(rt_data[0]['page_curve']))],
            'rt_holds': sum(1 for r in rt_data if r['rt_holds']) >= 4,
        }
        holds = 'YES' if all_results[size]['rt_holds'] else 'NO'
        print(f"  RT correlation: {all_results[size]['avg_r_rt']:.4f}")
        print(f"  G_N (slope): {all_results[size]['avg_slope']:.4f}")
        print(f"  RT holds: {holds}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for size, data in all_results.items():
        axes[0, 0].plot(data['rt_per_layer'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('RT correlation')
    axes[0, 0].set_title('(a) RT Correlation per Layer', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(fracs, data['page_curve'], '-o', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Subsystem fraction'); axes[0, 1].set_ylabel('S_A')
    axes[0, 1].set_title('(b) Page Curve', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['avg_r_rt'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].axhline(0.5, color='gold', ls='--', lw=2, label='Threshold')
    axes[0, 2].set_ylabel('r'); axes[0, 2].set_title('(c) Mean RT Correlation', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['page_symmetry_error'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Error'); axes[1, 0].set_title('(d) Page Symmetry Error', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "RYU-TAKAYANAGI\n\n"
    txt += "S_A = Area / (4*G_N)\n\n"
    for s in sizes:
        d = all_results[s]
        holds = 'YES' if d['rt_holds'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  r_RT = {d['avg_r_rt']:.3f}\n"
        txt += f"  G_N = {d['avg_slope']:.3f}\n"
        txt += f"  RT: {holds}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 334: Ryu-Takayanagi Formula", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase334_ryu_takayanagi')
    plt.close()
    save_results('phase334_ryu_takayanagi', {'experiment': 'Ryu-Takayanagi', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
