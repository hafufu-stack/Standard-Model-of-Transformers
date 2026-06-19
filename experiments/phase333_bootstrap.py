# -*- coding: utf-8 -*-
"""
Phase 333: Conformal Bootstrap -- Self-Consistency
=====================================================
The conformal bootstrap: correlation functions are fully determined
by a small set of "CFT data" (dimensions, OPE coefficients) plus
crossing symmetry. Test self-consistency of the Transformer's
conformal structure.
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


def measure_bootstrap(model, tok, prompt, device):
    """Test conformal bootstrap self-consistency."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float() for li in range(n_layers + 1)]

    # 2-point correlator: G2(l1, l2) = <h(l1) . h(l2)>
    G2 = np.zeros((n_layers + 1, n_layers + 1))
    for i in range(n_layers + 1):
        for j in range(n_layers + 1):
            G2[i, j] = float(torch.dot(hiddens[i], hiddens[j]).item())

    # Check power-law decay: G2(|l1-l2|) ~ |l1-l2|^(-2*Delta)
    seps = range(1, n_layers + 1)
    avg_G2 = []
    for sep in seps:
        vals = [G2[i, i+sep] for i in range(n_layers + 1 - sep)]
        avg_G2.append(float(np.mean(vals)))

    log_seps = np.log(np.array(list(seps), dtype=float))
    log_G2 = np.log(np.abs(avg_G2) + 1e-30)
    slope, _, r, _, _ = stats.linregress(log_seps, log_G2)
    delta = -slope / 2  # Scaling dimension
    r2_power = r**2

    # 4-point correlator and crossing symmetry
    # G4(l1,l2,l3,l4) = <h1.h2><h3.h4> + <h1.h3><h2.h4> + <h1.h4><h2.h3>
    n = min(n_layers + 1, 8)
    crossing_violations = 0
    total_tests = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    s_ch = G2[i,j] * G2[k,l]  # s-channel
                    t_ch = G2[i,k] * G2[j,l]  # t-channel
                    u_ch = G2[i,l] * G2[j,k]  # u-channel
                    total = s_ch + t_ch + u_ch
                    # Crossing: s = t + u  (approximate)
                    if total != 0:
                        asym = abs(s_ch - t_ch - u_ch) / (abs(total) + 1e-10)
                        if asym > 1.0:
                            crossing_violations += 1
                    total_tests += 1

    crossing_frac = 1.0 - crossing_violations / (total_tests + 1e-10)

    # Unitarity bound: Delta >= (d-2)/2 = 0 for d=2
    unitarity_ok = delta >= -0.1

    # OPE consistency: C_{ijk} relation
    # C_{12k} ~ G3 / (G2*G2)
    ope_coeffs = []
    for k in range(1, min(n_layers, 10)):
        g3 = float(torch.dot(hiddens[0], hiddens[k]).item()) * float(torch.dot(hiddens[k], hiddens[-1]).item())
        g2_1 = G2[0, k]
        g2_2 = G2[k, -1]
        if abs(g2_1 * g2_2) > 1e-10:
            c = g3 / (g2_1 * g2_2)
            ope_coeffs.append(float(c))

    ope_stability = float(np.std(ope_coeffs) / (np.mean(np.abs(ope_coeffs)) + 1e-10)) if ope_coeffs else 1.0

    return {
        'delta': round(float(delta), 4),
        'r2_power': round(float(r2_power), 4),
        'crossing_frac': round(float(crossing_frac), 4),
        'unitarity_ok': bool(unitarity_ok),
        'ope_stability': round(float(ope_stability), 4),
        'n_ope_coeffs': len(ope_coeffs),
        'G2_diagonal': [round(float(G2[i, i]), 4) for i in range(n_layers + 1)],
    }


def main():
    print("=" * 70)
    print("Phase 333: Conformal Bootstrap")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        bs_data = []
        for prompt in PROMPTS:
            b = measure_bootstrap(model, tok, prompt, device)
            bs_data.append(b)

        n = len(bs_data[0]['G2_diagonal'])
        all_results[size] = {
            'delta': round(float(np.mean([b['delta'] for b in bs_data])), 4),
            'r2_power': round(float(np.mean([b['r2_power'] for b in bs_data])), 4),
            'crossing_frac': round(float(np.mean([b['crossing_frac'] for b in bs_data])), 4),
            'unitarity_ok': all(b['unitarity_ok'] for b in bs_data),
            'ope_stability': round(float(np.mean([b['ope_stability'] for b in bs_data])), 4),
            'G2_diagonal': [round(float(np.mean([b['G2_diagonal'][i] for b in bs_data])), 4)
                          for i in range(n)],
        }
        uni = 'YES' if all_results[size]['unitarity_ok'] else 'NO'
        print(f"  Delta: {all_results[size]['delta']:.4f}")
        print(f"  Power law R2: {all_results[size]['r2_power']:.4f}")
        print(f"  Crossing: {all_results[size]['crossing_frac']:.4f}")
        print(f"  Unitarity: {uni}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['G2_diagonal'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('G2(l,l)')
    axes[0, 0].set_title('(a) 2-Point Correlator', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.25
    axes[0, 1].bar(x - w, [all_results[s]['delta'] for s in sizes], w,
                  label='Delta', color='#3498db')
    axes[0, 1].bar(x, [all_results[s]['r2_power'] for s in sizes], w,
                  label='R2', color='#e74c3c')
    axes[0, 1].bar(x + w, [all_results[s]['crossing_frac'] for s in sizes], w,
                  label='Crossing', color='#2ecc71')
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(sizes)
    axes[0, 1].set_title('(b) CFT Data', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(sizes, [all_results[s]['ope_stability'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('CV'); axes[0, 2].set_title('(c) OPE Stability', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')

    txt = "CONFORMAL BOOTSTRAP\n\n"
    for s in sizes:
        d = all_results[s]
        uni = 'YES' if d['unitarity_ok'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  Delta = {d['delta']:.3f}\n"
        txt += f"  R2 = {d['r2_power']:.3f}\n"
        txt += f"  Cross = {d['crossing_frac']:.3f}\n"
        txt += f"  Unitarity: {uni}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 333: Conformal Bootstrap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase333_bootstrap')
    plt.close()
    save_results('phase333_bootstrap', {'experiment': 'Conformal Bootstrap', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
