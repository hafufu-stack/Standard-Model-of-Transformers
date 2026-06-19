# -*- coding: utf-8 -*-
"""
Phase 332: c-theorem -- Irreversibility of RG Flow
=====================================================
Zamolodchikov's c-theorem: in 2D QFT, there exists a function c(g)
that decreases monotonically along RG flow and equals the central
charge at fixed points: c_UV >= c_IR.
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


def measure_c_function(model, tok, prompt, device):
    """Measure the c-function along the RG flow (layer direction)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # c-function: related to the 2-point correlator of T_mu_nu
    # Proxy: c(l) ~ <T(l) T(l)> where T is the stress tensor
    # Use: c(l) ~ variance(h_l) * dim(h_l) (energy fluctuation)
    c_values = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        var = float(torch.var(h).item())
        dim = h.shape[0]
        c = var * dim  # Central charge proxy
        c_values.append(c)

    # Alternative: entanglement entropy (EE) proxy for c
    # c ~ 3 * S_EE / log(L) for 1+1 CFT
    ee_values = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        p = torch.softmax(h, dim=0)
        s = float(-torch.sum(p * torch.log(p + 1e-30)).item())
        ee_values.append(s)

    # Check monotonicity (c-theorem predicts c should decrease)
    c_arr = np.array(c_values)
    monotonic_decrease = 0
    for i in range(1, len(c_arr)):
        if c_arr[i] <= c_arr[i-1]:
            monotonic_decrease += 1
    mono_frac = monotonic_decrease / (len(c_arr) - 1) if len(c_arr) > 1 else 0

    # Fit: c(l) = c_UV * exp(-alpha * l) + c_IR
    layers = np.arange(len(c_arr))
    if len(layers) > 3:
        try:
            from scipy.optimize import curve_fit
            def exp_decay(x, c_uv, alpha, c_ir):
                return c_uv * np.exp(-alpha * x) + c_ir
            popt, _ = curve_fit(exp_decay, layers, c_arr, p0=[c_arr[0], 0.1, c_arr[-1]],
                               maxfev=5000)
            c_uv, alpha, c_ir = popt
            c_fit = exp_decay(layers, *popt)
            r2 = 1 - np.sum((c_arr - c_fit)**2) / (np.sum((c_arr - np.mean(c_arr))**2) + 1e-30)
        except:
            c_uv, alpha, c_ir, r2 = c_arr[0], 0, c_arr[-1], 0
    else:
        c_uv, alpha, c_ir, r2 = c_arr[0], 0, c_arr[-1], 0

    # Ratio c_UV / c_IR
    ratio = float(c_arr[0] / (c_arr[-1] + 1e-10))

    return {
        'c_values': [round(c, 4) for c in c_values],
        'ee_values': [round(e, 4) for e in ee_values],
        'mono_frac': round(float(mono_frac), 4),
        'c_uv': round(float(c_uv), 4),
        'c_ir': round(float(c_ir), 4),
        'alpha': round(float(alpha), 4),
        'r2_fit': round(float(r2), 4),
        'ratio_uv_ir': round(float(ratio), 4),
        'c_theorem_holds': mono_frac > 0.6,
    }


def main():
    print("=" * 70)
    print("Phase 332: c-theorem")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ct_data = []
        for prompt in PROMPTS:
            c = measure_c_function(model, tok, prompt, device)
            ct_data.append(c)

        n = len(ct_data[0]['c_values'])
        all_results[size] = {
            'avg_c': [round(float(np.mean([c['c_values'][i] for c in ct_data])), 4)
                     for i in range(n)],
            'avg_ee': [round(float(np.mean([c['ee_values'][i] for c in ct_data])), 4)
                      for i in range(n)],
            'mono_frac': round(float(np.mean([c['mono_frac'] for c in ct_data])), 4),
            'c_uv': round(float(np.mean([c['c_uv'] for c in ct_data])), 4),
            'c_ir': round(float(np.mean([c['c_ir'] for c in ct_data])), 4),
            'ratio_uv_ir': round(float(np.mean([c['ratio_uv_ir'] for c in ct_data])), 4),
            'r2_fit': round(float(np.mean([c['r2_fit'] for c in ct_data])), 4),
            'c_theorem_holds': sum(1 for c in ct_data if c['c_theorem_holds']) >= 4,
        }
        holds = 'YES' if all_results[size]['c_theorem_holds'] else 'NO'
        print(f"  Monotonic frac: {all_results[size]['mono_frac']:.4f}")
        print(f"  c_UV/c_IR: {all_results[size]['ratio_uv_ir']:.4f}")
        print(f"  c-theorem: {holds}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_c'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('c(l)')
    axes[0, 0].set_title('(a) c-function', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_ee'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('S_EE')
    axes[0, 1].set_title('(b) Entanglement Entropy', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['mono_frac'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].axhline(0.6, color='gold', ls='--', lw=2, label='Threshold')
    axes[0, 2].set_ylabel('Fraction'); axes[0, 2].set_title('(c) Monotonicity', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['ratio_uv_ir'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('c_UV/c_IR'); axes[1, 0].set_title('(d) UV/IR Ratio', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')

    txt = "ZAMOLODCHIKOV c-THEOREM\n\n"
    txt += "c_UV >= c_IR (monotonic)\n\n"
    for s in sizes:
        d = all_results[s]
        holds = 'YES' if d['c_theorem_holds'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  mono = {d['mono_frac']:.2f}\n"
        txt += f"  UV/IR = {d['ratio_uv_ir']:.2f}\n"
        txt += f"  holds: {holds}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 332: Zamolodchikov c-theorem", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase332_ctheorem')
    plt.close()
    save_results('phase332_ctheorem', {'experiment': 'c-theorem', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
