# -*- coding: utf-8 -*-
"""
Phase 349: Spontaneous Symmetry Breaking -- Higgs Mechanism
=====================================================
In the Standard Model, the Higgs mechanism gives mass to particles
through spontaneous symmetry breaking. Test whether the Transformer's
hidden states undergo SSB: a symmetric Hamiltonian with asymmetric
ground state.
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


def measure_ssb(model, tok, prompt, device):
    """Measure spontaneous symmetry breaking."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Order parameter: test if there's a direction that becomes preferred
    # Mexican hat potential: V(phi) = -mu^2 |phi|^2 + lambda |phi|^4
    # SSB happens when <phi> != 0
    order_params = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Order parameter: directional bias
        mean_h = float(torch.mean(h).item())
        std_h = float(torch.std(h).item())
        order = abs(mean_h) / (std_h + 1e-10)  # Normalized bias
        order_params.append(round(float(order), 4))

    # 2. Goldstone modes: massless excitations from SSB
    # Count near-zero eigenvalues of the "mass matrix" M^2 = d^2V/dphi^2
    goldstone_counts = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Mass matrix proxy: Hessian of energy
        # Use sorted eigenvalue distribution
        sorted_h = torch.sort(h.abs())[0]
        # Count "massless" modes: eigenvalues close to zero
        threshold = float(torch.median(sorted_h).item()) * 0.01
        n_massless = int((sorted_h < threshold).sum().item())
        goldstone_counts.append(n_massless)

    # 3. Higgs mass: the massive excitation
    higgs_masses = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        sorted_vals = torch.sort(h.abs(), descending=True)[0]
        # Higgs = heaviest mode relative to average
        higgs = float(sorted_vals[0].item()) / (float(torch.mean(sorted_vals).item()) + 1e-10)
        higgs_masses.append(round(float(higgs), 4))

    # 4. VEV (vacuum expectation value): <h> per dimension
    vev_profile = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        vev = float(torch.mean(h).item())
        vev_profile.append(round(float(vev), 6))

    # 5. Phase transition: is there a critical layer where SSB occurs?
    # Look for sudden increase in order parameter
    max_jump = 0
    critical_layer = 0
    for li in range(1, len(order_params)):
        jump = order_params[li] - order_params[li - 1]
        if jump > max_jump:
            max_jump = jump
            critical_layer = li

    return {
        'order_params': order_params,
        'goldstone_counts': goldstone_counts,
        'higgs_masses': higgs_masses,
        'vev_profile': vev_profile,
        'avg_order': round(float(np.mean(order_params)), 4),
        'avg_goldstone': round(float(np.mean(goldstone_counts)), 2),
        'avg_higgs_mass': round(float(np.mean(higgs_masses)), 4),
        'critical_layer': critical_layer,
        'ssb_present': float(np.mean(order_params[n_layers//2:])) > 2 * float(np.mean(order_params[:n_layers//4])),
    }


def main():
    print("=" * 70)
    print("Phase 349: Spontaneous Symmetry Breaking")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ssb_data = []
        for prompt in PROMPTS:
            s = measure_ssb(model, tok, prompt, device)
            ssb_data.append(s)

        n = len(ssb_data[0]['order_params'])
        all_results[size] = {
            'order_params': [round(float(np.mean([s['order_params'][i] for s in ssb_data])), 4)
                            for i in range(n)],
            'goldstone_counts': [round(float(np.mean([s['goldstone_counts'][i] for s in ssb_data])), 2)
                                for i in range(n)],
            'higgs_masses': [round(float(np.mean([s['higgs_masses'][i] for s in ssb_data])), 4)
                            for i in range(n)],
            'vev_profile': [round(float(np.mean([s['vev_profile'][i] for s in ssb_data])), 6)
                           for i in range(n)],
            'avg_order': round(float(np.mean([s['avg_order'] for s in ssb_data])), 4),
            'avg_goldstone': round(float(np.mean([s['avg_goldstone'] for s in ssb_data])), 2),
            'avg_higgs_mass': round(float(np.mean([s['avg_higgs_mass'] for s in ssb_data])), 4),
            'critical_layer': int(np.median([s['critical_layer'] for s in ssb_data])),
            'ssb_present': sum(1 for s in ssb_data if s['ssb_present']) >= 4,
        }
        ssb = 'YES' if all_results[size]['ssb_present'] else 'NO'
        print(f"  Order param: {all_results[size]['avg_order']:.4f}")
        print(f"  Goldstone: {all_results[size]['avg_goldstone']:.2f}")
        print(f"  Higgs mass: {all_results[size]['avg_higgs_mass']:.4f}")
        print(f"  Critical layer: {all_results[size]['critical_layer']}")
        print(f"  SSB: {ssb}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['order_params'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Order parameter')
    axes[0, 0].set_title('(a) Order Parameter', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['goldstone_counts'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('(b) Goldstone Modes', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['higgs_masses'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Higgs mass ratio')
    axes[0, 2].set_title('(c) Higgs Mass', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[1, 0].plot(data['vev_profile'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('VEV')
    axes[1, 0].set_title('(d) Vacuum Expectation Value', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    sizes = list(all_results.keys())
    txt = "HIGGS MECHANISM\n\n"
    for s in sizes:
        d = all_results[s]
        ssb = 'YES' if d['ssb_present'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  order = {d['avg_order']:.3f}\n"
        txt += f"  goldstone = {d['avg_goldstone']:.0f}\n"
        txt += f"  m_H = {d['avg_higgs_mass']:.2f}\n"
        txt += f"  L_c = {d['critical_layer']}\n"
        txt += f"  SSB: {ssb}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 349: Spontaneous Symmetry Breaking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase349_ssb')
    plt.close()
    save_results('phase349_ssb', {'experiment': 'SSB/Higgs', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
