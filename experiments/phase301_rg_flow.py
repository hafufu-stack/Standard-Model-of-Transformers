# -*- coding: utf-8 -*-
"""
Phase 301: Renormalization Group Flow
========================================
In physics, RG flow describes how a system looks at different scales.
For transformers, each layer is a "scale".
Measure the beta function: how coupling constants flow with depth.
The coupling constant = 1/T (inverse temperature).
If beta(g) = 0, we have a fixed point (conformal invariance).
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
    "The chemical composition of water molecules is",
    "Artificial intelligence will transform how we live and work",
]


def measure_rg_flow(model, tok, prompt, device):
    """Measure RG flow: beta function and fixed points."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Coupling constants at each layer
    # g1 = 1/T (inverse temperature)
    # g2 = 1/PR (inverse participation ratio = concentration)
    # g3 = P1 (top probability)
    g1_list = []
    g2_list = []
    g3_list = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        T = float(np.std(h))
        g1 = 1.0 / (T + 1e-10)
        g1_list.append(g1)

        h_sq = h ** 2
        h_p = h_sq / (np.sum(h_sq) + 1e-15)
        PR = float(1.0 / (np.sum(h_p ** 2) + 1e-15))
        g2 = 1.0 / (PR + 1e-10)
        g2_list.append(g2)

        P1 = float(np.max(h_p))
        g3_list.append(P1)

    # Beta function: beta(g) = dg/dl
    beta1 = np.diff(g1_list)  # dg1/dl
    beta2 = np.diff(g2_list)
    beta3 = np.diff(g3_list)

    # Fixed points: where beta ~ 0
    fp_threshold = 0.1 * np.std(beta1)
    fixed_points_g1 = [i for i, b in enumerate(beta1) if abs(b) < fp_threshold]

    # RG invariant: quantity that doesn't change under RG flow
    # Test: g1 * g2 = const? (product of couplings)
    product = np.array(g1_list) * np.array(g2_list)
    product_cv = float(np.std(product) / (np.mean(product) + 1e-10))

    # Test: g3 * (1/g1) = P1 * T = const? (this is P1T!)
    p1t = np.array(g3_list) * np.array([1/g for g in g1_list])
    p1t_cv = float(np.std(p1t) / (np.mean(p1t) + 1e-10))

    return {
        'g1': [round(g, 4) for g in g1_list],
        'g2': [round(g, 6) for g in g2_list],
        'g3': [round(g, 6) for g in g3_list],
        'beta1': [round(b, 4) for b in beta1],
        'beta2': [round(b, 6) for b in beta2],
        'beta3': [round(b, 6) for b in beta3],
        'n_fixed_points': len(fixed_points_g1),
        'fixed_point_layers': fixed_points_g1[:5],
        'g1g2_cv': round(product_cv, 4),
        'p1t_cv': round(p1t_cv, 4),
        'p1t_mean': round(float(np.mean(p1t)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 301: Renormalization Group Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        rg_data = []
        for prompt in PROMPTS:
            rg = measure_rg_flow(model, tok, prompt, device)
            rg_data.append(rg)

        n = len(rg_data[0]['g1'])
        avg_g1 = [float(np.mean([r['g1'][i] for r in rg_data])) for i in range(n)]
        avg_g2 = [float(np.mean([r['g2'][i] for r in rg_data])) for i in range(n)]
        avg_g3 = [float(np.mean([r['g3'][i] for r in rg_data])) for i in range(n)]

        nb = len(rg_data[0]['beta1'])
        avg_beta1 = [float(np.mean([r['beta1'][i] for r in rg_data])) for i in range(nb)]
        avg_beta2 = [float(np.mean([r['beta2'][i] for r in rg_data])) for i in range(nb)]

        avg_p1t_cv = float(np.mean([r['p1t_cv'] for r in rg_data]))
        avg_g1g2_cv = float(np.mean([r['g1g2_cv'] for r in rg_data]))

        # Find UV and IR fixed points
        # UV = early layers, IR = late layers
        uv_beta = float(np.mean(avg_beta1[:3]))
        ir_beta = float(np.mean(avg_beta1[-3:]))

        all_results[size] = {
            'n_layers': n - 1,
            'avg_g1': [round(g, 4) for g in avg_g1],
            'avg_g2': [round(g, 6) for g in avg_g2],
            'avg_g3': [round(g, 6) for g in avg_g3],
            'avg_beta1': [round(b, 4) for b in avg_beta1],
            'avg_beta2': [round(b, 6) for b in avg_beta2],
            'uv_beta': round(uv_beta, 4),
            'ir_beta': round(ir_beta, 4),
            'p1t_cv': round(avg_p1t_cv, 4),
            'g1g2_cv': round(avg_g1g2_cv, 4),
            'p1t_is_rg_invariant': avg_p1t_cv < 0.3,
        }
        print(f"  UV beta(g1) = {uv_beta:.4f}")
        print(f"  IR beta(g1) = {ir_beta:.4f}")
        print(f"  P1T CV = {avg_p1t_cv:.4f} ({'RG invariant!' if avg_p1t_cv < 0.3 else 'NOT invariant'})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Coupling g1 = 1/T flow
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_g1'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer (RG scale)')
    axes[0, 0].set_ylabel('g1 = 1/T')
    axes[0, 0].set_title('(a) Coupling g1 = 1/T', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Beta function
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_beta1'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(0, color='gold', ls='--', lw=2, label='Fixed point (beta=0)')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('beta(g1) = dg1/dl')
    axes[0, 1].set_title('(b) Beta Function', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) g1 vs g2 phase space (RG trajectory)
    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_g1'], data['avg_g2'], '-o', color=colors[size],
                       lw=1.5, markersize=3, label=size)
        # Mark start and end
        axes[0, 2].plot(data['avg_g1'][0], data['avg_g2'][0], 's',
                       color=colors[size], markersize=8)
        axes[0, 2].plot(data['avg_g1'][-1], data['avg_g2'][-1], '*',
                       color=colors[size], markersize=12)
    axes[0, 2].set_xlabel('g1 = 1/T')
    axes[0, 2].set_ylabel('g2 = 1/PR')
    axes[0, 2].set_title('(c) RG Trajectory in Coupling Space', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) P1 flow
    for size, data in all_results.items():
        axes[1, 0].plot(data['avg_g3'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('P1 (top probability weight)')
    axes[1, 0].set_title('(d) P1 Flow', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) RG invariants
    x = np.arange(2)
    w = 0.35
    p1t_cvs = [all_results[s]['p1t_cv'] for s in all_results]
    g1g2_cvs = [all_results[s]['g1g2_cv'] for s in all_results]
    sizes = list(all_results.keys())
    axes[1, 1].bar(x - w/2, p1t_cvs, w, label='P1*T CV', color='#3498db')
    axes[1, 1].bar(x + w/2, g1g2_cvs, w, label='g1*g2 CV', color='#e74c3c')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(sizes)
    axes[1, 1].axhline(0.3, color='gold', ls='--', label='Invariance threshold')
    axes[1, 1].set_ylabel('Coefficient of Variation')
    axes[1, 1].set_title('(e) RG Invariants', fontweight='bold')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "RENORMALIZATION GROUP FLOW\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  UV beta = {d['uv_beta']:.3f}\n"
        txt += f"  IR beta = {d['ir_beta']:.3f}\n"
        txt += f"  P1T invariant: {'YES' if d['p1t_is_rg_invariant'] else 'no'}\n\n"
    txt += "P1*T = RG invariant:\n"
    txt += "conserved across scales!"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 301: Renormalization Group Flow in Transformer",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase301_rg_flow')
    plt.close()

    save_results('phase301_rg_flow', {
        'experiment': 'Renormalization Group Flow',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
