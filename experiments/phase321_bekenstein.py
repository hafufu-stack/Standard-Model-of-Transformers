# -*- coding: utf-8 -*-
"""
Phase 321: Bekenstein Bound -- Information-Area Law
=====================================================
The Bekenstein bound: maximum information in a region is
proportional to the boundary area, not volume.
S <= 2*pi*R*E / (hbar*c)
Test if transformer hidden states obey this bound.
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


def measure_bekenstein(model, tok, prompt, device):
    """Test Bekenstein bound on hidden state information."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    D = model.config.hidden_size

    # For subsystems of increasing size
    subsystem_info = []

    for k in range(2, min(seq_len, 20)):
        # Information content = entropy of SVD spectrum
        mid = n_layers // 2
        h_sub = out.hidden_states[mid][0, :k, :].float()  # (k, D)
        _, s, _ = torch.linalg.svd(h_sub, full_matrices=False)
        s_norm = s / (s.sum() + 1e-10)
        S = -(s_norm * torch.log(s_norm + 1e-15)).sum().item()

        # Energy = norm^2
        E = float((h_sub ** 2).sum().item())

        # "Radius" = number of tokens = k
        R = k

        # "Area" = boundary of the subsystem = 2*D (two boundaries)
        A = 2 * D

        # Bekenstein bound: S <= 2*pi*R*E
        # (with natural units where hbar*c = 1)
        bekenstein_limit = 2 * np.pi * R * np.sqrt(E)
        ratio = S / (bekenstein_limit + 1e-10)

        subsystem_info.append({
            'k': k, 'S': float(S), 'E': float(E),
            'bekenstein_limit': float(bekenstein_limit),
            'ratio': float(ratio),
            'saturated': ratio > 0.5,
        })

    # Scaling test: S vs k
    ks = [si['k'] for si in subsystem_info]
    Ss = [si['S'] for si in subsystem_info]

    # Area scaling: S ~ k^(d-1)/d for d-dim (boundary scaling)
    log_k = np.log(ks)
    log_S = np.log(np.array(Ss) + 1e-10)
    slope, _, r, _, _ = stats.linregress(log_k, log_S)
    scaling_exp = slope  # S ~ k^alpha

    return {
        'subsystem_info': subsystem_info,
        'scaling_exponent': round(float(scaling_exp), 4),
        'scaling_r2': round(float(r**2), 4),
        'mean_bekenstein_ratio': round(float(np.mean([si['ratio'] for si in subsystem_info])), 6),
        'max_bekenstein_ratio': round(float(np.max([si['ratio'] for si in subsystem_info])), 6),
        'n_saturated': sum(1 for si in subsystem_info if si['saturated']),
    }


def main():
    print("=" * 70)
    print("Phase 321: Bekenstein Bound")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        bek_data = []
        for prompt in PROMPTS:
            b = measure_bekenstein(model, tok, prompt, device)
            bek_data.append(b)

        all_results[size] = {
            'scaling_exponent': round(float(np.mean([b['scaling_exponent'] for b in bek_data])), 4),
            'scaling_r2': round(float(np.mean([b['scaling_r2'] for b in bek_data])), 4),
            'mean_bekenstein_ratio': round(float(np.mean([b['mean_bekenstein_ratio'] for b in bek_data])), 6),
            'max_bekenstein_ratio': round(float(np.mean([b['max_bekenstein_ratio'] for b in bek_data])), 6),
            'bound_respected': float(np.mean([b['max_bekenstein_ratio'] for b in bek_data])) < 1.0,
        }
        print(f"  Scaling exp: {all_results[size]['scaling_exponent']:.4f}")
        print(f"  Max Bek ratio: {all_results[size]['max_bekenstein_ratio']:.6f}")
        print(f"  Bound: {'RESPECTED' if all_results[size]['bound_respected'] else 'VIOLATED'}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    sizes = list(all_results.keys())

    axes[0, 0].bar(sizes, [all_results[s]['scaling_exponent'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 0].axhline(1.0, color='gold', ls='--', lw=2, label='Volume law')
    axes[0, 0].axhline(0.5, color='green', ls='--', lw=2, label='Area law (1D)')
    axes[0, 0].set_ylabel('Scaling Exponent'); axes[0, 0].set_title('(a) S ~ k^alpha', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(sizes, [all_results[s]['max_bekenstein_ratio'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 1].axhline(1.0, color='red', ls='--', lw=2, label='Bound')
    axes[0, 1].set_ylabel('S / S_Bek'); axes[0, 1].set_title('(b) Bekenstein Ratio', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].axis('off'); axes[1, 0].axis('off'); axes[1, 1].axis('off')

    txt = "BEKENSTEIN BOUND\n\n"
    txt += "S <= 2*pi*R*sqrt(E)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  alpha = {d['scaling_exponent']:.3f}\n"
        txt += f"  S/S_B = {d['max_bekenstein_ratio']:.5f}\n"
        txt += f"  {'RESPECTED' if d['bound_respected'] else 'VIOLATED'}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 321: Bekenstein Bound -- Information-Area Law",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase321_bekenstein')
    plt.close()
    save_results('phase321_bekenstein', {'experiment': 'Bekenstein Bound', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
