# -*- coding: utf-8 -*-
"""
Phase 305: Conformal Bootstrap -- Operator Product Expansion
==============================================================
The conformal bootstrap constrains CFTs by consistency of the
Operator Product Expansion (OPE). For two operators O1, O2:
  O1(x) * O2(0) = sum_k C_12k |x|^(Delta_k - Delta_1 - Delta_2) O_k(0)
Test if transformer hidden states satisfy OPE-like structure.
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


def measure_ope(model, tok, prompt, device):
    """Test operator product expansion in hidden states."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    seq_len = out.hidden_states[0].shape[1]

    # For each layer, test if h(l+2) can be predicted from h(l) and h(l+1)
    # OPE: h(l+2) ~ sum_k C_k * f(h(l), h(l+1))
    # Simplified: h(l+2) ~ alpha * h(l) + beta * h(l+1) + gamma * (h(l) * h(l+1))
    ope_r2_list = []
    ope_coeffs = []

    for li in range(n_layers - 1):
        h0 = out.hidden_states[li][0, -1, :].float()
        h1 = out.hidden_states[li + 1][0, -1, :].float()
        h2 = out.hidden_states[li + 2][0, -1, :].float()

        # Build basis: [h0, h1, h0*h1 (element-wise)]
        h01 = h0 * h1  # element-wise product
        basis = torch.stack([h0, h1, h01], dim=0)  # (3, D)

        # Least squares: h2 = basis^T @ coeffs
        # (D,) = (3, D)^T @ (3,) => solve in least-squares sense
        coeffs, residuals, _, _ = torch.linalg.lstsq(basis.T, h2.unsqueeze(1))
        h2_pred = (basis.T @ coeffs).squeeze()

        # R2
        ss_res = ((h2 - h2_pred) ** 2).sum().item()
        ss_tot = ((h2 - h2.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        ope_r2_list.append(float(r2))
        ope_coeffs.append([float(c) for c in coeffs.squeeze().cpu()])

    # Scaling dimensions from OPE
    # Delta = -log(|coeff|) / log(layer_separation)
    avg_coeffs = np.mean(ope_coeffs, axis=0) if ope_coeffs else [0, 0, 0]
    deltas = [-np.log(abs(c) + 1e-10) for c in avg_coeffs]

    return {
        'ope_r2': [round(r, 4) for r in ope_r2_list],
        'mean_ope_r2': round(float(np.mean(ope_r2_list)), 4),
        'ope_coeffs_alpha': round(float(avg_coeffs[0]), 4),
        'ope_coeffs_beta': round(float(avg_coeffs[1]), 4),
        'ope_coeffs_gamma': round(float(avg_coeffs[2]), 4),
        'scaling_dims': [round(d, 4) for d in deltas],
    }


def main():
    print("=" * 70)
    print("Phase 305: Conformal Bootstrap -- OPE")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ope_data = []
        for prompt in PROMPTS:
            o = measure_ope(model, tok, prompt, device)
            ope_data.append(o)

        n = len(ope_data[0]['ope_r2'])
        avg_r2 = [float(np.mean([o['ope_r2'][i] for o in ope_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n + 1,
            'avg_ope_r2': [round(r, 4) for r in avg_r2],
            'mean_ope_r2': round(float(np.mean(avg_r2)), 4),
            'max_ope_r2': round(float(np.max(avg_r2)), 4),
            'max_ope_layer': int(np.argmax(avg_r2)),
            'avg_alpha': round(float(np.mean([o['ope_coeffs_alpha'] for o in ope_data])), 4),
            'avg_beta': round(float(np.mean([o['ope_coeffs_beta'] for o in ope_data])), 4),
            'avg_gamma': round(float(np.mean([o['ope_coeffs_gamma'] for o in ope_data])), 6),
        }
        print(f"  Mean OPE R2 = {all_results[size]['mean_ope_r2']:.4f}")
        print(f"  Max OPE R2 = {all_results[size]['max_ope_r2']:.4f} at L{all_results[size]['max_ope_layer']}")
        print(f"  Coeffs: a={all_results[size]['avg_alpha']:.3f}, b={all_results[size]['avg_beta']:.3f}, g={all_results[size]['avg_gamma']:.6f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) OPE R2 profile
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_ope_r2'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('OPE R2')
    axes[0, 0].set_title('(a) OPE Reconstruction Quality', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Normalized depth
    for size, data in all_results.items():
        n = len(data['avg_ope_r2'])
        x = np.linspace(0, 1, n)
        axes[0, 1].plot(x, data['avg_ope_r2'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('OPE R2')
    axes[0, 1].set_title('(b) OPE R2 vs Depth', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) OPE coefficients
    sizes = list(all_results.keys())
    x = np.arange(3)
    w = 0.35
    for i, size in enumerate(sizes):
        d = all_results[size]
        vals = [d['avg_alpha'], d['avg_beta'], d['avg_gamma'] * 1000]  # scale gamma
        axes[0, 2].bar(x + i*w - w/2, vals, w, label=size, color=colors[size])
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(['alpha', 'beta', 'gamma x1000'])
    axes[0, 2].set_ylabel('Coefficient')
    axes[0, 2].set_title('(c) OPE Coefficients', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Mean R2 comparison
    axes[1, 0].bar(sizes, [all_results[s]['mean_ope_r2'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Mean OPE R2')
    axes[1, 0].set_title('(d) Mean OPE Quality', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e-f) empty
    axes[1, 1].axis('off')

    # (f) Summary
    txt = "CONFORMAL BOOTSTRAP\n\n"
    txt += "OPE: h(l+2) ~ a*h(l) + b*h(l+1)\n"
    txt += "          + g*(h(l)*h(l+1))\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}: R2={d['mean_ope_r2']:.3f}\n"
        txt += f"  a={d['avg_alpha']:.2f} b={d['avg_beta']:.2f}\n\n"
    txt += "High R2 -> OPE-like structure"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 305: Conformal Bootstrap -- Operator Product Expansion",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase305_ope')
    plt.close()

    save_results('phase305_ope', {
        'experiment': 'Conformal Bootstrap - OPE',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
