# -*- coding: utf-8 -*-
"""
Phase 352: T-duality -- Radius Inversion Symmetry
=====================================================
In string theory, T-duality maps a string on a circle of radius R
to a string on radius alpha'/R. Test whether the Transformer has
an analogous duality: small and large scales are equivalent.
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


def measure_tduality(model, tok, prompt, device):
    """Test T-duality in hidden state space."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Scale inversion: compare h and 1/h (element-wise)
    # T-duality: physics at scale R is same as at alpha'/R
    inversion_corrs = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        h_inv = 1.0 / (h + 1e-10 * torch.sign(h + 1e-30))
        # Normalize both
        h_norm = h / (torch.norm(h) + 1e-10)
        h_inv_norm = h_inv / (torch.norm(h_inv) + 1e-10)
        cos = float(torch.nn.functional.cosine_similarity(
            h_norm.unsqueeze(0), h_inv_norm.unsqueeze(0)).item())
        inversion_corrs.append(round(float(cos), 4))

    # 2. Mirror symmetry: early layers ↔ late layers
    mirror_corrs = []
    for li in range(n_layers // 2 + 1):
        mirror_li = n_layers - li
        cos = float(torch.nn.functional.cosine_similarity(
            hiddens[li].unsqueeze(0), hiddens[mirror_li].unsqueeze(0)).item())
        mirror_corrs.append(round(float(cos), 4))

    # 3. Winding modes: large-scale structure vs momentum modes
    # Winding ~ low-frequency components, Momentum ~ high-frequency
    winding_momentum_corrs = []
    for li in range(n_layers + 1):
        h = hiddens[li].numpy()
        fft = np.fft.rfft(h)
        n_freq = len(fft)
        half = n_freq // 2

        # Low freq (winding) vs high freq (momentum)
        low_energy = float(np.sum(np.abs(fft[:half])**2))
        high_energy = float(np.sum(np.abs(fft[half:])**2))
        ratio = low_energy / (high_energy + 1e-10)
        winding_momentum_corrs.append(round(float(ratio), 4))

    # 4. Self-dual point: is there a layer where R = alpha'/R?
    # At self-dual point, winding = momentum
    self_dual_layer = 0
    min_diff = float('inf')
    for li in range(n_layers + 1):
        h = hiddens[li].numpy()
        fft = np.fft.rfft(h)
        n_freq = len(fft)
        half = n_freq // 2
        low = float(np.sum(np.abs(fft[:half])**2))
        high = float(np.sum(np.abs(fft[half:])**2))
        diff = abs(low - high) / (low + high + 1e-10)
        if diff < min_diff:
            min_diff = diff
            self_dual_layer = li

    return {
        'inversion_corrs': inversion_corrs,
        'mirror_corrs': mirror_corrs,
        'winding_momentum': winding_momentum_corrs,
        'self_dual_layer': self_dual_layer,
        'avg_inversion': round(float(np.mean(inversion_corrs)), 4),
        'avg_mirror': round(float(np.mean(mirror_corrs)), 4),
        'tduality_present': abs(float(np.mean(inversion_corrs))) > 0.01 or float(np.mean(mirror_corrs)) > 0.3,
    }


def main():
    print("=" * 70)
    print("Phase 352: T-duality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        td_data = []
        for prompt in PROMPTS:
            t = measure_tduality(model, tok, prompt, device)
            td_data.append(t)

        n = len(td_data[0]['inversion_corrs'])
        n_m = len(td_data[0]['mirror_corrs'])
        n_w = len(td_data[0]['winding_momentum'])
        all_results[size] = {
            'inversion_corrs': [round(float(np.mean([t['inversion_corrs'][i] for t in td_data])), 4)
                               for i in range(n)],
            'mirror_corrs': [round(float(np.mean([t['mirror_corrs'][i] for t in td_data])), 4)
                            for i in range(n_m)],
            'winding_momentum': [round(float(np.mean([t['winding_momentum'][i] for t in td_data])), 4)
                                for i in range(n_w)],
            'self_dual_layer': int(np.median([t['self_dual_layer'] for t in td_data])),
            'avg_inversion': round(float(np.mean([t['avg_inversion'] for t in td_data])), 4),
            'avg_mirror': round(float(np.mean([t['avg_mirror'] for t in td_data])), 4),
        }
        print(f"  Avg inversion: {all_results[size]['avg_inversion']:.4f}")
        print(f"  Avg mirror: {all_results[size]['avg_mirror']:.4f}")
        print(f"  Self-dual layer: {all_results[size]['self_dual_layer']}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['inversion_corrs'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('cos(h, 1/h)')
    axes[0, 0].set_title('(a) Inversion Symmetry', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['mirror_corrs'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer pair'); axes[0, 1].set_ylabel('Mirror correlation')
    axes[0, 1].set_title('(b) Mirror Symmetry', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['winding_momentum'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Winding/Momentum')
    axes[0, 2].set_title('(c) Winding vs Momentum', fontweight='bold')
    axes[0, 2].axhline(1.0, color='gold', ls='--', label='Self-dual')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['self_dual_layer'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_title('(d) Self-Dual Layer', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "T-DUALITY\n\n"
    txt += "R <-> alpha'/R\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  inv = {d['avg_inversion']:.4f}\n"
        txt += f"  mirror = {d['avg_mirror']:.4f}\n"
        txt += f"  self-dual = L{d['self_dual_layer']}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 352: T-Duality", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase352_tduality')
    plt.close()
    save_results('phase352_tduality', {'experiment': 'T-Duality', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
