# -*- coding: utf-8 -*-
"""
Phase 344: Topological Invariants -- Chern Number
=====================================================
In condensed matter physics, the Chern number classifies
topological phases. The Berry curvature F = dA integrated over
the Brillouin zone gives a quantized integer invariant.
Test whether the Transformer's layer manifold has non-trivial
Chern numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_chern(model, tok, prompt, device):
    """Measure Chern number from Berry curvature on 2D slices."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # Berry phase: gamma = oint <n(k)|d/dk|n(k)> dk
    # Approximate over layer-pairs as a discrete "Brillouin zone"
    # Berry connection A_l = Im(<h(l)|h(l+1)>) / |<h(l)|h(l+1)>|
    berry_connections = []
    for li in range(n_layers):
        overlap = torch.dot(hiddens[li], hiddens[li + 1])
        A = float(torch.atan2(torch.tensor(0.0), overlap).item())  # Phase
        # Better: use explicit phase from complex inner product
        # Treat even/odd dims as real/imag parts
        half = min(dim // 2, 256)
        z1 = hiddens[li][:half] + 1j * hiddens[li][half:2*half]
        z2 = hiddens[li + 1][:half] + 1j * hiddens[li + 1][half:2*half]
        inner = np.sum((z1.numpy().conj() * z2.numpy()))
        phase = float(np.angle(inner))
        berry_connections.append(phase)

    # Berry phase = sum of all connections
    berry_phase = float(np.sum(berry_connections))
    # Chern number = berry_phase / (2*pi), should be integer
    chern_raw = berry_phase / (2 * np.pi)
    chern_number = round(chern_raw)
    chern_frac = abs(chern_raw - chern_number)  # Fractional part

    # Berry curvature at each layer: F(l) = A(l+1) - A(l)
    berry_curvature = []
    for li in range(len(berry_connections) - 1):
        F = berry_connections[li + 1] - berry_connections[li]
        # Wrap to [-pi, pi]
        F = float((F + np.pi) % (2 * np.pi) - np.pi)
        berry_curvature.append(round(F, 6))

    # Winding number: alternative topological invariant
    # Count how many times the phase winds around 2*pi
    cumulative_phase = np.cumsum(berry_connections)
    winding = float(cumulative_phase[-1] / (2 * np.pi)) if len(cumulative_phase) > 0 else 0

    return {
        'berry_phase': round(float(berry_phase), 4),
        'chern_raw': round(float(chern_raw), 4),
        'chern_number': chern_number,
        'chern_frac': round(float(chern_frac), 4),
        'berry_curvature': berry_curvature,
        'berry_connections': [round(b, 4) for b in berry_connections],
        'winding_number': round(float(winding), 4),
        'quantized': chern_frac < 0.15,
    }


def main():
    print("=" * 70)
    print("Phase 344: Topological Invariants - Chern Number")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        chern_data = []
        for prompt in PROMPTS:
            c = measure_chern(model, tok, prompt, device)
            chern_data.append(c)

        n_bc = len(chern_data[0]['berry_curvature'])
        all_results[size] = {
            'avg_berry_phase': round(float(np.mean([c['berry_phase'] for c in chern_data])), 4),
            'avg_chern_raw': round(float(np.mean([c['chern_raw'] for c in chern_data])), 4),
            'chern_number': int(np.median([c['chern_number'] for c in chern_data])),
            'avg_chern_frac': round(float(np.mean([c['chern_frac'] for c in chern_data])), 4),
            'berry_curvature': [round(float(np.mean([c['berry_curvature'][i] for c in chern_data])), 6)
                               for i in range(n_bc)],
            'winding_number': round(float(np.mean([c['winding_number'] for c in chern_data])), 4),
            'quantized': sum(1 for c in chern_data if c['quantized']) >= 4,
        }
        q = 'YES' if all_results[size]['quantized'] else 'NO'
        print(f"  Berry phase: {all_results[size]['avg_berry_phase']:.4f}")
        print(f"  Chern: {all_results[size]['chern_number']} (frac={all_results[size]['avg_chern_frac']:.4f})")
        print(f"  Winding: {all_results[size]['winding_number']:.4f}")
        print(f"  Quantized: {q}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['berry_curvature'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('F (Berry curvature)')
    axes[0, 0].set_title('(a) Berry Curvature', fontweight='bold')
    axes[0, 0].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 1].bar(x, [all_results[s]['chern_number'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(sizes)
    axes[0, 1].set_ylabel('Chern number')
    axes[0, 1].set_title('(b) Chern Number', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(x, [all_results[s]['winding_number'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_ylabel('Winding number')
    axes[0, 2].set_title('(c) Winding Number', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(x, [all_results[s]['avg_chern_frac'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].axhline(0.15, color='gold', ls='--', lw=2, label='Quantization threshold')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) Quantization Error', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "TOPOLOGICAL INVARIANTS\n\n"
    txt += "Chern = (1/2pi) * oint F dk\n\n"
    for s in sizes:
        d = all_results[s]
        q = 'YES' if d['quantized'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  gamma = {d['avg_berry_phase']:.3f}\n"
        txt += f"  C = {d['chern_number']}\n"
        txt += f"  frac = {d['avg_chern_frac']:.3f}\n"
        txt += f"  wind = {d['winding_number']:.3f}\n"
        txt += f"  quant: {q}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 344: Topological Invariants - Chern Number", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase344_chern')
    plt.close()
    save_results('phase344_chern', {'experiment': 'Chern Number', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
