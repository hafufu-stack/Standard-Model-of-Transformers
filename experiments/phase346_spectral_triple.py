# -*- coding: utf-8 -*-
"""
Phase 346: Non-Commutative Geometry -- Spectral Triple
=====================================================
In Connes' non-commutative geometry, the spectral triple
(A, H, D) defines geometry from algebra. A = algebra of functions,
H = Hilbert space, D = Dirac operator. Test whether the Transformer
has a well-defined spectral triple.
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


def measure_spectral_triple(model, tok, prompt, device):
    """Measure spectral triple properties."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # Dirac operator D: approximate as the "difference operator" between layers
    # D|h(l)> = h(l+1) - h(l) (discrete derivative)
    dirac_eigenvalues = []
    for li in range(n_layers):
        Dh = hiddens[li + 1] - hiddens[li]
        # Eigenvalues of D^2 (squared Dirac)
        D2 = torch.dot(Dh, Dh)
        dirac_eigenvalues.append(float(D2.item()))

    # Spectral dimension from Dirac operator
    # d_s = -2 * d(log K(t)) / d(log t) where K(t) = Tr(exp(-t*D^2))
    # Use heat kernel trace at different "times"
    D2_arr = np.array(dirac_eigenvalues)
    t_values = np.logspace(-2, 1, 20)
    heat_traces = []
    for t in t_values:
        K = float(np.sum(np.exp(-t * D2_arr)))
        heat_traces.append(max(K, 1e-30))

    log_t = np.log(t_values)
    log_K = np.log(np.array(heat_traces))
    if len(log_t) > 5:
        # Use central region for best fit
        mid = len(log_t) // 3
        slope, _, r, _, _ = stats.linregress(log_t[mid:2*mid], log_K[mid:2*mid])
        spectral_dim = -2 * slope
    else:
        spectral_dim = 0

    # Connes distance: d(h1, h2) = sup{|f(h1) - f(h2)| : ||[D,f]|| <= 1}
    # Approximate as geodesic distance
    connes_distances = []
    for li in range(n_layers):
        dist = float(torch.norm(hiddens[li + 1] - hiddens[li]).item())
        connes_distances.append(round(dist, 4))

    # KO dimension: from reality operator J
    # J is complex conjugation in the simplest case
    # Test: does J commute or anti-commute with D?
    # Use sign structure of Dirac eigenvalues
    positive = sum(1 for d in dirac_eigenvalues if d > float(np.median(dirac_eigenvalues)))
    negative = len(dirac_eigenvalues) - positive
    ko_signature = abs(positive - negative) / len(dirac_eigenvalues)

    # Spectral action: S = Tr(f(D/Lambda))
    # Count eigenvalues below cutoff
    cutoff = float(np.median(D2_arr))
    spectral_action = sum(1 for d in D2_arr if d < cutoff) / len(D2_arr)

    return {
        'dirac_spectrum': [round(d, 4) for d in dirac_eigenvalues],
        'spectral_dim': round(float(spectral_dim), 4),
        'connes_distances': connes_distances,
        'avg_distance': round(float(np.mean(connes_distances)), 4),
        'ko_signature': round(float(ko_signature), 4),
        'spectral_action': round(float(spectral_action), 4),
        'heat_trace_r2': round(float(r**2 if 'r' in dir() else 0), 4),
    }


def main():
    print("=" * 70)
    print("Phase 346: Non-Commutative Geometry")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        st_data = []
        for prompt in PROMPTS:
            s = measure_spectral_triple(model, tok, prompt, device)
            st_data.append(s)

        n = len(st_data[0]['dirac_spectrum'])
        n_cd = len(st_data[0]['connes_distances'])
        all_results[size] = {
            'dirac_spectrum': [round(float(np.mean([s['dirac_spectrum'][i] for s in st_data])), 4)
                              for i in range(n)],
            'spectral_dim': round(float(np.mean([s['spectral_dim'] for s in st_data])), 4),
            'connes_distances': [round(float(np.mean([s['connes_distances'][i] for s in st_data])), 4)
                                for i in range(n_cd)],
            'avg_distance': round(float(np.mean([s['avg_distance'] for s in st_data])), 4),
            'ko_signature': round(float(np.mean([s['ko_signature'] for s in st_data])), 4),
            'spectral_action': round(float(np.mean([s['spectral_action'] for s in st_data])), 4),
        }
        print(f"  Spectral dim: {all_results[size]['spectral_dim']:.4f}")
        print(f"  Avg Connes dist: {all_results[size]['avg_distance']:.4f}")
        print(f"  KO signature: {all_results[size]['ko_signature']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['dirac_spectrum'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('D^2 eigenvalue')
    axes[0, 0].set_title('(a) Dirac Spectrum', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['connes_distances'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Connes distance')
    axes[0, 1].set_title('(b) Connes Distances', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['spectral_dim'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('d_s')
    axes[0, 2].set_title('(c) Spectral Dimension', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['ko_signature'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_title('(d) KO Signature', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "SPECTRAL TRIPLE (A, H, D)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  d_s = {d['spectral_dim']:.2f}\n"
        txt += f"  d_C = {d['avg_distance']:.2f}\n"
        txt += f"  KO = {d['ko_signature']:.3f}\n"
        txt += f"  S_sp = {d['spectral_action']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 346: Non-Commutative Geometry", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase346_spectral_triple')
    plt.close()
    save_results('phase346_spectral_triple', {'experiment': 'Spectral Triple', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
