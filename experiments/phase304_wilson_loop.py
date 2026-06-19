# -*- coding: utf-8 -*-
"""
Phase 304: Wilson Loop -- Gauge-Invariant Observables
======================================================
Wilson loops W(C) = tr[P exp(i * integral A_mu dx^mu)]
In the transformer context:
- The "gauge field" A = attention weights
- A "loop" = tracing information flow around a cycle of layers
- Wilson loop value determines confinement vs deconfinement
W(C) ~ exp(-Area): confinement (info trapped)
W(C) ~ exp(-Perimeter): deconfinement (info free)
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


def compute_wilson_loops(model, tok, prompt, device):
    """Compute Wilson loop analogue for transformer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Transport matrix: T(l1, l2) = h(l2)^T @ h(l1) / (|h(l2)||h(l1)|)
    # This is the cosine similarity matrix between all layer pairs
    h_list = [out.hidden_states[li][0, -1, :].float() for li in range(n_layers + 1)]

    # Wilson loop of size R (layer separation) x L (position separation)
    # Simplified: loop around layers only
    loop_values = {}
    for R in range(1, min(8, n_layers)):
        loops = []
        for start in range(n_layers - R):
            # Parallel transport: product of cosine similarities along the loop
            # Forward path: start -> start+1 -> ... -> start+R
            forward = 1.0
            for i in range(start, start + R):
                cos = torch.nn.functional.cosine_similarity(
                    h_list[i].unsqueeze(0), h_list[i+1].unsqueeze(0)).item()
                forward *= cos

            # Return: direct similarity start <-> start+R
            direct = torch.nn.functional.cosine_similarity(
                h_list[start].unsqueeze(0), h_list[start + R].unsqueeze(0)).item()

            # Wilson loop = (forward transport) vs (direct transport)
            # If loop is trivial: forward ≈ direct
            # If confinement: forward << direct (info lost in transport)
            W = forward / (direct + 1e-10)
            loops.append(W)

        loop_values[R] = float(np.mean(loops))

    # Test area law vs perimeter law
    # Area law: log(W) ~ -sigma * R (linear in R = "area" in 1+1D)
    # Perimeter law: log(W) ~ -mu * R^0 = constant offset
    R_arr = np.array(list(loop_values.keys()), dtype=float)
    log_W = np.array([np.log(abs(v) + 1e-15) for v in loop_values.values()])

    slope, intercept, r, p, se = stats.linregress(R_arr, log_W)
    string_tension = -slope  # sigma = string tension

    # Confinement if sigma > 0 (area law)
    confinement = string_tension > 0.01

    return {
        'loop_values': {str(k): round(v, 6) for k, v in loop_values.items()},
        'string_tension': round(float(string_tension), 4),
        'R2_area_law': round(float(r**2), 4),
        'confinement': confinement,
    }


def main():
    print("=" * 70)
    print("Phase 304: Wilson Loop -- Gauge-Invariant Observables")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        wilson_data = []
        for prompt in PROMPTS:
            w = compute_wilson_loops(model, tok, prompt, device)
            wilson_data.append(w)

        avg_tension = float(np.mean([w['string_tension'] for w in wilson_data]))
        avg_r2 = float(np.mean([w['R2_area_law'] for w in wilson_data]))

        # Average loop values
        all_R = sorted(wilson_data[0]['loop_values'].keys())
        avg_loops = {r: float(np.mean([w['loop_values'].get(r, 0) for w in wilson_data])) for r in all_R}

        all_results[size] = {
            'avg_string_tension': round(avg_tension, 4),
            'avg_R2_area_law': round(avg_r2, 4),
            'confinement': avg_tension > 0.01,
            'avg_loop_values': {k: round(v, 6) for k, v in avg_loops.items()},
        }
        regime = 'Confinement (area law)' if avg_tension > 0.01 else 'Deconfinement (perimeter)'
        print(f"  String tension sigma = {avg_tension:.4f}")
        print(f"  Area law R2 = {avg_r2:.4f}")
        print(f"  Regime: {regime}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Wilson loop vs R
    for size, data in all_results.items():
        Rs = [int(r) for r in sorted(data['avg_loop_values'].keys())]
        Ws = [data['avg_loop_values'][str(r)] for r in Rs]
        axes[0, 0].plot(Rs, Ws, 'o-', color=colors[size], lw=2, markersize=8, label=size)
    axes[0, 0].set_xlabel('Loop Size R')
    axes[0, 0].set_ylabel('Wilson Loop W(R)')
    axes[0, 0].set_title('(a) Wilson Loop', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) log(W) vs R (area law test)
    for size, data in all_results.items():
        Rs = [int(r) for r in sorted(data['avg_loop_values'].keys())]
        Ws = [data['avg_loop_values'][str(r)] for r in Rs]
        log_Ws = [np.log(abs(w) + 1e-15) for w in Ws]
        axes[0, 1].plot(Rs, log_Ws, 'o-', color=colors[size], lw=2, markersize=8,
                       label=f"{size} (sigma={data['avg_string_tension']:.3f})")
    axes[0, 1].set_xlabel('Loop Size R')
    axes[0, 1].set_ylabel('log W(R)')
    axes[0, 1].set_title('(b) Area Law Test', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) String tension
    sizes = list(all_results.keys())
    tensions = [all_results[s]['avg_string_tension'] for s in sizes]
    bars = axes[0, 2].bar(sizes, tensions, color=[colors[s] for s in sizes])
    axes[0, 2].axhline(0, color='gold', ls='--', lw=2, label='Deconfinement (sigma=0)')
    axes[0, 2].set_ylabel('String Tension sigma')
    axes[0, 2].set_title('(c) String Tension', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d-e) empty for now
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    # (f) Summary
    txt = "WILSON LOOP ANALYSIS\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  sigma = {d['avg_string_tension']:.4f}\n"
        txt += f"  R2 = {d['avg_R2_area_law']:.4f}\n"
        txt += f"  {'CONFINEMENT' if d['confinement'] else 'DECONFINEMENT'}\n\n"
    txt += "Area law: info confined\n"
    txt += "Perimeter law: info free"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 304: Wilson Loop -- Confinement in Transformer",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase304_wilson_loop')
    plt.close()

    save_results('phase304_wilson_loop', {
        'experiment': 'Wilson Loop - Gauge Invariant Observables',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
