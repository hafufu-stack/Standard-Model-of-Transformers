# -*- coding: utf-8 -*-
"""
Phase 345: Topological Edge States -- Bulk-Edge Correspondence
=====================================================
Topological systems have protected edge states at boundaries.
Test whether the Transformer has "edge states" at the first/last
layers that are topologically distinct from bulk layers.
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


def measure_edge_states(model, tok, prompt, device):
    """Test bulk-edge correspondence."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]

    # Measure "band gap" at each layer: gap in eigenvalue spectrum
    gaps = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        sorted_vals = torch.sort(h.abs())[0]
        # Find largest gap in spectrum
        diffs = sorted_vals[1:] - sorted_vals[:-1]
        max_gap = float(torch.max(diffs).item())
        avg_gap = float(torch.mean(diffs).item())
        gaps.append(round(float(max_gap / (avg_gap + 1e-10)), 4))

    # Edge detection: first and last layers vs bulk
    n_edge = 3  # Number of edge layers
    edge_layers = list(range(n_edge)) + list(range(n_layers + 1 - n_edge, n_layers + 1))
    bulk_layers = list(range(n_edge, n_layers + 1 - n_edge))

    edge_norms = [float(torch.norm(hiddens[li]).item()) for li in edge_layers]
    bulk_norms = [float(torch.norm(hiddens[li]).item()) for li in bulk_layers]

    edge_entropies = []
    bulk_entropies = []
    for li in range(n_layers + 1):
        p = torch.softmax(hiddens[li], dim=0)
        s = float(-torch.sum(p * torch.log(p + 1e-30)).item())
        if li in edge_layers:
            edge_entropies.append(s)
        else:
            bulk_entropies.append(s)

    # Edge localization: participation ratio difference
    edge_pr = []
    bulk_pr = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        p = (h**2) / (torch.sum(h**2) + 1e-30)
        pr = float(1.0 / (torch.sum(p**2).item() + 1e-30))
        if li in edge_layers:
            edge_pr.append(pr)
        else:
            bulk_pr.append(pr)

    # Conductance at boundaries: information flow rate
    conductances = []
    for li in range(n_layers):
        cos = float(torch.nn.functional.cosine_similarity(
            hiddens[li].unsqueeze(0), hiddens[li + 1].unsqueeze(0)).item())
        conductances.append(round(1 - cos, 6))

    edge_conductance = float(np.mean(conductances[:n_edge] + conductances[-n_edge:]))
    bulk_conductance = float(np.mean(conductances[n_edge:-n_edge]))

    return {
        'gap_profile': gaps,
        'edge_norm': round(float(np.mean(edge_norms)), 4),
        'bulk_norm': round(float(np.mean(bulk_norms)), 4),
        'edge_entropy': round(float(np.mean(edge_entropies)), 4),
        'bulk_entropy': round(float(np.mean(bulk_entropies)), 4),
        'edge_pr': round(float(np.mean(edge_pr)), 2),
        'bulk_pr': round(float(np.mean(bulk_pr)), 2),
        'edge_conductance': round(edge_conductance, 6),
        'bulk_conductance': round(bulk_conductance, 6),
        'conductance_profile': conductances,
        'edge_distinct': abs(float(np.mean(edge_entropies)) - float(np.mean(bulk_entropies))) > 0.1,
    }


def main():
    print("=" * 70)
    print("Phase 345: Topological Edge States")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        es_data = []
        for prompt in PROMPTS:
            e = measure_edge_states(model, tok, prompt, device)
            es_data.append(e)

        n = len(es_data[0]['gap_profile'])
        n_c = len(es_data[0]['conductance_profile'])
        all_results[size] = {
            'gap_profile': [round(float(np.mean([e['gap_profile'][i] for e in es_data])), 4)
                           for i in range(n)],
            'edge_norm': round(float(np.mean([e['edge_norm'] for e in es_data])), 4),
            'bulk_norm': round(float(np.mean([e['bulk_norm'] for e in es_data])), 4),
            'edge_entropy': round(float(np.mean([e['edge_entropy'] for e in es_data])), 4),
            'bulk_entropy': round(float(np.mean([e['bulk_entropy'] for e in es_data])), 4),
            'edge_pr': round(float(np.mean([e['edge_pr'] for e in es_data])), 2),
            'bulk_pr': round(float(np.mean([e['bulk_pr'] for e in es_data])), 2),
            'edge_conductance': round(float(np.mean([e['edge_conductance'] for e in es_data])), 6),
            'bulk_conductance': round(float(np.mean([e['bulk_conductance'] for e in es_data])), 6),
            'conductance_profile': [round(float(np.mean([e['conductance_profile'][i] for e in es_data])), 6)
                                   for i in range(n_c)],
            'edge_distinct': sum(1 for e in es_data if e['edge_distinct']) >= 4,
        }
        distinct = 'YES' if all_results[size]['edge_distinct'] else 'NO'
        print(f"  Edge S: {all_results[size]['edge_entropy']:.4f} vs Bulk S: {all_results[size]['bulk_entropy']:.4f}")
        print(f"  Edge G: {all_results[size]['edge_conductance']:.6f} vs Bulk G: {all_results[size]['bulk_conductance']:.6f}")
        print(f"  Edge distinct: {distinct}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['gap_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Gap ratio')
    axes[0, 0].set_title('(a) Spectral Gap', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['conductance_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('1-cos')
    axes[0, 1].set_title('(b) Conductance Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.25
    axes[0, 2].bar(x - w, [all_results[s]['edge_entropy'] for s in sizes], w, label='Edge S', color='#3498db')
    axes[0, 2].bar(x, [all_results[s]['bulk_entropy'] for s in sizes], w, label='Bulk S', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Edge vs Bulk Entropy', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(x - w/2, [all_results[s]['edge_pr'] for s in sizes], w, label='Edge PR', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['bulk_pr'] for s in sizes], w, label='Bulk PR', color='#e74c3c')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) Participation Ratio', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "EDGE STATES\n\n"
    for s in sizes:
        d = all_results[s]
        dist = 'YES' if d['edge_distinct'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  edge S = {d['edge_entropy']:.3f}\n"
        txt += f"  bulk S = {d['bulk_entropy']:.3f}\n"
        txt += f"  edge G = {d['edge_conductance']:.4f}\n"
        txt += f"  bulk G = {d['bulk_conductance']:.4f}\n"
        txt += f"  distinct: {dist}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 345: Topological Edge States", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase345_edge_states')
    plt.close()
    save_results('phase345_edge_states', {'experiment': 'Edge States', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
