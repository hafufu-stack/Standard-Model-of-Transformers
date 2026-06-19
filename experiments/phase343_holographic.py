# -*- coding: utf-8 -*-
"""
Phase 343: Holographic Duality -- Bulk-Boundary Correspondence
=====================================================
Test the holographic dictionary: boundary quantities (logits,
attention) correspond to bulk quantities (hidden states) via
a precise mapping. Measure the consistency of this mapping.
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


def measure_holographic_duality(model, tok, prompt, device):
    """Test bulk-boundary correspondence."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]

    # Boundary = final logits (the "boundary CFT")
    logits = out.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=0)
    boundary_entropy = float(-torch.sum(probs * torch.log(probs + 1e-30)).item())

    # Bulk = hidden states at each layer
    bulk_entropies = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        p = torch.softmax(h, dim=0)
        s = float(-torch.sum(p * torch.log(p + 1e-30)).item())
        bulk_entropies.append(s)

    # Test 1: Boundary entropy determined by bulk
    # S_boundary = f(S_bulk_final)
    r_bulk_boundary, _ = stats.pearsonr(
        bulk_entropies[1:],
        list(range(1, n_layers + 1))
    )

    # Test 2: Radial direction = RG flow
    # The "radial" coordinate in AdS is the layer index
    # Bulk fields at different radial positions should
    # reconstruct the boundary via HKLL kernel
    # Proxy: correlation between each bulk layer and boundary
    bulk_boundary_corrs = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Project bulk to boundary via linear map (lm_head)
        # Instead, use correlation
        cos = float(torch.nn.functional.cosine_similarity(
            h.unsqueeze(0)[:, :min(h.shape[0], logits.shape[0])],
            logits.unsqueeze(0)[:, :min(h.shape[0], logits.shape[0])]
        ).item())
        bulk_boundary_corrs.append(round(float(cos), 4))

    # Test 3: Holographic dictionary
    # In AdS/CFT: phi(z,x) = integral K(z,x;x') O(x') dx'
    # Test: h(l) = sum_{l'} K(l,l') h(l')
    # Reconstruction: can we reconstruct h(l) from boundary?
    h_final = hiddens[-1]
    reconstruction_fidelity = []
    for li in range(n_layers + 1):
        # How well does the final layer predict each layer?
        cos = float(torch.nn.functional.cosine_similarity(
            hiddens[li].unsqueeze(0), h_final.unsqueeze(0)).item())
        reconstruction_fidelity.append(round(float(cos), 4))

    # Test 4: Entanglement wedge
    # Information about subsystem A is encoded in the entanglement wedge
    dim = hiddens[0].shape[0]
    wedge_fracs = [0.1, 0.25, 0.5, 0.75, 0.9]
    wedge_info = []
    for frac in wedge_fracs:
        k = max(1, int(frac * dim))
        # Subsystem: first k dims of boundary
        h_sub = h_final[:k]
        h_full = h_final
        cos = float(torch.nn.functional.cosine_similarity(
            h_sub.unsqueeze(0), h_full[:k].unsqueeze(0)).item())
        wedge_info.append(round(float(cos), 4))

    return {
        'boundary_entropy': round(boundary_entropy, 4),
        'bulk_entropies': [round(s, 4) for s in bulk_entropies],
        'bulk_boundary_corrs': bulk_boundary_corrs,
        'reconstruction_fidelity': reconstruction_fidelity,
        'wedge_info': wedge_info,
        'max_corr_layer': int(np.argmax(bulk_boundary_corrs)),
        'avg_reconstruction': round(float(np.mean(reconstruction_fidelity)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 343: Holographic Duality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        hd_data = []
        for prompt in PROMPTS:
            h = measure_holographic_duality(model, tok, prompt, device)
            hd_data.append(h)

        n = len(hd_data[0]['bulk_boundary_corrs'])
        n_be = len(hd_data[0]['bulk_entropies'])
        all_results[size] = {
            'boundary_entropy': round(float(np.mean([h['boundary_entropy'] for h in hd_data])), 4),
            'bulk_boundary_corrs': [round(float(np.mean([h['bulk_boundary_corrs'][i] for h in hd_data])), 4)
                                   for i in range(n)],
            'reconstruction_fidelity': [round(float(np.mean([h['reconstruction_fidelity'][i] for h in hd_data])), 4)
                                       for i in range(n)],
            'bulk_entropies': [round(float(np.mean([h['bulk_entropies'][i] for h in hd_data])), 4)
                              for i in range(n_be)],
            'max_corr_layer': int(np.median([h['max_corr_layer'] for h in hd_data])),
            'avg_reconstruction': round(float(np.mean([h['avg_reconstruction'] for h in hd_data])), 4),
            'wedge_info': [round(float(np.mean([h['wedge_info'][i] for h in hd_data])), 4)
                          for i in range(len(hd_data[0]['wedge_info']))],
        }
        print(f"  Boundary S: {all_results[size]['boundary_entropy']:.4f}")
        print(f"  Max corr layer: {all_results[size]['max_corr_layer']}")
        print(f"  Avg reconstruction: {all_results[size]['avg_reconstruction']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['bulk_boundary_corrs'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('(a) Bulk-Boundary Correlation', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['reconstruction_fidelity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Fidelity')
    axes[0, 1].set_title('(b) Boundary Reconstruction', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['bulk_entropies'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('(c) Bulk Entropy Profile', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    fracs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for size, data in all_results.items():
        axes[1, 0].plot(fracs, data['wedge_info'], '-o', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Subsystem fraction'); axes[1, 0].set_ylabel('Wedge fidelity')
    axes[1, 0].set_title('(d) Entanglement Wedge', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    sizes = list(all_results.keys())
    txt = "HOLOGRAPHIC DUALITY\n\n"
    txt += "Bulk = hidden states\n"
    txt += "Boundary = logits/probs\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  S_bdy = {d['boundary_entropy']:.3f}\n"
        txt += f"  max_corr = L{d['max_corr_layer']}\n"
        txt += f"  recon = {d['avg_reconstruction']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 343: Holographic Duality", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase343_holographic')
    plt.close()
    save_results('phase343_holographic', {'experiment': 'Holographic Duality', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
