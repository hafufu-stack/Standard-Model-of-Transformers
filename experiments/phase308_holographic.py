# -*- coding: utf-8 -*-
"""
Phase 308: Holographic Principle -- Boundary-Bulk Correspondence
================================================================
The holographic principle (AdS/CFT): information in a volume is encoded
on its boundary. For transformers:
- "Bulk" = internal hidden states (all layers, all positions)
- "Boundary" = first/last layer or first/last token
Test: can the boundary reconstruct the bulk?
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


def measure_holography(model, tok, prompt, device):
    """Test holographic principle: boundary encodes bulk."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Boundary = last layer, last token
    h_boundary = out.hidden_states[-1][0, -1, :].float()
    # Also test: boundary = first layer, last token
    h_boundary_0 = out.hidden_states[0][0, -1, :].float()

    # Bulk = all intermediate layers, last token
    bulk_states = [out.hidden_states[li][0, -1, :].float() for li in range(1, n_layers)]

    # Test 1: How much of each bulk state is encoded in the boundary?
    # Measure via cosine similarity
    cos_to_last = []
    cos_to_first = []
    for h_bulk in bulk_states:
        c_last = torch.nn.functional.cosine_similarity(
            h_bulk.unsqueeze(0), h_boundary.unsqueeze(0)).item()
        c_first = torch.nn.functional.cosine_similarity(
            h_bulk.unsqueeze(0), h_boundary_0.unsqueeze(0)).item()
        cos_to_last.append(c_last)
        cos_to_first.append(c_first)

    # Test 2: Mutual information proxy (shared singular value structure)
    # Compare SVD spectrum of boundary vs each bulk layer
    _, s_bound, _ = torch.linalg.svd(h_boundary.unsqueeze(0), full_matrices=False)

    # Test 3: Reconstruction quality
    # Can boundary (last layer) + linear map reconstruct each bulk layer?
    recon_r2 = []
    for h_bulk in bulk_states:
        # Simple: R2 of linear regression h_bulk = W @ h_boundary
        # Use cosine similarity as proxy for linear reconstructability
        r2 = torch.nn.functional.cosine_similarity(
            h_bulk.unsqueeze(0), h_boundary.unsqueeze(0)).item() ** 2
        recon_r2.append(r2)

    # Holographic entropy bound
    # S_bulk <= S_boundary (area bound)
    # Compute entropy of bulk vs boundary
    h_b_np = h_boundary.cpu().numpy()
    h_b_sq = h_b_np ** 2
    h_b_p = h_b_sq / (h_b_sq.sum() + 1e-15)
    S_boundary = float(-np.sum(h_b_p * np.log(h_b_p + 1e-15)))

    bulk_entropies = []
    for h_bulk in bulk_states:
        hb_np = h_bulk.cpu().numpy()
        hb_sq = hb_np ** 2
        hb_p = hb_sq / (hb_sq.sum() + 1e-15)
        S_bulk = float(-np.sum(hb_p * np.log(hb_p + 1e-15)))
        bulk_entropies.append(S_bulk)

    # Check if S_bulk <= S_boundary (holographic bound)
    violations = sum(1 for s in bulk_entropies if s > S_boundary)

    return {
        'cos_to_last': [round(c, 4) for c in cos_to_last],
        'cos_to_first': [round(c, 4) for c in cos_to_first],
        'recon_r2': [round(r, 4) for r in recon_r2],
        'S_boundary': round(S_boundary, 4),
        'bulk_entropies': [round(s, 4) for s in bulk_entropies],
        'bound_violations': violations,
        'mean_cos_last': round(float(np.mean(cos_to_last)), 4),
        'mean_cos_first': round(float(np.mean(cos_to_first)), 4),
        'mean_recon_r2': round(float(np.mean(recon_r2)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 308: Holographic Principle")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        holo_data = []
        for prompt in PROMPTS:
            h = measure_holography(model, tok, prompt, device)
            holo_data.append(h)

        n = len(holo_data[0]['cos_to_last'])
        avg_cos_last = [float(np.mean([h['cos_to_last'][i] for h in holo_data])) for i in range(n)]
        avg_cos_first = [float(np.mean([h['cos_to_first'][i] for h in holo_data])) for i in range(n)]
        avg_recon = [float(np.mean([h['recon_r2'][i] for h in holo_data])) for i in range(n)]

        avg_violations = float(np.mean([h['bound_violations'] for h in holo_data]))

        all_results[size] = {
            'n_bulk_layers': n,
            'avg_cos_to_last': [round(c, 4) for c in avg_cos_last],
            'avg_cos_to_first': [round(c, 4) for c in avg_cos_first],
            'avg_recon_r2': [round(r, 4) for r in avg_recon],
            'mean_cos_last': round(float(np.mean(avg_cos_last)), 4),
            'mean_cos_first': round(float(np.mean(avg_cos_first)), 4),
            'mean_recon': round(float(np.mean(avg_recon)), 4),
            'bound_violations': round(avg_violations, 2),
            'holographic': avg_violations < 3,
        }
        print(f"  Mean cos(bulk, boundary_last): {all_results[size]['mean_cos_last']:.4f}")
        print(f"  Mean cos(bulk, boundary_first): {all_results[size]['mean_cos_first']:.4f}")
        print(f"  Bound violations: {avg_violations:.1f}")
        print(f"  Holographic: {'YES' if all_results[size]['holographic'] else 'NO'}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Cosine to last layer
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_cos_to_last'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Bulk Layer')
    axes[0, 0].set_ylabel('cos(bulk, boundary_last)')
    axes[0, 0].set_title('(a) Similarity to Last Layer', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Cosine to first layer
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_cos_to_first'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Bulk Layer')
    axes[0, 1].set_ylabel('cos(bulk, boundary_first)')
    axes[0, 1].set_title('(b) Similarity to First Layer', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Reconstruction R2
    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_recon_r2'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Bulk Layer')
    axes[0, 2].set_ylabel('Reconstruction R2')
    axes[0, 2].set_title('(c) Boundary Reconstruction', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Boundary vs bulk entropy
    axes[1, 0].axis('off')

    # (e) Summary bars
    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.25
    axes[1, 1].bar(x - w, [all_results[s]['mean_cos_last'] for s in sizes], w,
                  label='cos(last)', color='#3498db')
    axes[1, 1].bar(x, [all_results[s]['mean_cos_first'] for s in sizes], w,
                  label='cos(first)', color='#e74c3c')
    axes[1, 1].bar(x + w, [all_results[s]['mean_recon'] for s in sizes], w,
                  label='recon R2', color='#2ecc71')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(sizes)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('(e) Holographic Metrics', fontweight='bold')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "HOLOGRAPHIC PRINCIPLE\n\n"
    txt += "Boundary encodes bulk?\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  cos(last): {d['mean_cos_last']:.3f}\n"
        txt += f"  cos(first): {d['mean_cos_first']:.3f}\n"
        txt += f"  violations: {d['bound_violations']:.0f}\n"
        txt += f"  {'HOLOGRAPHIC' if d['holographic'] else 'non-holographic'}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 308: Holographic Principle -- Boundary Encodes Bulk?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase308_holographic')
    plt.close()

    save_results('phase308_holographic', {
        'experiment': 'Holographic Principle',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
