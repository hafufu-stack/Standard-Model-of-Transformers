# -*- coding: utf-8 -*-
"""
Phase 24: The Holographic Principle (Opus Original)
=====================================================
In black hole physics, all bulk information is encoded on the boundary.
Test if the final layer's hidden state (boundary) encodes ALL information
from intermediate layers (bulk) -- measuring information compression.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 24: The Holographic Principle")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The theory of general relativity describes gravity as curved spacetime",
        "Proteins fold into three dimensional structures determined by amino acid",
        "The stock market reflects collective decisions of millions of investors",
        "Photosynthesis converts sunlight into chemical energy in plant cells",
        "The history of civilization spans thousands of years of human progress",
        "Artificial intelligence systems process information through neural layers",
    ]

    all_cos_boundary = []
    all_projection = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Extract hidden states at last token position
        hs = [out.hidden_states[l][0, -1, :].float().cpu() for l in range(len(out.hidden_states))]
        boundary = hs[-1]  # "boundary" = final layer
        boundary_norm = boundary.norm()

        # Cosine similarity between each interior layer and boundary
        cos_boundary = []
        for l in range(len(hs) - 1):
            h = hs[l]
            h_norm = h.norm()
            if h_norm > 1e-6 and boundary_norm > 1e-6:
                cos = torch.dot(h, boundary) / (h_norm * boundary_norm)
                cos_boundary.append(cos.item())
            else:
                cos_boundary.append(0)
        all_cos_boundary.append(cos_boundary)

        # Projection: how much of each layer's information is preserved
        # in the boundary? |proj(h_l onto boundary)| / |h_l|
        proj_frac = []
        boundary_unit = boundary / (boundary_norm + 1e-10)
        for l in range(len(hs) - 1):
            h = hs[l]
            proj = torch.dot(h, boundary_unit).item()
            frac = abs(proj) / (h.norm().item() + 1e-10)
            proj_frac.append(frac)
        all_projection.append(proj_frac)

    # Average
    min_len = min(len(c) for c in all_cos_boundary)
    avg_cos = np.mean([c[:min_len] for c in all_cos_boundary], axis=0)
    avg_proj = np.mean([p[:min_len] for p in all_projection], axis=0)

    print("\n--- Holographic Encoding ---")
    for l in range(0, min_len, 4):
        print(f"  Layer {l}: cos(h_l, boundary)={avg_cos[l]:.4f}, "
              f"projection frac={avg_proj[l]:.4f}")

    # Information uniqueness: how much of each layer is orthogonal to boundary?
    orthogonal_frac = 1 - avg_proj

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(range(min_len), avg_cos, 'o-', color='#e74c3c', ms=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity to Boundary')
    ax.set_title('(a) Alignment with Boundary (Final Layer)')
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.3, label='Perfect alignment')
    ax.legend()

    ax = axes[1]
    ax.fill_between(range(min_len), avg_proj, alpha=0.3, color='#3498db',
                    label='Preserved in boundary')
    ax.fill_between(range(min_len), avg_proj, 1.0, alpha=0.3, color='#e74c3c',
                    label='Lost (orthogonal)')
    ax.plot(range(min_len), avg_proj, 'o-', color='#3498db', ms=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction Preserved')
    ax.set_title('(b) Information Preservation')
    ax.legend(fontsize=8)

    ax = axes[2]
    # Information loss rate
    info_loss = np.gradient(orthogonal_frac)
    ax.plot(range(min_len), info_loss, 'o-', color='#9b59b6', ms=3)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('d(lost info)/dLayer')
    ax.set_title('(c) Information Loss Rate')

    # How much of the bulk is on the boundary?
    bulk_on_boundary = np.mean(avg_proj)
    fig.suptitle(
        f"Phase 24: Holographic Principle\n"
        f"Mean boundary encoding = {bulk_on_boundary:.3f} "
        f"({bulk_on_boundary*100:.0f}% of bulk info preserved on boundary)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase24_holographic_principle")
    plt.close()

    if bulk_on_boundary > 0.8:
        verdict = (f"STRONG HOLOGRAPHY: {bulk_on_boundary*100:.0f}% preserved. "
                   f"The final layer encodes nearly all intermediate information!")
    elif bulk_on_boundary > 0.5:
        verdict = (f"PARTIAL HOLOGRAPHY: {bulk_on_boundary*100:.0f}% preserved. "
                   f"The boundary captures most but not all bulk info.")
    else:
        verdict = (f"WEAK HOLOGRAPHY: {bulk_on_boundary*100:.0f}% preserved. "
                   f"Each layer contains unique information not in the boundary.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 24: Holographic Principle',
        'summary': {'verdict': verdict, 'bulk_on_boundary': float(bulk_on_boundary)},
    }
    save_results("phase24_holographic_principle", result)
    return result


if __name__ == '__main__':
    main()
