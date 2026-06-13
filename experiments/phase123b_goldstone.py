# -*- coding: utf-8 -*-
"""
Phase 123b: Goldstone Bosons v2
Original P123 used perturbation_scale=0.5 which was too small.
Use larger perturbations and more PCs. Also use hidden state norm
as the natural scale for perturbation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]


def main():
    print("=" * 70)
    print("Phase 123b: Goldstone Bosons v2 (larger perturbation)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 22

    test_layers = [8, 14, 20, 22, 25, 27]
    n_pcs = 8

    results = {}

    for test_layer in test_layers:
        if test_layer >= n_layers or test_layer < 1:
            continue

        # Collect hidden states at this layer for PCA
        hidden_vecs = []
        hidden_norms = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[test_layer][0, -1, :].float()
            hidden_vecs.append(h.cpu().numpy())
            hidden_norms.append(h.norm().item())

        avg_norm = np.mean(hidden_norms)
        # Use 10% of average norm as perturbation scale
        perturb_scale = avg_norm * 0.10

        h_mat = np.array(hidden_vecs)
        n_actual_pcs = min(n_pcs, len(PROMPTS) - 1)
        pca = PCA(n_components=n_actual_pcs)
        pca.fit(h_mat)
        pc_dirs = pca.components_

        pc_results = []
        for pc_idx in range(n_actual_pcs):
            direction = torch.tensor(pc_dirs[pc_idx], dtype=torch.float32, device=device)
            direction = direction / direction.norm()

            delta_losses = []
            top1_matches = []
            kl_divs = []

            for prompt in PROMPTS:
                inp = tok(prompt, return_tensors='pt').to(device)

                # Baseline
                with torch.no_grad():
                    out_base = model(**inp, labels=inp['input_ids'])
                base_loss = out_base.loss.item()
                base_logits = out_base.logits[0, -1, :].float()
                base_probs = torch.softmax(base_logits, dim=-1)
                base_top1 = torch.argmax(base_probs).item()

                # Perturb
                layer_idx = test_layer - 1
                if layer_idx >= len(model.model.layers):
                    continue

                def make_hook(dir_vec, scale):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            h[:, -1, :] += scale * dir_vec
                            return (h,) + output[1:]
                        return output
                    return hook_fn

                hook = model.model.layers[layer_idx].register_forward_hook(
                    make_hook(direction, perturb_scale))

                with torch.no_grad():
                    out_pert = model(**inp, labels=inp['input_ids'])
                pert_loss = out_pert.loss.item()
                pert_logits = out_pert.logits[0, -1, :].float()
                pert_probs = torch.softmax(pert_logits, dim=-1)
                pert_top1 = torch.argmax(pert_probs).item()

                hook.remove()

                delta_losses.append(abs(pert_loss - base_loss))
                top1_matches.append(int(pert_top1 == base_top1))

                # KL divergence
                kl = torch.nn.functional.kl_div(
                    torch.log(pert_probs + 1e-10), base_probs,
                    reduction='sum').item()
                kl_divs.append(min(kl, 100))

            if delta_losses:
                pc_results.append({
                    'pc': pc_idx,
                    'delta_loss': float(np.mean(delta_losses)),
                    'top1_preserved': float(np.mean(top1_matches)),
                    'kl': float(np.mean(kl_divs)),
                    'explained_var': float(pca.explained_variance_ratio_[pc_idx]),
                })

        # Classify: Goldstone = low delta_loss AND high top1_preserved
        if pc_results:
            median_dl = np.median([r['delta_loss'] for r in pc_results])
            goldstone = [r for r in pc_results
                         if r['delta_loss'] < median_dl * 0.3 and r['top1_preserved'] > 0.7]
            massive = [r for r in pc_results
                       if r['delta_loss'] > median_dl * 1.5]
        else:
            goldstone, massive = [], []

        results[test_layer] = {
            'pc_results': pc_results,
            'n_goldstone': len(goldstone),
            'n_massive': len(massive),
            'perturb_scale': float(perturb_scale),
        }
        print(f"  L{test_layer}: {len(goldstone)} Goldstone, {len(massive)} massive "
              f"(scale={perturb_scale:.1f})")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_l = plt.cm.coolwarm(np.linspace(0, 1, len(test_layers)))

    # (a) Delta loss per PC
    for i, tl in enumerate(test_layers):
        if tl in results and results[tl]['pc_results']:
            pcs = [r['pc'] for r in results[tl]['pc_results']]
            dls = [r['delta_loss'] for r in results[tl]['pc_results']]
            axes[0,0].plot(pcs, dls, 'o-', color=colors_l[i], markersize=5,
                          label=f'L{tl} ({results[tl]["n_goldstone"]}G)')
    axes[0,0].set_xlabel('PC Direction')
    axes[0,0].set_ylabel('$|\\Delta Loss|$')
    axes[0,0].set_title('(a) Loss Sensitivity')
    axes[0,0].legend(fontsize=7)

    # (b) KL divergence per PC
    for i, tl in enumerate(test_layers):
        if tl in results and results[tl]['pc_results']:
            pcs = [r['pc'] for r in results[tl]['pc_results']]
            kls = [r['kl'] for r in results[tl]['pc_results']]
            axes[0,1].plot(pcs, kls, 'o-', color=colors_l[i], markersize=5,
                          label=f'L{tl}')
    axes[0,1].set_xlabel('PC Direction')
    axes[0,1].set_ylabel('KL Divergence')
    axes[0,1].set_title('(b) Distribution Shift')
    axes[0,1].legend(fontsize=7)

    # (c) Top-1 preservation
    for i, tl in enumerate(test_layers):
        if tl in results and results[tl]['pc_results']:
            pcs = [r['pc'] for r in results[tl]['pc_results']]
            t1 = [r['top1_preserved'] for r in results[tl]['pc_results']]
            axes[0,2].plot(pcs, t1, 'o-', color=colors_l[i], markersize=5,
                          label=f'L{tl}')
    axes[0,2].set_xlabel('PC Direction')
    axes[0,2].set_ylabel('Top-1 Preserved')
    axes[0,2].set_title('(c) Prediction Stability')
    axes[0,2].legend(fontsize=7)

    # (d) Goldstone count vs layer
    valid_layers = [tl for tl in test_layers if tl in results]
    golds = [results[tl]['n_goldstone'] for tl in valid_layers]
    massives = [results[tl]['n_massive'] for tl in valid_layers]
    x = np.arange(len(valid_layers))
    axes[1,0].bar(x - 0.15, golds, 0.3, color='#f39c12', label='Goldstone')
    axes[1,0].bar(x + 0.15, massives, 0.3, color='#8e44ad', label='Massive')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([f'L{tl}' for tl in valid_layers])
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('(d) Mode Classification')
    axes[1,0].legend()

    # (e) Ratio Goldstone/total vs layer
    ratios = [g/(g+m+1e-10) for g, m in zip(golds, massives)]
    bar_c = ['#f39c12' if tl >= L0 else '#2980b9' for tl in valid_layers]
    axes[1,1].bar(range(len(valid_layers)), ratios, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(valid_layers)))
    axes[1,1].set_xticklabels([f'L{tl}' for tl in valid_layers])
    axes[1,1].set_ylabel('Goldstone / (G+M)')
    axes[1,1].set_title('(e) Goldstone Fraction')

    # (f) Summary
    pre_g = np.mean([results[tl]['n_goldstone'] for tl in valid_layers if tl < L0])
    post_g = np.mean([results[tl]['n_goldstone'] for tl in valid_layers if tl >= L0])
    summary = (
        f"Goldstone Bosons v2\n\n"
        + "\n".join(f"L{tl}: {results[tl]['n_goldstone']}G / {results[tl]['n_massive']}M"
                    for tl in valid_layers)
        + f"\n\nPre-L0: {pre_g:.1f} Goldstone\n"
        f"Post-L0: {post_g:.1f} Goldstone\n\n"
        f"{'MORE flat dirs post-transition' if post_g > pre_g else 'Similar'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 123b: Goldstone Bosons v2', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase123b_goldstone')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-L0 Goldstone: {pre_g:.1f}, Post: {post_g:.1f}")
    print(f"{'='*70}")

    save_results('phase123b_goldstone', {
        'experiment': 'Goldstone Bosons v2',
        'results': {str(k): {'n_goldstone': v['n_goldstone'], 'n_massive': v['n_massive']}
                    for k, v in results.items()},
        'summary': {'pre_gold': float(pre_g), 'post_gold': float(post_g)},
    })


if __name__ == '__main__':
    main()
