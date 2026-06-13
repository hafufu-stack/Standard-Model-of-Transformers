# -*- coding: utf-8 -*-
"""
Phase 123: Goldstone Bosons of Meaning
After symmetry breaking (L22+), find "flat directions" in the loss
landscape where style changes but facts don't.
Perturb hidden states along different principal components and measure
which ones change loss vs which are "massless" (Goldstone modes).
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
]


def main():
    print("=" * 70)
    print("Phase 123: Goldstone Bosons of Meaning")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 22

    # Test layers: pre-transition, at transition, post-transition
    test_layers = [10, 15, 20, 22, 24, 27]
    n_pcs = 10  # number of PCA directions to test
    perturbation_scale = 0.5

    results = {}

    for test_layer in test_layers:
        if test_layer >= n_layers:
            continue

        loss_changes = []  # how much loss changes for each PC direction
        top1_changes = []  # does top-1 prediction change?

        # Collect hidden states at this layer
        hidden_vecs = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            hidden_vecs.append(out.hidden_states[test_layer][0, -1, :].float().cpu().numpy())

        # PCA on hidden states
        h_mat = np.array(hidden_vecs)
        pca = PCA(n_components=min(n_pcs, len(PROMPTS) - 1))
        pca.fit(h_mat)
        pc_dirs = pca.components_  # (n_pcs, d)

        # For each PC direction, perturb and measure loss change
        for pc_idx in range(min(n_pcs, len(pc_dirs))):
            direction = torch.tensor(pc_dirs[pc_idx], dtype=torch.float32, device=device)
            direction = direction / direction.norm()  # unit vector

            delta_losses = []
            top1_preserved = []

            for prompt in PROMPTS:
                inp = tok(prompt, return_tensors='pt').to(device)

                # Baseline loss
                with torch.no_grad():
                    out_base = model(**inp, labels=inp['input_ids'])
                base_loss = out_base.loss.item()
                base_top1 = torch.argmax(out_base.logits[0, -1, :]).item()

                # Perturb hidden state at test_layer
                def make_perturb_hook(dir_vec, scale):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            h[:, -1, :] = h[:, -1, :] + scale * dir_vec.unsqueeze(0)
                            return (h,) + output[1:]
                        else:
                            output[:, -1, :] = output[:, -1, :] + scale * dir_vec.unsqueeze(0)
                            return output
                    return hook_fn

                if test_layer > 0 and test_layer <= len(model.model.layers):
                    hook = model.model.layers[test_layer - 1].register_forward_hook(
                        make_perturb_hook(direction, perturbation_scale))
                else:
                    continue

                with torch.no_grad():
                    out_perturbed = model(**inp, labels=inp['input_ids'])
                perturbed_loss = out_perturbed.loss.item()
                perturbed_top1 = torch.argmax(out_perturbed.logits[0, -1, :]).item()

                hook.remove()

                delta_losses.append(abs(perturbed_loss - base_loss))
                top1_preserved.append(int(perturbed_top1 == base_top1))

            if delta_losses:
                loss_changes.append({
                    'pc': pc_idx,
                    'mean_delta_loss': float(np.mean(delta_losses)),
                    'top1_preserved': float(np.mean(top1_preserved)),
                    'explained_var': float(pca.explained_variance_ratio_[pc_idx]) if pc_idx < len(pca.explained_variance_ratio_) else 0,
                })

        # Count Goldstone modes (delta_loss < threshold AND top1 preserved)
        if loss_changes:
            threshold = np.median([lc['mean_delta_loss'] for lc in loss_changes]) * 0.5
            goldstone_count = sum(1 for lc in loss_changes
                                 if lc['mean_delta_loss'] < threshold and lc['top1_preserved'] > 0.8)
        else:
            goldstone_count = 0

        results[test_layer] = {
            'loss_changes': loss_changes,
            'goldstone_count': goldstone_count,
        }
        print(f"  L{test_layer}: {goldstone_count} Goldstone modes / {len(loss_changes)} PCs")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Delta loss per PC at different layers
    colors_l = plt.cm.coolwarm(np.linspace(0, 1, len(test_layers)))
    for i, tl in enumerate(test_layers):
        if tl in results and results[tl]['loss_changes']:
            pcs = [lc['pc'] for lc in results[tl]['loss_changes']]
            dls = [lc['mean_delta_loss'] for lc in results[tl]['loss_changes']]
            axes[0, 0].plot(pcs, dls, 'o-', color=colors_l[i], markersize=4,
                           label=f'L{tl}')
    axes[0, 0].set_xlabel('PC Direction')
    axes[0, 0].set_ylabel('$|\\Delta Loss|$')
    axes[0, 0].set_title('(a) Loss Sensitivity per Direction')
    axes[0, 0].legend(fontsize=7)

    # (b) Top-1 preservation per PC
    for i, tl in enumerate(test_layers):
        if tl in results and results[tl]['loss_changes']:
            pcs = [lc['pc'] for lc in results[tl]['loss_changes']]
            t1p = [lc['top1_preserved'] for lc in results[tl]['loss_changes']]
            axes[0, 1].plot(pcs, t1p, 'o-', color=colors_l[i], markersize=4,
                           label=f'L{tl}')
    axes[0, 1].set_xlabel('PC Direction')
    axes[0, 1].set_ylabel('Top-1 Preserved')
    axes[0, 1].set_title('(b) Prediction Stability')
    axes[0, 1].legend(fontsize=7)

    # (c) Goldstone count vs layer
    golds = [results.get(tl, {}).get('goldstone_count', 0) for tl in test_layers if tl in results]
    valid_layers = [tl for tl in test_layers if tl in results]
    bar_c = ['#27ae60' if tl > L0 else '#2980b9' for tl in valid_layers]
    axes[0, 2].bar(range(len(valid_layers)), golds, color=bar_c, alpha=0.8, edgecolor='black')
    axes[0, 2].set_xticks(range(len(valid_layers)))
    axes[0, 2].set_xticklabels([f'L{tl}' for tl in valid_layers])
    axes[0, 2].set_ylabel('Goldstone Modes')
    axes[0, 2].set_title(f'(c) Goldstone Count')

    # (d) "Massive" vs "Massless" spectrum at L=27 (post-transition)
    if 27 in results and results[27]['loss_changes']:
        dls_27 = [lc['mean_delta_loss'] for lc in results[27]['loss_changes']]
        colors_m = ['#f39c12' if d < np.median(dls_27)*0.5 else '#8e44ad' for d in dls_27]
        axes[1, 0].bar(range(len(dls_27)), dls_27, color=colors_m, alpha=0.8, edgecolor='black')
        axes[1, 0].set_xlabel('PC Direction')
        axes[1, 0].set_ylabel('$|\\Delta Loss|$')
        axes[1, 0].set_title('(d) L27 Spectrum (gold=Goldstone)')

    # (e) Explained variance
    if 27 in results and results[27]['loss_changes']:
        evs = [lc['explained_var'] for lc in results[27]['loss_changes']]
        dls = [lc['mean_delta_loss'] for lc in results[27]['loss_changes']]
        axes[1, 1].scatter(evs, dls, s=80, c=range(len(evs)), cmap='coolwarm', edgecolors='black')
        axes[1, 1].set_xlabel('Explained Variance')
        axes[1, 1].set_ylabel('$|\\Delta Loss|$')
        axes[1, 1].set_title('(e) Variance vs Sensitivity')

    # (f) Summary
    pre_gold = np.mean([results[tl]['goldstone_count'] for tl in test_layers
                        if tl < L0 and tl in results])
    post_gold = np.mean([results[tl]['goldstone_count'] for tl in test_layers
                         if tl >= L0 and tl in results])
    summary = (
        f"Goldstone Mode Analysis\n\n"
        + "\n".join(f"L{tl}: {results[tl]['goldstone_count']} Goldstone modes"
                    for tl in test_layers if tl in results)
        + f"\n\nPre-transition: {pre_gold:.1f} modes\n"
        f"Post-transition: {post_gold:.1f} modes\n\n"
        f"Symmetry breaking creates\n"
        f"{'MORE' if post_gold > pre_gold else 'FEWER'} flat directions"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 123: Goldstone Bosons of Meaning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase123_goldstone')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-transition Goldstone: {pre_gold:.1f}")
    print(f"Post-transition Goldstone: {post_gold:.1f}")
    print(f"{'='*70}")

    save_results('phase123_goldstone', {
        'experiment': 'Goldstone Bosons',
        'results': {str(k): v for k, v in results.items()},
        'summary': {
            'pre_gold': float(pre_gold),
            'post_gold': float(post_gold),
        }
    })


if __name__ == '__main__':
    main()
