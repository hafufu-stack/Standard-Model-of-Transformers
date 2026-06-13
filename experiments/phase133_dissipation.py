# -*- coding: utf-8 -*-
"""
Phase 133: Dissipation-Compression Tradeoff
Phase 131 showed dissipation halves after transition.
Phase 114 showed 21% compression is possible in the cooling valley.
This phase connects the two: is low-dissipation the REASON layers
can be pruned? And does dissipation predict pruning safety?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

TEST_TEXTS = [
    "The theory of general relativity predicts that massive objects warp the fabric of spacetime",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy",
    "The human brain contains approximately eighty six billion neurons connected by trillions",
    "Machine learning algorithms can identify complex patterns in large datasets with accuracy",
    "The periodic table organizes all known chemical elements according to their atomic number",
    "Quantum entanglement allows two particles to be correlated regardless of distance",
    "Climate models predict significant changes in global temperature patterns over decades",
    "The discovery of antibiotics revolutionized medicine and saved countless lives",
]


def main():
    print("=" * 70)
    print("Phase 133: Dissipation-Compression Tradeoff")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # Step 1: Measure per-layer dissipation (from P131 methodology)
    all_logp = []
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        logps = []
        for li in range(n_layers + 1):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            logp = torch.log(probs.max() + 1e-10).item()
            logps.append(logp if not np.isnan(logp) else -20)
        all_logp.append(logps)

    # Per-layer dissipation
    dissipation = []
    for li in range(n_layers):
        W_vals = []
        for pi in range(len(TEST_TEXTS)):
            w = (-all_logp[pi][li + 1]) - (-all_logp[pi][li])
            W_vals.append(w)
        exp_neg_W = np.mean([np.exp(-w) for w in W_vals])
        W_avg = np.mean(W_vals)
        jarzynski_F = -np.log(exp_neg_W + 1e-20)
        W_diss = W_avg - jarzynski_F
        dissipation.append(float(W_diss))

    # Step 2: Measure per-layer pruning impact (skip each layer individually)
    ppl_impacts = []
    baseline_loss = 0
    for text in TEST_TEXTS:
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        baseline_loss += out.loss.item() / len(TEST_TEXTS)

    for li in range(n_layers):
        def make_skip_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return hook_fn

        hook = model.model.layers[li].register_forward_hook(make_skip_hook())

        skip_loss = 0
        for text in TEST_TEXTS:
            inp = tok(text, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, labels=inp['input_ids'])
            skip_loss += out.loss.item() / len(TEST_TEXTS)

        hook.remove()
        ppl_impact = np.exp(skip_loss) / np.exp(baseline_loss)
        ppl_impacts.append(float(ppl_impact))

    layers = np.arange(n_layers)

    # Correlation between dissipation and pruning safety
    # Low dissipation = safe to prune?
    pruning_safety = [1.0 / (pi + 1e-10) for pi in ppl_impacts]  # inverse PPL impact
    r_corr, p_corr = sp_stats.pearsonr(dissipation, pruning_safety)

    # Safe pruning zone: layers with both low dissipation AND low PPL impact
    diss_threshold = np.percentile(dissipation, 30)
    ppl_threshold = np.percentile(ppl_impacts, 30)
    safe_layers = [li for li in range(n_layers)
                   if dissipation[li] < diss_threshold and ppl_impacts[li] < ppl_threshold]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Dissipation profile
    axes[0,0].plot(layers, dissipation, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    for sl in safe_layers:
        axes[0,0].axvspan(sl-0.5, sl+0.5, alpha=0.15, color='#27ae60')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$W_{diss}$')
    axes[0,0].set_title('(a) Dissipation per Layer')
    axes[0,0].legend()

    # (b) PPL impact profile
    axes[0,1].plot(layers, ppl_impacts, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    for sl in safe_layers:
        axes[0,1].axvspan(sl-0.5, sl+0.5, alpha=0.15, color='#27ae60')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('PPL Impact (skip layer)')
    axes[0,1].set_title('(b) Single-Layer Pruning Impact')

    # (c) Dissipation vs PPL impact scatter
    axes[0,2].scatter(dissipation, ppl_impacts, c=layers, cmap='coolwarm',
                      s=80, edgecolors='black', zorder=5)
    for i in range(0, n_layers, 3):
        axes[0,2].annotate(f'L{i}', (dissipation[i], ppl_impacts[i]), fontsize=7)
    axes[0,2].set_xlabel('$W_{diss}$')
    axes[0,2].set_ylabel('PPL Impact')
    axes[0,2].set_title(f'(c) Dissipation vs Pruning ($r={r_corr:.3f}$)')

    # (d) Safe pruning zone
    for li in range(n_layers):
        c = '#27ae60' if li in safe_layers else '#c0392b'
        axes[1,0].bar(li, ppl_impacts[li], color=c, alpha=0.7, edgecolor='black')
    axes[1,0].axhline(y=ppl_threshold, color='#f39c12', linestyle='--', label='Safety threshold')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('PPL Impact')
    axes[1,0].set_title(f'(d) Safe Layers (green): {safe_layers}')
    axes[1,0].legend(fontsize=8)

    # (e) Cumulative savings
    sorted_by_impact = sorted(range(n_layers), key=lambda i: ppl_impacts[i])
    cum_savings = []
    cum_ppl = []
    running_ppl = 1.0
    for i, li in enumerate(sorted_by_impact):
        running_ppl *= ppl_impacts[li]
        cum_savings.append((i + 1) / n_layers * 100)
        cum_ppl.append(running_ppl)
    axes[1,1].plot(cum_savings, cum_ppl, 'o-', color='#8e44ad', markersize=4)
    axes[1,1].axhline(y=1.5, color='#f39c12', linestyle='--', label='PPL 1.5x limit')
    axes[1,1].set_xlabel('Layers Pruned (%)')
    axes[1,1].set_ylabel('Cumulative PPL Ratio')
    axes[1,1].set_title('(e) Pareto Frontier')
    axes[1,1].legend()

    # (f) Summary
    max_prune = 0
    for i, ppl in enumerate(cum_ppl):
        if ppl < 1.5:
            max_prune = cum_savings[i]
    summary = (
        f"Dissipation-Compression Tradeoff\n\n"
        f"Correlation: r={r_corr:.3f} (p={p_corr:.3f})\n"
        f"Safe layers: {safe_layers}\n"
        f"Max pruning at PPL<1.5x: {max_prune:.0f}%\n\n"
        f"Low dissipation {'PREDICTS' if abs(r_corr) > 0.3 else 'does NOT predict'}\n"
        f"pruning safety"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 133: Dissipation-Compression Tradeoff',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase133_dissipation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Correlation dissipation-safety: r={r_corr:.3f}")
    print(f"Safe layers: {safe_layers}")
    print(f"Max pruning at PPL<1.5x: {max_prune:.0f}%")
    print(f"{'='*70}")

    save_results('phase133_dissipation', {
        'experiment': 'Dissipation-Compression Tradeoff',
        'dissipation': dissipation,
        'ppl_impacts': ppl_impacts,
        'safe_layers': safe_layers,
        'summary': {
            'correlation': float(r_corr),
            'p_value': float(p_corr),
            'safe_layers': safe_layers,
            'max_prune_pct': float(max_prune),
        }
    })


if __name__ == '__main__':
    main()
