# -*- coding: utf-8 -*-
"""
Phase 154: Active Matter Defibrillation v4
Monitor the skewness flip (Phase 119 signature) in real-time and
inject a "defibrillation" shock when the model is stuck in the
exploration phase (hallucination signature: late skewness flip).
Force the phase transition to fire on time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

# Prompts where model might hallucinate (uncertain territory)
UNCERTAIN_PROMPTS = [
    "The most efficient way to solve all NP-complete problems is to",
    "The unified theory of quantum gravity states that",
    "In the year 3000, humans will have evolved to become",
    "The mathematical proof that P equals NP involves",
    "The chemical formula for the philosopher's stone is",
    "The exact solution to turbulence is given by",
]

# Factual prompts (control)
FACTUAL_PROMPTS = [
    "The chemical formula for water is",
    "The speed of light is approximately",
    "The capital of Japan is",
    "The boiling point of water at sea level is",
    "The formula for the area of a circle is",
    "The atomic number of carbon is",
]


def run_with_defibrillation(model, tok, prompt, device, n_layers, defib_layer=None, defib_scale=0):
    """Run inference with optional defibrillation at specified layer."""
    inp = tok(prompt, return_tensors='pt').to(device)

    hooks = []
    if defib_layer is not None and defib_layer < len(model.model.layers):
        def make_defib_hook(scale):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    # Defibrillation: reduce variance, increase skewness
                    # This pushes the system toward the localized (ordered) phase
                    h_float = h[:, -1, :].float()
                    mean = h_float.mean()
                    # Contract toward mean (reduce exploration)
                    contracted = mean + (h_float - mean) * (1.0 / (1.0 + scale))
                    h[:, -1, :] = contracted.to(h.dtype)
                    return (h,) + output[1:]
                return output
            return hook_fn
        hooks.append(model.model.layers[defib_layer].register_forward_hook(
            make_defib_hook(defib_scale)))

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()

    # Collect metrics
    S_vals = []
    skew_vals = []
    for li in range(n_layers):
        hs = out.hidden_states[li]
        h = hs[0, -1, :].float()
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        S = -(probs * torch.log(probs + 1e-10)).sum().item()
        S_vals.append(S if not np.isnan(S) else 0)

        sk = sp_stats.skew(h.cpu().numpy())
        skew_vals.append(float(sk) if not np.isnan(sk) else 0)

    # Final output
    final_logits = out.logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    confidence = final_probs.max().item()
    top_token = tok.decode([torch.argmax(final_probs)])

    # Eta
    etas = []
    for li in range(n_layers):
        T_subset = S_vals[:li+1]
        if len(T_subset) >= 4:
            T_hot = max(T_subset)
            T_cold = min(T_subset[len(T_subset)//2:])
            eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
        else:
            eta = 0
        etas.append(eta)

    return {
        'S': S_vals, 'skew': skew_vals, 'eta': etas,
        'confidence': confidence, 'top_token': top_token.strip(),
        'final_S': S_vals[-1],
    }


def main():
    print("=" * 70)
    print("Phase 154: Active Matter Defibrillation v4")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Defibrillation configs
    defib_configs = [
        ("none", None, 0),
        ("L19_s0.5", 19, 0.5),
        ("L20_s0.5", 20, 0.5),
        ("L21_s0.5", 21, 0.5),
        ("L21_s1.0", 21, 1.0),
        ("L21_s2.0", 21, 2.0),
    ]

    # Track confidence improvement
    uncertain_conf = {name: [] for name, _, _ in defib_configs}
    factual_conf = {name: [] for name, _, _ in defib_configs}
    uncertain_S = {name: [] for name, _, _ in defib_configs}
    factual_S = {name: [] for name, _, _ in defib_configs}

    for prompt in UNCERTAIN_PROMPTS:
        for config_name, dl, ds in defib_configs:
            r = run_with_defibrillation(model, tok, prompt, device, n_layers, dl, ds)
            uncertain_conf[config_name].append(r['confidence'])
            uncertain_S[config_name].append(r['final_S'])

    for prompt in FACTUAL_PROMPTS:
        for config_name, dl, ds in defib_configs:
            r = run_with_defibrillation(model, tok, prompt, device, n_layers, dl, ds)
            factual_conf[config_name].append(r['confidence'])
            factual_S[config_name].append(r['final_S'])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    names = [n for n, _, _ in defib_configs]

    # (a) Uncertain prompt confidence
    uc_means = [np.mean(uncertain_conf[n]) for n in names]
    uc_colors = ['#2980b9'] + ['#27ae60' if u > uc_means[0] else '#c0392b' for u in uc_means[1:]]
    axes[0,0].bar(range(len(names)), uc_means, color=uc_colors, alpha=0.8, edgecolor='black')
    axes[0,0].set_xticks(range(len(names)))
    axes[0,0].set_xticklabels(names, fontsize=7, rotation=20)
    axes[0,0].set_ylabel('Mean Confidence')
    axes[0,0].set_title('(a) Uncertain Prompts: Confidence')

    # (b) Uncertain prompt entropy
    us_means = [np.mean(uncertain_S[n]) for n in names]
    us_colors = ['#2980b9'] + ['#27ae60' if u < us_means[0] else '#c0392b' for u in us_means[1:]]
    axes[0,1].bar(range(len(names)), us_means, color=us_colors, alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(range(len(names)))
    axes[0,1].set_xticklabels(names, fontsize=7, rotation=20)
    axes[0,1].set_ylabel('Mean Final $S$')
    axes[0,1].set_title('(b) Uncertain Prompts: Entropy')

    # (c) Factual prompt confidence (control - should stay same)
    fc_means = [np.mean(factual_conf[n]) for n in names]
    fc_colors = ['#2980b9'] + ['#27ae60' if abs(f - fc_means[0]) < 0.05 else '#c0392b' for f in fc_means[1:]]
    axes[0,2].bar(range(len(names)), fc_means, color=fc_colors, alpha=0.8, edgecolor='black')
    axes[0,2].set_xticks(range(len(names)))
    axes[0,2].set_xticklabels(names, fontsize=7, rotation=20)
    axes[0,2].set_ylabel('Mean Confidence')
    axes[0,2].set_title('(c) Factual Prompts (Control)')

    # (d) Improvement ratio
    improvement = [(u - uc_means[0]) / (uc_means[0] + 1e-10) * 100 for u in uc_means]
    imp_colors = ['#27ae60' if i > 0 else '#c0392b' for i in improvement]
    axes[1,0].bar(range(len(names)), improvement, color=imp_colors, alpha=0.8, edgecolor='black')
    axes[1,0].axhline(y=0, color='black', linewidth=1)
    axes[1,0].set_xticks(range(len(names)))
    axes[1,0].set_xticklabels(names, fontsize=7, rotation=20)
    axes[1,0].set_ylabel('Confidence Improvement (%)')
    axes[1,0].set_title('(d) Defibrillation Effect')

    # (e) Example: entropy + skewness profiles
    test_prompt = UNCERTAIN_PROMPTS[0]
    r_base = run_with_defibrillation(model, tok, test_prompt, device, n_layers)
    r_defib = run_with_defibrillation(model, tok, test_prompt, device, n_layers, 21, 1.0)
    axes[1,1].plot(range(n_layers), r_base['skew'], 'o-', color='#c0392b',
                  markersize=3, label='No defib')
    axes[1,1].plot(range(n_layers), r_defib['skew'], 's-', color='#27ae60',
                  markersize=3, label='Defib L21')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Skewness')
    axes[1,1].set_title('(e) Skewness Profile')
    axes[1,1].legend(fontsize=8)

    # (f) Summary
    best_config = names[np.argmax(uc_means)]
    best_improvement = max(improvement)
    summary = (
        f"Active Matter Defibrillation v4\n\n"
        f"Baseline uncertain conf: {uc_means[0]:.3f}\n"
        f"Best config: {best_config}\n"
        f"Best conf: {max(uc_means):.3f}\n"
        f"Improvement: {best_improvement:+.1f}%\n\n"
        f"Factual baseline: {fc_means[0]:.3f}\n"
        f"Factual damage: {(min(fc_means)-fc_means[0])/fc_means[0]*100:+.1f}%\n\n"
        f"Defibrillation\n"
        f"{'IMPROVES' if best_improvement > 5 else 'MINIMAL EFFECT on'}\n"
        f"uncertain predictions"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 154: Active Matter Defibrillation v4',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase154_defibrillation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Best: {best_config} (improvement: {best_improvement:+.1f}%)")
    print(f"{'='*70}")

    save_results('phase154_defibrillation', {
        'experiment': 'Active Matter Defibrillation v4',
        'summary': {
            'best_config': best_config,
            'best_improvement_pct': float(best_improvement),
            'baseline_conf': float(uc_means[0]),
        }
    })


if __name__ == '__main__':
    main()
