# -*- coding: utf-8 -*-
"""
Phase 88: Thermodynamic Token Routing
Skip FFN computation for tokens whose thermodynamic state indicates
cooling is complete (|dU/dT| threshold), measuring accuracy vs savings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

EVAL_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("Water boils at", "100"),
    ("The speed of light is approximately", "300"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Photosynthesis converts sunlight into", "energy"),
    ("The chemical symbol for gold is", "Au"),
    ("The human body has approximately", "206"),
    ("The boiling point of nitrogen is about", "minus"),
    ("Einstein is famous for the theory of", "relativity"),
]


def measure_per_token_thermodynamics(model, tok, device, prompt):
    """Measure U and T at each layer for the last token."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    Us = []
    Ts = []
    for li, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        U = h.norm().item()
        Us.append(U)

        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(T):
            T = 0.0
        Ts.append(T)

    # Compute dU/dT (specific heat) at each layer
    dUdTs = []
    for i in range(1, len(Us)):
        dU = Us[i] - Us[i-1]
        dT = Ts[i] - Ts[i-1]
        if abs(dT) > 1e-6:
            dUdTs.append(dU / dT)
        else:
            dUdTs.append(0.0)

    return Us, Ts, dUdTs


def simulate_ffn_skip(model, tok, device, prompt, answer, skip_threshold):
    """Simulate skipping FFN at layers where |dU/dT| exceeds threshold."""
    # First: get thermodynamic profile (no skip)
    Us, Ts, dUdTs = measure_per_token_thermodynamics(model, tok, device, prompt)

    # Determine which layers to skip
    n_layers = len(model.model.layers)
    skip_layers = set()
    for i, cv in enumerate(dUdTs):
        # Skip FFN if specific heat magnitude is very large (cooling saturated)
        if abs(cv) > skip_threshold and i > 3:  # never skip first few layers
            skip_layers.add(i)

    n_skipped = len(skip_layers)
    skip_fraction = n_skipped / n_layers if n_layers > 0 else 0

    # Run with FFN scaling to zero at skip layers
    hooks = []

    def make_zero_hook():
        def hook(module, input, output):
            return output * 0.0
        return hook

    for li in skip_layers:
        if li < n_layers:
            h = model.model.layers[li].mlp.register_forward_hook(make_zero_hook())
            hooks.append(h)

    # Evaluate
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    for h in hooks:
        h.remove()

    # Check accuracy
    ans_ids = tok(answer, add_special_tokens=False)['input_ids']
    if ans_ids:
        correct_prob = probs[ans_ids[0]].item()
        top_id = torch.argmax(probs).item()
        correct = (top_id == ans_ids[0])
    else:
        correct_prob = 0.0
        correct = False

    return {
        'skip_fraction': float(skip_fraction),
        'n_skipped': n_skipped,
        'correct': bool(correct),
        'correct_prob': float(correct_prob),
        'top_token': tok.decode([torch.argmax(probs).item()]),
    }


def baseline_accuracy(model, tok, device):
    """Get baseline accuracy without any skipping."""
    correct_count = 0
    for prompt, answer in EVAL_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1, :].float()
        top_id = torch.argmax(logits).item()
        ans_ids = tok(answer, add_special_tokens=False)['input_ids']
        if ans_ids and top_id == ans_ids[0]:
            correct_count += 1
    return correct_count / len(EVAL_PROMPTS)


def main():
    print("=" * 70)
    print("Phase 88: Thermodynamic Token Routing")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Baseline
    base_acc = baseline_accuracy(model, tok, device)
    print(f"  Baseline accuracy: {base_acc:.1%}")

    # Sweep thresholds
    thresholds = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    sweep_results = []

    for thresh in thresholds:
        correct_count = 0
        total_skip_frac = 0

        for prompt, answer in EVAL_PROMPTS:
            result = simulate_ffn_skip(model, tok, device, prompt, answer, thresh)
            if result['correct']:
                correct_count += 1
            total_skip_frac += result['skip_fraction']

        acc = correct_count / len(EVAL_PROMPTS)
        avg_skip = total_skip_frac / len(EVAL_PROMPTS)

        sweep_results.append({
            'threshold': float(thresh),
            'accuracy': float(acc),
            'avg_skip_fraction': float(avg_skip),
            'accuracy_retention': float(acc / (base_acc + 1e-10)),
        })
        print(f"  threshold={thresh:.0f}: acc={acc:.1%}, skip={avg_skip:.1%}, "
              f"retention={acc/(base_acc+1e-10):.1%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    threshs = [r['threshold'] for r in sweep_results]
    accs = [r['accuracy'] for r in sweep_results]
    skips = [r['avg_skip_fraction'] for r in sweep_results]
    retentions = [r['accuracy_retention'] for r in sweep_results]

    # (a) Accuracy vs Skip fraction (Pareto)
    axes[0].scatter(skips, accs, s=100, c=threshs, cmap='viridis',
                    edgecolors='black', zorder=5)
    axes[0].plot(skips, accs, '--', color='gray', alpha=0.5)
    axes[0].axhline(y=base_acc, color='#c0392b', linestyle='--',
                    label=f'Baseline = {base_acc:.1%}')
    axes[0].set_xlabel('FFN Skip Fraction')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('(a) Pareto Frontier')
    axes[0].legend()
    cb = plt.colorbar(axes[0].collections[0], ax=axes[0], shrink=0.7)
    cb.set_label('Threshold')

    # (b) Threshold sweep
    axes[1].plot(threshs, accs, 'o-', color='#c0392b', linewidth=2, label='Accuracy')
    ax1b = axes[1].twinx()
    ax1b.plot(threshs, skips, 's-', color='#2980b9', linewidth=2, label='Skip %')
    axes[1].set_xlabel('|dU/dT| Threshold')
    axes[1].set_ylabel('Accuracy', color='#c0392b')
    ax1b.set_ylabel('Skip Fraction', color='#2980b9')
    axes[1].set_title('(b) Threshold Sweep')
    axes[1].set_xscale('log')

    # (c) Retention vs savings
    savings = [s * 100 for s in skips]
    retention_pct = [r * 100 for r in retentions]
    axes[2].scatter(savings, retention_pct, s=100, c='#27ae60', edgecolors='black')
    for i, t in enumerate(threshs):
        axes[2].annotate(f'{t:.0f}', (savings[i], retention_pct[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[2].axhline(y=99, color='#c0392b', linestyle='--', label='99% retention')
    axes[2].set_xlabel('FFN Compute Saved (%)')
    axes[2].set_ylabel('Accuracy Retention (%)')
    axes[2].set_title('(c) Efficiency Frontier')
    axes[2].legend()

    # Find optimal
    optimal = max(sweep_results,
                  key=lambda r: r['avg_skip_fraction'] if r['accuracy_retention'] >= 0.95 else -1)

    fig.suptitle(f'Phase 88: Thermodynamic Token Routing '
                 f'(Best: {optimal["avg_skip_fraction"]:.0%} saved at '
                 f'{optimal["accuracy_retention"]:.0%} retention)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase88_token_routing')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Baseline: {base_acc:.1%}")
    print(f"Best 95%+ retention: threshold={optimal['threshold']}, "
          f"skip={optimal['avg_skip_fraction']:.1%}, "
          f"acc={optimal['accuracy']:.1%}")
    print(f"{'='*70}")

    save_results('phase88_token_routing', {
        'experiment': 'Thermodynamic Token Routing',
        'baseline_accuracy': float(base_acc),
        'sweep': sweep_results,
        'optimal': optimal,
    })


if __name__ == '__main__':
    main()
