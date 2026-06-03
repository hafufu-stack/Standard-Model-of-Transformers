# -*- coding: utf-8 -*-
"""
Phase 33: Dark Energy Phase Diagram (Opus Original)
=====================================================
Fine-grained FFN suppression to find the exact critical point
where coherent output collapses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 33: Dark Energy Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    norm_layer = model.model.norm
    lm_head = model.lm_head

    test_cases = [
        ("The capital of France is", "Paris"),
        ("Water boils at", "100"),
        ("The largest planet is", "Jupiter"),
        ("Two plus three equals", "five"),
    ]

    # Fine-grained beta sweep
    betas = np.arange(1.0, -0.05, -0.05)
    betas = np.round(betas, 2)

    results = {prompt: {'betas': [], 'correct_probs': [], 'top_tokens': [], 'entropies': []}
               for prompt, _ in test_cases}

    for beta in betas:
        # Set up hooks
        handles = []
        if beta < 1.0:
            def make_hook(scale):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (output[0] * scale,) + output[1:]
                    return output * scale
                return hook
            for li in range(n_layers):
                h = model.model.layers[li].mlp.register_forward_hook(make_hook(beta))
                handles.append(h)

        for prompt, expected in test_cases:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)

            # Check expected token probability
            exp_ids = tok.encode(f" {expected}", add_special_tokens=False)
            exp_prob = max(probs[eid].item() for eid in exp_ids) if exp_ids else 0

            top_token = tok.decode(torch.argmax(probs).item())
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

            results[prompt]['betas'].append(float(beta))
            results[prompt]['correct_probs'].append(exp_prob)
            results[prompt]['top_tokens'].append(top_token)
            results[prompt]['entropies'].append(entropy)

        for h in handles:
            h.remove()

        if abs(beta % 0.1) < 0.01 or beta < 0.15:
            for prompt, expected in test_cases:
                idx = len(results[prompt]['betas']) - 1
                p = results[prompt]['correct_probs'][idx]
                tok_str = results[prompt]['top_tokens'][idx]
                print(f"  beta={beta:.2f} | '{prompt[:25]}' -> '{tok_str}' "
                      f"(P({expected})={p:.4f})")

    # Find critical points (where correct answer probability drops below 50% of baseline)
    critical_betas = {}
    for prompt, expected in test_cases:
        baseline_p = results[prompt]['correct_probs'][0]
        threshold = baseline_p * 0.5
        critical = 1.0
        for i, (b, p) in enumerate(zip(results[prompt]['betas'], results[prompt]['correct_probs'])):
            if p < threshold:
                critical = b
                break
        critical_betas[prompt] = critical

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    ax = axes[0]
    for i, (prompt, expected) in enumerate(test_cases):
        ax.plot(results[prompt]['betas'], results[prompt]['correct_probs'],
                'o-', color=colors[i], ms=3, label=f'{expected}', alpha=0.8)
        ax.axvline(x=critical_betas[prompt], color=colors[i], ls=':', alpha=0.4)
    ax.set_xlabel('FFN Scale (beta)')
    ax.set_ylabel(f'P(correct)')
    ax.set_title('(a) Correct Answer Probability')
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    ax = axes[1]
    for i, (prompt, expected) in enumerate(test_cases):
        ax.plot(results[prompt]['betas'], results[prompt]['entropies'],
                'o-', color=colors[i], ms=3, alpha=0.8)
    ax.set_xlabel('FFN Scale (beta)')
    ax.set_ylabel('Output Entropy')
    ax.set_title('(b) Entropy vs Dark Energy')
    ax.invert_xaxis()

    ax = axes[2]
    crits = list(critical_betas.values())
    labels = [exp for _, exp in test_cases]
    ax.bar(labels, crits, color=colors, alpha=0.8)
    for i, c in enumerate(crits):
        ax.text(i, c + 0.02, f'{c:.2f}', ha='center', fontsize=10)
    ax.set_ylabel('Critical beta')
    ax.set_title('(c) Phase Transition Points')
    ax.axhline(y=np.mean(crits), color='red', ls='--', label=f'Mean={np.mean(crits):.2f}')
    ax.legend()

    mean_crit = np.mean(crits)
    fig.suptitle(
        f"Phase 33: Dark Energy Phase Diagram\n"
        f"Critical beta = {mean_crit:.2f} (below this, coherent output collapses)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase33_phase_diagram")
    plt.close()

    verdict = (f"CRITICAL POINT at beta={mean_crit:.2f}. "
               f"Per-task: {dict(zip(labels, [f'{c:.2f}' for c in crits]))}. "
               f"FFN must contribute >{mean_crit*100:.0f}% of its normal output for coherent answers.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase33_phase_diagram", {
        'name': 'Phase 33: Dark Energy Phase Diagram',
        'summary': {'verdict': verdict, 'mean_critical_beta': float(mean_crit),
                    'critical_betas': {k: float(v) for k, v in critical_betas.items()}},
    })


if __name__ == '__main__':
    main()
