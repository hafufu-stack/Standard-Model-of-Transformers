# -*- coding: utf-8 -*-
"""
Phase 3: Surgical SNN Noise (Aletheia + SNN-Genesis Fusion)
=============================================================
Deep Think hypothesis: the reason stochastic resonance (noise)
improves LLM reasoning is that it disables "Grammar Police" heads
at L9-L11 (discovered by Aletheia), freeing the factual/logical
signal at deeper layers.

Test: inject noise ONLY at L9-L11 vs uniform noise vs L18 only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, get_logits, save_results, save_figure

def main():
    print("=" * 70)
    print("Phase 3: Surgical SNN Noise (Grammar Police Ablation)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    # ================================================================
    # Reasoning test cases (arithmetic + logic)
    # ================================================================
    test_cases = [
        {"prompt": "If all roses are flowers and all flowers are plants, then all roses are",
         "correct": " plants", "type": "logic"},
        {"prompt": "2 + 2 * 3 =",
         "correct": " 8", "type": "arithmetic"},
        {"prompt": "The capital of Japan is",
         "correct": " Tokyo", "type": "factual"},
        {"prompt": "If it rains, the ground gets wet. The ground is wet. Therefore",
         "correct": " it", "type": "logic"},
        {"prompt": "What comes after Monday, Tuesday, Wednesday?",
         "correct": " Thursday", "type": "pattern"},
        {"prompt": "5 * 7 =",
         "correct": " 35", "type": "arithmetic"},
        {"prompt": "The opposite of hot is",
         "correct": " cold", "type": "semantic"},
        {"prompt": "If A > B and B > C, then A",
         "correct": " >", "type": "logic"},
    ]

    # ================================================================
    # Noise injection configurations
    # ================================================================
    noise_configs = {
        'baseline': {'layers': [], 'sigma': 0.0},
        'grammar_police_L9-L11': {'layers': [9, 10, 11], 'sigma': 0.15},
        'uniform_all': {'layers': list(range(n_layers)), 'sigma': 0.15},
        'L18_only': {'layers': [18], 'sigma': 0.15},
        'execution_L16-L20': {'layers': [16, 17, 18, 19, 20], 'sigma': 0.15},
        'early_L0-L3': {'layers': [0, 1, 2, 3], 'sigma': 0.15},
    }

    # Test multiple noise levels for surgical injection
    sigma_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    def inject_noise_and_evaluate(model, tok, prompt, correct_token,
                                  noise_layers, sigma, n_trials=5):
        """Inject Gaussian noise at specified layers and measure accuracy."""
        correct_id = tok.encode(correct_token)[-1]
        probs_correct = []

        for trial in range(n_trials):
            handles = []
            for li in noise_layers:
                def make_noise_hook(s):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            noise = torch.randn_like(h) * s
                            return (h + noise,) + output[1:]
                        return output + torch.randn_like(output) * s
                    return hook
                h = model.model.layers[li].register_forward_hook(make_noise_hook(sigma))
                handles.append(h)

            logits = get_logits(model, tok, prompt, device)

            for h in handles:
                h.remove()

            probs = torch.softmax(logits.float(), dim=-1)
            probs_correct.append(probs[correct_id].item())

        return np.mean(probs_correct), np.std(probs_correct)

    # ================================================================
    # Main experiment: Compare noise configurations
    # ================================================================
    print("\n--- Comparing noise configurations ---")
    config_results = {}

    for config_name, config in noise_configs.items():
        print(f"\n  Config: {config_name} (layers={config['layers']}, sigma={config['sigma']})")
        scores = []
        for tc in test_cases:
            mean_p, std_p = inject_noise_and_evaluate(
                model, tok, tc['prompt'], tc['correct'],
                config['layers'], config['sigma']
            )
            scores.append(mean_p)
            print(f"    {tc['type']:12s} P(correct)={mean_p:.4f} +/- {std_p:.4f}")

        avg = np.mean(scores)
        config_results[config_name] = {
            'avg_score': avg,
            'scores': scores,
            'config': config,
        }
        print(f"    AVERAGE: {avg:.4f}")

    # ================================================================
    # Sigma sweep for surgical injection
    # ================================================================
    print("\n--- Sigma sweep for surgical (L9-L11) injection ---")
    sigma_sweep = []
    for sigma in sigma_values:
        scores = []
        for tc in test_cases:
            mean_p, _ = inject_noise_and_evaluate(
                model, tok, tc['prompt'], tc['correct'],
                [9, 10, 11], sigma, n_trials=3
            )
            scores.append(mean_p)
        avg = np.mean(scores)
        sigma_sweep.append({'sigma': sigma, 'avg_score': avg})
        print(f"  sigma={sigma:.2f}: avg P(correct) = {avg:.4f}")

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Config comparison
    ax = axes[0]
    names = list(config_results.keys())
    avgs = [config_results[n]['avg_score'] for n in names]
    colors = ['#95a5a6', '#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    bars = ax.barh(range(len(names)), avgs, color=colors[:len(names)], alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Average P(correct)')
    ax.set_title('(a) Noise Configuration Comparison')
    # Mark the best
    best_idx = np.argmax(avgs)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)

    # (b) Sigma sweep
    ax = axes[1]
    sigmas = [s['sigma'] for s in sigma_sweep]
    sweep_scores = [s['avg_score'] for s in sigma_sweep]
    ax.plot(sigmas, sweep_scores, 'o-', color='#e74c3c', lw=2, ms=8)
    best_sigma = sigmas[np.argmax(sweep_scores)]
    ax.axvline(x=best_sigma, color='green', ls='--', lw=1.5,
               label=f'Optimal sigma={best_sigma:.2f}')
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Avg P(correct)')
    ax.set_title('(b) Surgical Noise (L9-L11) Sigma Sweep')
    ax.legend()

    # (c) Per-task comparison: surgical vs uniform
    ax = axes[2]
    surgical = config_results.get('grammar_police_L9-L11', {}).get('scores', [])
    uniform = config_results.get('uniform_all', {}).get('scores', [])
    baseline = config_results.get('baseline', {}).get('scores', [])
    x = np.arange(len(test_cases))
    w = 0.25
    if baseline:
        ax.bar(x - w, baseline, w, label='Baseline', color='#95a5a6', alpha=0.8)
    if surgical:
        ax.bar(x, surgical, w, label='Surgical (L9-11)', color='#e74c3c', alpha=0.8)
    if uniform:
        ax.bar(x + w, uniform, w, label='Uniform', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([tc['type'][:6] for tc in test_cases], fontsize=8, rotation=45)
    ax.set_ylabel('P(correct)')
    ax.set_title('(c) Per-Task: Surgical vs Uniform vs Baseline')
    ax.legend(fontsize=8)

    surgical_avg = config_results.get('grammar_police_L9-L11', {}).get('avg_score', 0)
    uniform_avg = config_results.get('uniform_all', {}).get('avg_score', 0)
    baseline_avg = config_results.get('baseline', {}).get('avg_score', 0)

    fig.suptitle(
        f"Phase 3: Surgical SNN Noise\n"
        f"Baseline={baseline_avg:.4f} | Surgical(L9-11)={surgical_avg:.4f} | "
        f"Uniform={uniform_avg:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase3_surgical_noise")
    plt.close()

    # ================================================================
    # Verdict
    # ================================================================
    best_config = max(config_results.items(), key=lambda x: x[1]['avg_score'])
    improvement = surgical_avg / max(baseline_avg, 1e-10)

    if best_config[0] == 'grammar_police_L9-L11':
        verdict = (f"GRAMMAR POLICE CONFIRMED: Surgical noise at L9-L11 is BEST "
                   f"(P={surgical_avg:.4f}, {improvement:.2f}x baseline). "
                   f"Stochastic resonance works by disabling suppression heads.")
    else:
        verdict = (f"GRAMMAR POLICE PARTIAL: Best config = {best_config[0]} "
                   f"(P={best_config[1]['avg_score']:.4f}). Surgical L9-L11 = {surgical_avg:.4f}.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 3: Surgical SNN Noise',
        'summary': {'verdict': verdict, 'best_config': best_config[0],
                    'surgical_avg': surgical_avg, 'baseline_avg': baseline_avg},
        'config_results': {k: {'avg_score': v['avg_score'], 'scores': v['scores']}
                          for k, v in config_results.items()},
        'sigma_sweep': sigma_sweep,
    }
    save_results("phase3_surgical_noise", result)
    return result


if __name__ == '__main__':
    main()
