# -*- coding: utf-8 -*-
"""
Phase 234: Thermodynamic Early Exit
======================================
Use thermodynamic convergence criteria to determine when a transformer
has "finished thinking" and can exit early without quality loss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The square root of sixteen is",
    "Gravity pulls objects toward the",
    "The largest planet in our solar system is",
    "Photosynthesis requires sunlight and",
    "The chemical symbol for gold is",
    "The speed of light in vacuum is approximately",
    "Shakespeare wrote the play Romeo and",
    "The boiling point of water is one hundred",
    "Mitochondria are known as the powerhouse of",
    "The Great Wall of China was built to",
    "DNA stands for deoxyribonucleic",
    "The human body has approximately 206",
    "Isaac Newton discovered the law of",
    "The Mona Lisa was painted by Leonardo",
    "Electrons orbit the nucleus of an",
    "The Amazon River flows through South",
    "Albert Einstein developed the theory of",
    "The periodic table has 118 confirmed",
]


def thermodynamic_early_exit(model, tok, device, model_name):
    """Test early exit using thermodynamic convergence criteria."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # For each prompt: find where it converges
    results_per_prompt = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get final prediction (ground truth)
        final_logits = out.logits[0, -1, :].float()
        final_probs = torch.softmax(final_logits, dim=-1)
        final_token = final_probs.argmax().item()
        final_p1 = final_probs.max().item()

        # Track per-layer predictions
        layer_data = []
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            token = probs.argmax().item()
            p1 = float(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T = float(S) if not np.isnan(S) else 0

            # KL divergence from final
            kl = torch.nn.functional.kl_div(
                torch.log(probs + 1e-10), final_probs,
                reduction='sum').item()

            layer_data.append({
                'layer': li,
                'token': token,
                'p1': p1,
                'T': T,
                'kl_from_final': float(kl) if not np.isnan(kl) else 1e6,
                'correct': token == final_token,
            })

        # Find earliest convergence layer
        # Criterion 1: dT/dl < threshold for 3 consecutive layers
        # Criterion 2: Same token as final for 3 consecutive layers
        # Criterion 3: P1 > threshold

        def find_convergence(criterion_fn, window=3):
            for i in range(len(layer_data) - window):
                if all(criterion_fn(layer_data[i+j]) for j in range(window)):
                    return i
            return len(layer_data) - 1

        # Token stability
        L_token = find_convergence(lambda d: d['correct'])

        # P1 threshold
        L_p1_50 = find_convergence(lambda d: d['p1'] > 0.5)
        L_p1_30 = find_convergence(lambda d: d['p1'] > 0.3)

        # KL convergence
        L_kl = find_convergence(lambda d: d['kl_from_final'] < 0.1)

        results_per_prompt.append({
            'prompt': prompt[:40],
            'L_token': L_token,
            'L_p1_50': L_p1_50,
            'L_p1_30': L_p1_30,
            'L_kl': L_kl,
            'n_layers': len(layer_data),
            'layer_data': layer_data,
        })

    # Aggregate
    mean_L_token = float(np.mean([r['L_token'] for r in results_per_prompt]))
    mean_L_p1 = float(np.mean([r['L_p1_50'] for r in results_per_prompt]))
    mean_L_kl = float(np.mean([r['L_kl'] for r in results_per_prompt]))

    # Savings calculation
    savings_token = 1 - mean_L_token / n_layers
    savings_p1 = 1 - mean_L_p1 / n_layers
    savings_kl = 1 - mean_L_kl / n_layers

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_L_token': mean_L_token,
        'mean_L_p1': mean_L_p1,
        'mean_L_kl': mean_L_kl,
        'savings_token': savings_token,
        'savings_p1': savings_p1,
        'savings_kl': savings_kl,
        'per_prompt': results_per_prompt,
    }


def main():
    print("=" * 70)
    print("Phase 234: Thermodynamic Early Exit")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = thermodynamic_early_exit(model, tok, device, size)
        results[size] = r
        print(f"  Convergence: token=L{r['mean_L_token']:.1f}, P1=L{r['mean_L_p1']:.1f}, KL=L{r['mean_L_kl']:.1f}")
        print(f"  Savings: token={r['savings_token']:.0%}, P1={r['savings_p1']:.0%}, KL={r['savings_kl']:.0%}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) KL divergence from final over layers (avg)
    for size, r in results.items():
        n = r['per_prompt'][0]['n_layers']
        avg_kl = [float(np.mean([r['per_prompt'][p]['layer_data'][l]['kl_from_final']
                                  for p in range(len(PROMPTS)) if l < r['per_prompt'][p]['n_layers']]))
                  for l in range(n)]
        axes[0, 0].plot(range(n), avg_kl, '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(y=0.1, color='green', ls='--', alpha=0.5, label='KL=0.1')
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('KL from Final')
    axes[0, 0].set_title('(a) KL Divergence from Final')
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_yscale('log')

    # (b) P1 over layers
    for size, r in results.items():
        n = r['per_prompt'][0]['n_layers']
        avg_p1 = [float(np.mean([r['per_prompt'][p]['layer_data'][l]['p1']
                                  for p in range(len(PROMPTS)) if l < r['per_prompt'][p]['n_layers']]))
                  for l in range(n)]
        axes[0, 1].plot(range(n), avg_p1, '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(y=0.5, color='green', ls='--', alpha=0.5, label='P1=0.5')
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('P1')
    axes[0, 1].set_title('(b) Order Parameter Evolution')
    axes[0, 1].legend(fontsize=7)

    # (c) Token accuracy over layers
    for size, r in results.items():
        n = r['per_prompt'][0]['n_layers']
        accuracy = [float(np.mean([r['per_prompt'][p]['layer_data'][l]['correct']
                                    for p in range(len(PROMPTS)) if l < r['per_prompt'][p]['n_layers']]))
                   for l in range(n)]
        axes[0, 2].plot(range(n), accuracy, '-', color=colors[size], lw=2, label=size)
    axes[0, 2].axhline(y=1.0, color='green', ls='--', alpha=0.3)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('(c) Token Prediction Accuracy')
    axes[0, 2].legend(fontsize=7)

    # (d) Convergence histogram
    for si, (size, r) in enumerate(results.items()):
        L_tokens = [rp['L_token'] for rp in r['per_prompt']]
        axes[1, 0].hist(L_tokens, bins=range(r['n_layers']+1), alpha=0.5,
                       color=colors[size], label=size)
    axes[1, 0].set_xlabel('Convergence Layer')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('(d) Token Convergence Distribution')
    axes[1, 0].legend(fontsize=8)

    # (e) Savings bar chart
    criteria = ['Token', 'P1>0.5', 'KL<0.1']
    x = np.arange(len(criteria))
    width = 0.35
    for si, (size, r) in enumerate(results.items()):
        savings = [r['savings_token'], r['savings_p1'], r['savings_kl']]
        axes[1, 1].bar(x + si*width, savings, width, label=size, color=colors[size], alpha=0.7)
    axes[1, 1].set_xticks(x + width/2)
    axes[1, 1].set_xticklabels(criteria)
    axes[1, 1].set_ylabel('Compute Savings')
    axes[1, 1].set_title('(e) Potential Savings')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "EARLY EXIT ANALYSIS\n\n"
    for size, r in results.items():
        summary += f"{size} ({r['n_layers']}L):\n"
        summary += f"  Token conv: L{r['mean_L_token']:.1f} ({r['savings_token']:.0%})\n"
        summary += f"  P1 conv:    L{r['mean_L_p1']:.1f} ({r['savings_p1']:.0%})\n"
        summary += f"  KL conv:    L{r['mean_L_kl']:.1f} ({r['savings_kl']:.0%})\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 234: Thermodynamic Early Exit", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase234_early_exit')
    plt.close()
    save_results('phase234_early_exit', {'experiment': 'Early Exit', 'results': results})


if __name__ == '__main__':
    main()
