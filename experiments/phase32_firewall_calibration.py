# -*- coding: utf-8 -*-
"""
Phase 32: Firewall Threshold Calibration (Opus Original)
==========================================================
Find optimal PR*T variance threshold for hallucination detection.
Compute ROC-like curves for false positive / false negative rates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def generate_and_measure(model, tok, prompt, n_tokens, device):
    """Generate tokens and track PR*T at each step."""
    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']
    trace = []
    past_kv = None

    for t in range(n_tokens):
        curr_input = input_ids if past_kv is None else next_id
        with torch.no_grad():
            out = model(input_ids=curr_input, past_key_values=past_kv,
                       use_cache=True, output_hidden_states=False)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        PR = 1.0 / (probs ** 2).sum().item()
        trace.append(PR * T)
        next_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)

    return trace


def main():
    print("=" * 70)
    print("Phase 32: Firewall Threshold Calibration")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    factual_prompts = [
        "The chemical symbol for gold is",
        "Water consists of hydrogen and",
        "The speed of light is approximately",
        "Earth orbits around the",
        "DNA stands for deoxyribonucleic",
        "The square root of 144 is",
        "Photosynthesis converts sunlight into",
        "The atomic number of carbon is",
    ]

    ambiguous_prompts = [
        "The secret conspiracy behind the moon landing was",
        "In an alternate universe where gravity works backwards",
        "The ancient prophecy foretold that in 2030",
        "The hidden truth about consciousness is that",
        "According to leaked documents from the future",
        "The real reason dinosaurs went extinct was because",
        "A classified experiment revealed that dreams are actually",
        "The forbidden knowledge of the universe states that",
    ]

    n_tokens = 80

    factual_traces = []
    ambiguous_traces = []

    for p in factual_prompts:
        trace = generate_and_measure(model, tok, p, n_tokens, device)
        factual_traces.append(trace)
        print(f"  Factual: std={np.std(trace):.1f}")

    for p in ambiguous_prompts:
        trace = generate_and_measure(model, tok, p, n_tokens, device)
        ambiguous_traces.append(trace)
        print(f"  Ambiguous: std={np.std(trace):.1f}")

    # Compute per-trace std (firewall signal)
    factual_stds = [np.std(t) for t in factual_traces]
    ambiguous_stds = [np.std(t) for t in ambiguous_traces]

    # ROC curve: sweep threshold
    all_stds = factual_stds + ambiguous_stds
    labels = [0]*len(factual_stds) + [1]*len(ambiguous_stds)  # 0=factual, 1=ambiguous

    thresholds = np.linspace(min(all_stds)*0.5, max(all_stds)*1.5, 200)
    tpr_list, fpr_list = [], []
    for thresh in thresholds:
        tp = sum(1 for s, l in zip(all_stds, labels) if s > thresh and l == 1)
        fp = sum(1 for s, l in zip(all_stds, labels) if s > thresh and l == 0)
        fn = sum(1 for s, l in zip(all_stds, labels) if s <= thresh and l == 1)
        tn = sum(1 for s, l in zip(all_stds, labels) if s <= thresh and l == 0)
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # AUC
    auc = abs(np.trapz(tpr_list, fpr_list))

    # Optimal threshold (Youden's J)
    j_scores = [t - f for t, f in zip(tpr_list, fpr_list)]
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_tpr = tpr_list[best_idx]
    best_fpr = fpr_list[best_idx]

    print(f"\n--- Firewall ROC ---")
    print(f"  AUC = {auc:.3f}")
    print(f"  Optimal threshold = {best_thresh:.1f}")
    print(f"  TPR = {best_tpr:.2f}, FPR = {best_fpr:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.hist(factual_stds, bins=8, alpha=0.7, color='#2ecc71', label='Factual')
    ax.hist(ambiguous_stds, bins=8, alpha=0.7, color='#e74c3c', label='Ambiguous')
    ax.axvline(x=best_thresh, color='gold', ls='--', lw=2, label=f'Threshold={best_thresh:.0f}')
    ax.set_xlabel('PR*T Std Dev')
    ax.set_ylabel('Count')
    ax.set_title('(a) Firewall Signal Distribution')
    ax.legend()

    ax = axes[1]
    ax.plot(fpr_list, tpr_list, '-', color='#3498db', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    ax.plot(best_fpr, best_tpr, 'o', color='red', ms=10, label=f'Optimal (TPR={best_tpr:.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'(b) ROC Curve (AUC={auc:.3f})')
    ax.legend()

    ax = axes[2]
    for i, t in enumerate(factual_traces[:3]):
        ax.plot(range(len(t)), t, '-', color='#2ecc71', alpha=0.5, lw=1)
    for i, t in enumerate(ambiguous_traces[:3]):
        ax.plot(range(len(t)), t, '-', color='#e74c3c', alpha=0.5, lw=1)
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('PR x T')
    ax.set_title('(c) PRT Traces (green=factual, red=ambiguous)')

    fig.suptitle(
        f"Phase 32: Firewall Calibration\n"
        f"AUC={auc:.3f} | Optimal: thresh={best_thresh:.0f}, "
        f"TPR={best_tpr:.0f}%, FPR={best_fpr:.0f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase32_firewall_calibration")
    plt.close()

    verdict = (f"AUC={auc:.3f}. Optimal threshold={best_thresh:.0f} "
               f"achieves TPR={best_tpr:.0%}, FPR={best_fpr:.0%}.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase32_firewall_calibration", {
        'name': 'Phase 32: Firewall Calibration',
        'summary': {'verdict': verdict, 'auc': float(auc),
                    'optimal_threshold': float(best_thresh),
                    'tpr': float(best_tpr), 'fpr': float(best_fpr)},
    })


if __name__ == '__main__':
    main()
