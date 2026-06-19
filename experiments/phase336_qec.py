# -*- coding: utf-8 -*-
"""
Phase 336: Quantum Error Correction -- Redundancy & Recovery
=====================================================
Does the Transformer encode information with error-correcting
redundancy? Test whether hidden states can recover from erasure
(partial deletion of dimensions), analogous to quantum error
correction codes.
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_qec(model, tok, prompt, device):
    """Test quantum error correction properties."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    logits_original = out.logits[0, -1, :].float().cpu()
    top_token = int(torch.argmax(logits_original).item())

    # Test erasure at different layers and fractions
    erasure_fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    layer_recovery = []

    for li in range(1, n_layers + 1):
        frac_results = []
        for frac in erasure_fracs:
            # Erase a fraction of dimensions at layer li
            h = out.hidden_states[li][0, -1, :].clone()
            dim = h.shape[0]
            n_erase = int(frac * dim)

            # Random erasure (zero out)
            idx = torch.randperm(dim)[:n_erase]
            h_erased = h.clone()
            h_erased[idx] = 0

            # Measure recovery: cosine similarity with original
            cos = float(torch.nn.functional.cosine_similarity(
                h.unsqueeze(0), h_erased.unsqueeze(0)).item())

            # Measure information preservation: how much of the norm is preserved
            norm_ratio = float(torch.norm(h_erased).item() / (torch.norm(h).item() + 1e-10))

            frac_results.append({
                'erasure_frac': frac,
                'cosine': round(cos, 4),
                'norm_ratio': round(norm_ratio, 4),
            })

        # Critical erasure: max fraction with cos > 0.9
        critical = 0
        for fr in frac_results:
            if fr['cosine'] > 0.9:
                critical = fr['erasure_frac']
        layer_recovery.append({
            'critical_erasure': critical,
            'cos_at_10pct': frac_results[1]['cosine'],
            'cos_at_50pct': frac_results[4]['cosine'],
        })

    # SVD-based redundancy: how many singular values capture 99% of energy?
    redundancy_per_layer = []
    for li in range(1, n_layers + 1):
        h = out.hidden_states[li][0, :, :].float().cpu()  # (seq, dim)
        try:
            U, S, V = torch.linalg.svd(h, full_matrices=False)
            total_energy = float(torch.sum(S**2).item())
            cumsum = torch.cumsum(S**2, dim=0) / (total_energy + 1e-10)
            k99 = int((cumsum < 0.99).sum().item()) + 1
            effective_rank = float(k99 / S.shape[0])
            redundancy_per_layer.append(round(1.0 - effective_rank, 4))
        except:
            redundancy_per_layer.append(0.0)

    avg_redundancy = float(np.mean(redundancy_per_layer))
    avg_critical = float(np.mean([lr['critical_erasure'] for lr in layer_recovery]))
    avg_cos_10 = float(np.mean([lr['cos_at_10pct'] for lr in layer_recovery]))
    avg_cos_50 = float(np.mean([lr['cos_at_50pct'] for lr in layer_recovery]))

    return {
        'avg_redundancy': round(avg_redundancy, 4),
        'avg_critical_erasure': round(avg_critical, 4),
        'avg_cos_10pct': round(avg_cos_10, 4),
        'avg_cos_50pct': round(avg_cos_50, 4),
        'redundancy_profile': redundancy_per_layer,
        'critical_profile': [lr['critical_erasure'] for lr in layer_recovery],
        'qec_robust': avg_cos_10 > 0.95,
    }


def main():
    print("=" * 70)
    print("Phase 336: Quantum Error Correction")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        qec_data = []
        for prompt in PROMPTS:
            q = measure_qec(model, tok, prompt, device)
            qec_data.append(q)

        n = len(qec_data[0]['redundancy_profile'])
        all_results[size] = {
            'avg_redundancy': round(float(np.mean([q['avg_redundancy'] for q in qec_data])), 4),
            'avg_critical_erasure': round(float(np.mean([q['avg_critical_erasure'] for q in qec_data])), 4),
            'avg_cos_10pct': round(float(np.mean([q['avg_cos_10pct'] for q in qec_data])), 4),
            'avg_cos_50pct': round(float(np.mean([q['avg_cos_50pct'] for q in qec_data])), 4),
            'redundancy_profile': [round(float(np.mean([q['redundancy_profile'][i] for q in qec_data])), 4)
                                  for i in range(n)],
            'critical_profile': [round(float(np.mean([q['critical_profile'][i] for q in qec_data])), 4)
                                for i in range(n)],
            'qec_robust': sum(1 for q in qec_data if q['qec_robust']) >= 4,
        }
        robust = 'YES' if all_results[size]['qec_robust'] else 'NO'
        print(f"  Redundancy: {all_results[size]['avg_redundancy']:.4f}")
        print(f"  Critical erasure: {all_results[size]['avg_critical_erasure']:.4f}")
        print(f"  Cos@10%%: {all_results[size]['avg_cos_10pct']:.4f}")
        print(f"  QEC robust: {robust}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['redundancy_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Redundancy')
    axes[0, 0].set_title('(a) SVD Redundancy', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['critical_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Critical fraction')
    axes[0, 1].set_title('(b) Critical Erasure Fraction', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 2].bar(x - w/2, [all_results[s]['avg_cos_10pct'] for s in sizes], w,
                  label='cos@10%', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['avg_cos_50pct'] for s in sizes], w,
                  label='cos@50%', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Cosine After Erasure', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "QUANTUM ERROR CORRECTION\n\n"
    for s in sizes:
        d = all_results[s]
        robust = 'YES' if d['qec_robust'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  redundancy = {d['avg_redundancy']:.3f}\n"
        txt += f"  critical = {d['avg_critical_erasure']:.1%}\n"
        txt += f"  cos@10%% = {d['avg_cos_10pct']:.3f}\n"
        txt += f"  robust: {robust}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 336: Quantum Error Correction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase336_qec')
    plt.close()
    save_results('phase336_qec', {'experiment': 'Quantum Error Correction', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
