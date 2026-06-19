# -*- coding: utf-8 -*-
"""
Phase 273: Holographic Entanglement (Ryu-Takayanagi)
======================================================
In AdS/CFT, the entanglement entropy of a boundary region scales with
the area of the minimal surface, not the volume (Area Law).

Test: Does attention-based mutual information between prompt halves
scale with hidden_dim (area) or seq_len (volume)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

# Local snapshot paths
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_SNAP_1B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
_SNAP_0B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                         "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")


def load_model_eager(device, size='1.5B'):
    """Load model with attn_implementation='eager' for attention output support."""
    mid = _SNAP_0B5 if size == '0.5B' else (_SNAP_1B5 if os.path.exists(_SNAP_1B5) else _SNAP_0B5)
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    # Use float32 + eager to avoid NaN with attention outputs
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float32, device_map=device,
        local_files_only=True, attn_implementation='eager',
    )
    model.eval()
    return model, tok

# Prompts of varying length to test scaling
PROMPTS = {
    'short': "The cat sat on the mat and looked around",
    'medium': "The fundamental theorem of calculus states that differentiation and integration are inverse operations that connect the concepts of rate of change and accumulated quantity in mathematics",
    'long': "In theoretical physics the concept of gauge symmetry plays a fundamental role in our understanding of the electromagnetic weak and strong nuclear forces these symmetries constrain the form of interactions between elementary particles and lead to the prediction of gauge bosons which mediate the fundamental forces of nature including photons W and Z bosons and gluons",
}


def compute_attention_entanglement(model, tok, prompt, device, split_ratio=0.5):
    """Compute entanglement entropy between first and second half of prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    split_pos = int(seq_len * split_ratio)

    if split_pos < 2 or split_pos >= seq_len - 1:
        return None

    # Forward pass with attention outputs
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, output_attentions=True)

    n_layers = len(out.attentions)
    hidden_dim = out.hidden_states[-1].shape[-1]

    layer_data = []
    for li in range(n_layers):
        attn = out.attentions[li][0].float()  # (n_heads, seq, seq)
        n_heads = attn.shape[0]

        # Attention from region B (second half) to region A (first half)
        # This is the cross-attention that creates entanglement
        attn_BA = attn[:, split_pos:, :split_pos]  # (heads, B_len, A_len)

        # Compute entanglement entropy from singular values of cross-attention
        # SVD of the cross-attention matrix (averaged over heads)
        cross_mean = attn_BA.mean(dim=0).cpu().numpy()  # (B_len, A_len)

        # SVD
        try:
            U, s, Vt = np.linalg.svd(cross_mean, full_matrices=False)
            # Normalize singular values as probabilities
            s_norm = s / (s.sum() + 1e-10)
            s_norm = s_norm[s_norm > 1e-10]
            entropy = -float((s_norm * np.log(s_norm)).sum())
        except Exception:
            entropy = 0.0

        # Also compute mutual information via attention weights
        # P(A|B) = attention weights from B positions to A positions
        total_attn_to_A = attn_BA.sum().item()
        total_attn_from_B = attn[:, split_pos:, :].sum().item()
        mi_proxy = total_attn_to_A / (total_attn_from_B + 1e-10)

        layer_data.append({
            'layer': li,
            'entropy': round(entropy, 6),
            'mi_proxy': round(mi_proxy, 6),
            'n_singular_values': len(s_norm) if 's_norm' in dir() else 0,
        })

    return {
        'seq_len': seq_len,
        'split_pos': split_pos,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'region_A_len': split_pos,
        'region_B_len': seq_len - split_pos,
        'layer_data': layer_data,
        'mean_entropy': round(float(np.mean([d['entropy'] for d in layer_data])), 6),
    }


def main():
    print("=" * 70)
    print("Phase 273: Holographic Entanglement (Ryu-Takayanagi)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model_eager(device, size=size)

        size_results = {}
        for length_name, prompt in PROMPTS.items():
            print(f"  {length_name}: {len(prompt)} chars")
            r = compute_attention_entanglement(model, tok, prompt, device)
            if r:
                size_results[length_name] = r
                print(f"    seq_len={r['seq_len']}, hidden_dim={r['hidden_dim']}, "
                      f"mean_entropy={r['mean_entropy']:.4f}")

        # Test different split ratios on the long prompt
        split_results = []
        for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            r = compute_attention_entanglement(model, tok, PROMPTS['long'], device,
                                              split_ratio=ratio)
            if r:
                split_results.append({
                    'ratio': ratio,
                    'region_A_len': r['region_A_len'],
                    'mean_entropy': r['mean_entropy'],
                })
        size_results['split_scaling'] = split_results

        all_results[size] = size_results

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Area Law vs Volume Law Analysis ===
    # If entropy ~ log(seq_len), it's area law in 1+1D CFT
    # If entropy ~ seq_len, it's volume law
    scaling_analysis = {}
    for size in all_results:
        lengths = []
        entropies = []
        for name in ['short', 'medium', 'long']:
            if name in all_results[size]:
                r = all_results[size][name]
                lengths.append(r['seq_len'])
                entropies.append(r['mean_entropy'])
        if len(lengths) >= 2:
            # Test log scaling (area law)
            log_lengths = np.log(lengths)
            slope_log, _, r_log, _, _ = stats.linregress(log_lengths, entropies)
            # Test linear scaling (volume law)
            slope_lin, _, r_lin, _, _ = stats.linregress(lengths, entropies)
            scaling_analysis[size] = {
                'r2_log': round(r_log**2, 4),
                'r2_linear': round(r_lin**2, 4),
                'area_law': r_log**2 > r_lin**2,
                'slope_log': round(slope_log, 4),
                'slope_linear': round(slope_lin, 6),
            }
            print(f"\n  {size} Scaling: log R2={r_log**2:.4f}, linear R2={r_lin**2:.4f}")
            print(f"    -> {'AREA LAW' if r_log**2 > r_lin**2 else 'VOLUME LAW'}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Entropy per layer (long prompt)
    for size in all_results:
        if 'long' in all_results[size]:
            ld = all_results[size]['long']['layer_data']
            axes[0, 0].plot([d['layer'] for d in ld], [d['entropy'] for d in ld],
                           '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Entanglement Entropy')
    axes[0, 0].set_title('(a) Entanglement Entropy per Layer', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Entropy vs seq_len (area vs volume)
    for size in all_results:
        ls, es = [], []
        for name in ['short', 'medium', 'long']:
            if name in all_results[size]:
                ls.append(all_results[size][name]['seq_len'])
                es.append(all_results[size][name]['mean_entropy'])
        axes[0, 1].plot(ls, es, 'o-', color=colors[size], lw=2, markersize=8,
                       label=size)
    axes[0, 1].set_xlabel('Sequence Length (Volume)')
    axes[0, 1].set_ylabel('Mean Entanglement Entropy')
    axes[0, 1].set_title('(b) Entropy Scaling: Area vs Volume?', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Entropy vs log(seq_len)
    for size in all_results:
        ls, es = [], []
        for name in ['short', 'medium', 'long']:
            if name in all_results[size]:
                ls.append(np.log(all_results[size][name]['seq_len']))
                es.append(all_results[size][name]['mean_entropy'])
        axes[0, 2].plot(ls, es, 'o-', color=colors[size], lw=2, markersize=8,
                       label=size)
    axes[0, 2].set_xlabel('log(Sequence Length)')
    axes[0, 2].set_ylabel('Mean Entanglement Entropy')
    axes[0, 2].set_title('(c) Log Scaling Test (Area Law)', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Split ratio vs entropy
    for size in all_results:
        if 'split_scaling' in all_results[size]:
            sr = all_results[size]['split_scaling']
            axes[1, 0].plot([d['ratio'] for d in sr], [d['mean_entropy'] for d in sr],
                           'o-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Split Ratio')
    axes[1, 0].set_ylabel('Entanglement Entropy')
    axes[1, 0].set_title('(d) Entropy vs Partition Size', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) MI proxy per layer
    for size in all_results:
        if 'long' in all_results[size]:
            ld = all_results[size]['long']['layer_data']
            axes[1, 1].plot([d['layer'] for d in ld], [d['mi_proxy'] for d in ld],
                           '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('MI Proxy (cross-attention ratio)')
    axes[1, 1].set_title('(e) Mutual Information per Layer', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "HOLOGRAPHIC ENTANGLEMENT\n\n"
    for size, sa in scaling_analysis.items():
        summary += f"{size}:\n"
        summary += f"  log R2: {sa['r2_log']:.4f}\n"
        summary += f"  lin R2: {sa['r2_linear']:.4f}\n"
        summary += f"  -> {'AREA LAW' if sa['area_law'] else 'VOLUME LAW'}\n\n"
    summary += "Area Law = holographic\nVolume Law = non-holographic"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 273: Holographic Entanglement (Ryu-Takayanagi)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase273_holographic')
    plt.close()

    save_results('phase273_holographic', {
        'experiment': 'Holographic Entanglement (Ryu-Takayanagi)',
        'scaling_analysis': scaling_analysis,
        'results': {k: {kk: vv for kk, vv in v.items()
                        if kk != 'layer_data' or kk == 'layer_data'}
                   for k, v in all_results.items()},
    })


if __name__ == '__main__':
    main()
