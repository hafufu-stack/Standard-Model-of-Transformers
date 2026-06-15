# -*- coding: utf-8 -*-
"""
Phase 231: Attention Entropy Thermodynamics
=============================================
Measure the entropy of attention patterns at each layer.
Attention as a heat engine: how much information does each head extract?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
]


def attention_thermodynamics(model, tok, device, model_name):
    """Measure attention entropy at each layer and head."""
    n_layers = len(model.model.layers)

    all_attn_entropy = []  # [prompt][layer] = list of per-head entropies
    all_logit_T = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)

        attn_entropy_layers = []
        for li, attn in enumerate(out.attentions):
            # attn: (batch, n_heads, seq, seq)
            # Last token's attention to all previous tokens
            a = attn[0, :, -1, :].float()  # (n_heads, seq)
            # Entropy per head
            head_entropies = []
            for hi in range(a.shape[0]):
                p = a[hi]
                p = p / (p.sum() + 1e-10)
                ent = -(p * torch.log(p + 1e-10)).sum().item()
                head_entropies.append(float(ent) if not np.isnan(ent) else 0)
            attn_entropy_layers.append(head_entropies)

        all_attn_entropy.append(attn_entropy_layers)

        # Logit temperature
        norm_layer = model.model.norm
        lm_head = model.lm_head
        T_list = []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(float(S) if not np.isnan(S) else 0)
        all_logit_T.append(T_list)

    # Aggregate
    n_attn_layers = min(len(a) for a in all_attn_entropy)
    n_heads = len(all_attn_entropy[0][0]) if all_attn_entropy else 0

    # Mean attention entropy per layer (averaged over heads and prompts)
    mean_attn_entropy_per_layer = []
    std_attn_entropy_per_layer = []
    mean_attn_entropy_per_head = []  # [layer][head]

    for l in range(n_attn_layers):
        all_ent = []
        head_ents = [[] for _ in range(n_heads)]
        for p in range(len(PROMPTS)):
            if l < len(all_attn_entropy[p]):
                for hi, e in enumerate(all_attn_entropy[p][l]):
                    all_ent.append(e)
                    if hi < n_heads:
                        head_ents[hi].append(e)
        mean_attn_entropy_per_layer.append(float(np.mean(all_ent)))
        std_attn_entropy_per_layer.append(float(np.std(all_ent)))
        mean_attn_entropy_per_head.append([float(np.mean(h)) for h in head_ents])

    # Logit T
    n_states = min(len(t) for t in all_logit_T)
    mean_logit_T = [float(np.mean([all_logit_T[p][l] for p in range(len(PROMPTS))]))
                    for l in range(n_states)]

    # Correlation between attention entropy and logit temperature
    min_len = min(len(mean_attn_entropy_per_layer), len(mean_logit_T) - 1)
    if min_len > 2:
        r_attn_T, p_attn_T = stats.pearsonr(mean_attn_entropy_per_layer[:min_len],
                                              mean_logit_T[1:min_len+1])
    else:
        r_attn_T, p_attn_T = 0, 1

    # Head specialization: variance of entropy across heads at each layer
    head_specialization = []
    for l in range(n_attn_layers):
        head_means = mean_attn_entropy_per_head[l]
        head_specialization.append(float(np.std(head_means)))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'mean_attn_entropy': mean_attn_entropy_per_layer,
        'std_attn_entropy': std_attn_entropy_per_layer,
        'mean_logit_T': mean_logit_T,
        'head_specialization': head_specialization,
        'r_attn_T': float(r_attn_T),
        'p_attn_T': float(p_attn_T),
        'mean_attn_entropy_per_head': mean_attn_entropy_per_head,
    }


def main():
    print("=" * 70)
    print("Phase 231: Attention Entropy Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        # Need eager attention for output_attentions
        # Re-load with eager
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        from utils import _SNAP_0B5, _SNAP_1B5
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mid = _SNAP_0B5 if size == '0.5B' else _SNAP_1B5
        dtype = torch.float16 if device == 'cuda' else torch.float32
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=dtype, device_map=device, local_files_only=True,
            attn_implementation='eager',
        )
        model.eval()

        r = attention_thermodynamics(model, tok, device, size)
        results[size] = r
        print(f"  {r['n_heads']} heads, {r['n_layers']} layers")
        print(f"  r(attn_entropy, T) = {r['r_attn_T']:.4f} (p={r['p_attn_T']:.2e})")
        del model, tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Mean attention entropy per layer
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['mean_attn_entropy'])), r['mean_attn_entropy'],
                       '-', color=colors[size], lw=2, label=size)
        axes[0, 0].fill_between(range(len(r['mean_attn_entropy'])),
                                [m-s for m,s in zip(r['mean_attn_entropy'], r['std_attn_entropy'])],
                                [m+s for m,s in zip(r['mean_attn_entropy'], r['std_attn_entropy'])],
                                color=colors[size], alpha=0.1)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Attention Entropy')
    axes[0, 0].set_title('(a) Attention Entropy vs Depth')
    axes[0, 0].legend(fontsize=8)

    # (b) Logit T vs Attention Entropy
    for size, r in results.items():
        min_len = min(len(r['mean_attn_entropy']), len(r['mean_logit_T'])-1)
        axes[0, 1].scatter(r['mean_attn_entropy'][:min_len],
                          r['mean_logit_T'][1:min_len+1],
                          color=colors[size], s=30, alpha=0.7, label=size)
    axes[0, 1].set_xlabel('Attention Entropy')
    axes[0, 1].set_ylabel('Logit Temperature')
    axes[0, 1].set_title('(b) Attn Entropy vs Logit T')
    axes[0, 1].legend(fontsize=8)

    # (c) Head specialization
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['head_specialization'])), r['head_specialization'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Head Entropy Std')
    axes[0, 2].set_title('(c) Head Specialization')
    axes[0, 2].legend(fontsize=8)

    # (d) Heatmap of per-head entropy (0.5B)
    if '0.5B' in results:
        r05 = results['0.5B']
        head_data = np.array(r05['mean_attn_entropy_per_head'])
        im = axes[1, 0].imshow(head_data.T, aspect='auto', cmap='viridis')
        axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('Head')
        axes[1, 0].set_title('(d) Head Entropy Map (0.5B)')
        fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # (e) Heatmap of per-head entropy (1.5B)
    if '1.5B' in results:
        r15 = results['1.5B']
        head_data = np.array(r15['mean_attn_entropy_per_head'])
        im = axes[1, 1].imshow(head_data.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('Head')
        axes[1, 1].set_title('(e) Head Entropy Map (1.5B)')
        fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    # (f) Summary
    summary = "Attention Thermodynamics\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Heads: {r['n_heads']}\n"
        summary += f"  r(Attn,T) = {r['r_attn_T']:.3f}\n"
        summary += f"  p = {r['p_attn_T']:.2e}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 231: Attention Entropy Thermodynamics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase231_attention')
    plt.close()
    save_results('phase231_attention', {'experiment': 'Attention Thermodynamics', 'results': results})


if __name__ == '__main__':
    main()
