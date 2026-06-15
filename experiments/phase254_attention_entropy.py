# -*- coding: utf-8 -*-
"""
Phase 254: Attention Entropy Thermodynamics
=============================================
Measure the entropy of attention patterns at each layer.
Compare attention entropy with output distribution entropy.
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
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
]


def attention_entropy(model, tok, device, model_name):
    """Measure attention entropy at each layer for each head."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    
    all_attn_entropy = []  # [prompt][layer] = mean entropy across heads
    all_output_entropy = []
    all_attn_head_entropy = []  # [prompt][layer][head]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)
        
        attn_ent_l = []
        head_ent_l = []
        out_ent_l = []
        
        # Attention patterns
        if out.attentions:
            for li, attn in enumerate(out.attentions):
                # attn shape: (batch, n_heads, seq_len, seq_len)
                a = attn[0].float()  # (n_heads, seq_len, seq_len)
                # Entropy of attention from last token
                last_attn = a[:, -1, :]  # (n_heads, seq_len)
                head_entropies = []
                for h in range(last_attn.shape[0]):
                    p = last_attn[h]
                    p = p / (p.sum() + 1e-10)
                    ent = -(p * torch.log(p + 1e-10)).sum().item()
                    head_entropies.append(float(ent) if not np.isnan(ent) else 0)
                attn_ent_l.append(float(np.mean(head_entropies)))
                head_ent_l.append(head_entropies)
        
        # Output entropy at each layer
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            out_ent_l.append(float(S) if not np.isnan(S) else 0)
        
        all_attn_entropy.append(attn_ent_l)
        all_output_entropy.append(out_ent_l)
        all_attn_head_entropy.append(head_ent_l)

    # Average
    if all_attn_entropy and all_attn_entropy[0]:
        n_attn = min(len(a) for a in all_attn_entropy)
        mean_attn = [float(np.mean([a[i] for a in all_attn_entropy if i < len(a)])) for i in range(n_attn)]
        std_attn = [float(np.std([a[i] for a in all_attn_entropy if i < len(a)])) for i in range(n_attn)]
    else:
        mean_attn, std_attn = [], []
    
    n_out = min(len(o) for o in all_output_entropy)
    mean_out = [float(np.mean([o[i] for o in all_output_entropy])) for i in range(n_out)]

    # Correlation between attention entropy and output entropy
    if mean_attn and len(mean_attn) >= 3:
        # Align lengths
        m = min(len(mean_attn), len(mean_out) - 1)  # attn has n_layers, out has n_layers+1
        r_ao, p_ao = stats.pearsonr(mean_attn[:m], mean_out[1:m+1])
    else:
        r_ao, p_ao = 0, 1

    # Head specialization: variance across heads at each layer
    if all_attn_head_entropy and all_attn_head_entropy[0]:
        n_hl = min(len(h) for h in all_attn_head_entropy)
        head_var = []
        for li in range(n_hl):
            all_h = []
            for p in range(len(PROMPTS)):
                if li < len(all_attn_head_entropy[p]):
                    all_h.extend(all_attn_head_entropy[p][li])
            head_var.append(float(np.var(all_h)) if all_h else 0)
    else:
        head_var = []

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_attn_entropy': mean_attn,
        'std_attn_entropy': std_attn,
        'mean_output_entropy': mean_out,
        'r_attn_output': float(r_ao),
        'head_variance': head_var,
    }


def main():
    print("=" * 70)
    print("Phase 254: Attention Entropy Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = attention_entropy(model, tok, device, size)
        results[size] = r
        print(f"  Attn entropy: {len(r['mean_attn_entropy'])} layers")
        print(f"  r(attn, output) = {r['r_attn_output']:.3f}")
        if r['mean_attn_entropy']:
            print(f"  Attn S range: {min(r['mean_attn_entropy']):.2f} - {max(r['mean_attn_entropy']):.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Attention entropy profile
    for size, r in results.items():
        if r['mean_attn_entropy']:
            x = range(len(r['mean_attn_entropy']))
            axes[0, 0].plot(x, r['mean_attn_entropy'], '-', color=colors[size], lw=2, label=size)
            axes[0, 0].fill_between(x,
                np.array(r['mean_attn_entropy']) - np.array(r['std_attn_entropy']),
                np.array(r['mean_attn_entropy']) + np.array(r['std_attn_entropy']),
                color=colors[size], alpha=0.15)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Attention Entropy')
    axes[0, 0].set_title('(a) Attention Entropy Profile')
    axes[0, 0].legend(fontsize=8)

    # (b) Output entropy profile
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['mean_output_entropy'])), r['mean_output_entropy'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Output Entropy')
    axes[0, 1].set_title('(b) Output Entropy Profile')
    axes[0, 1].legend(fontsize=8)

    # (c) Attn entropy vs Output entropy
    for size, r in results.items():
        if r['mean_attn_entropy']:
            m = min(len(r['mean_attn_entropy']), len(r['mean_output_entropy'])-1)
            axes[0, 2].scatter(r['mean_attn_entropy'][:m], r['mean_output_entropy'][1:m+1],
                              color=colors[size], s=30, alpha=0.7,
                              label=f'{size} (r={r["r_attn_output"]:.2f})')
    axes[0, 2].set_xlabel('Attention Entropy'); axes[0, 2].set_ylabel('Output Entropy')
    axes[0, 2].set_title('(c) Attn vs Output Entropy')
    axes[0, 2].legend(fontsize=7)

    # (d) Head variance (specialization)
    for size, r in results.items():
        if r['head_variance']:
            axes[1, 0].plot(range(len(r['head_variance'])), r['head_variance'],
                           '-o', color=colors[size], lw=2, markersize=3, label=size)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('Head Entropy Variance')
    axes[1, 0].set_title('(d) Head Specialization')
    axes[1, 0].legend(fontsize=8)

    # (e) Dual entropy comparison
    r15 = results[list(results.keys())[-1]]
    if r15['mean_attn_entropy']:
        n = min(len(r15['mean_attn_entropy']), len(r15['mean_output_entropy'])-1)
        x = range(n)
        axes[1, 1].plot(x, r15['mean_attn_entropy'][:n], '-', color='steelblue', lw=2, label='Attn S')
        axes[1, 1].plot(x, r15['mean_output_entropy'][1:n+1], '-', color='coral', lw=2, label='Output S')
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('(e) Dual Entropy Comparison')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "ATTENTION ENTROPY\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  r(attn, out) = {r['r_attn_output']:.3f}\n"
        if r['mean_attn_entropy']:
            summary += f"  Attn S: {r['mean_attn_entropy'][0]:.2f}"
            summary += f" -> {r['mean_attn_entropy'][-1]:.2f}\n"
        summary += "\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 254: Attention Entropy Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase254_attention_entropy')
    plt.close()
    save_results('phase254_attention_entropy', {
        'experiment': 'Attention Entropy',
        'results': results,
    })


if __name__ == '__main__':
    main()
