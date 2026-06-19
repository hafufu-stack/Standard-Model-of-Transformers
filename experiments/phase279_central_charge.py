# -*- coding: utf-8 -*-
"""
Phase 279: Central Charge from Area Law
==========================================
Phase 273 confirmed S ~ log(L) (Area Law) with R^2=0.987.
In 1+1D CFT, S = (c/3) * log(L) + const, where c is the central charge.

Extract c from the slope and test:
  - Does c relate to n_heads or hidden_dim?
  - Is c invariant across model sizes?
  - Does c match known universality classes?
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

_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_SNAP_1B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
_SNAP_0B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                         "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")
_SNAP_7B = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-7B-Instruct",
                        "snapshots", "a09a35458c702b33eeacc393d103063234e8bc28")

MODELS = [
    ('0.5B', _SNAP_0B5, 896, 14, 24),   # hidden, n_heads, n_layers
    ('1.5B', _SNAP_1B5, 1536, 12, 28),
]

# Prompts of increasing length for precise slope measurement
PROMPTS = [
    "The cat sat",
    "The cat sat on the mat",
    "The cat sat on the mat and looked around the room",
    "The cat sat on the mat and looked around the room with curiosity at the strange objects",
    "The fundamental theorem of calculus states that differentiation and integration are inverse operations that connect the concepts of rate of change and accumulated quantity in mathematics",
    "In theoretical physics the concept of gauge symmetry plays a fundamental role in our understanding of the electromagnetic weak and strong nuclear forces these symmetries constrain the form of interactions between elementary particles and lead to the prediction of gauge bosons which mediate the fundamental forces of nature",
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world from the ancient Babylonians who developed a base sixty number system to the ancient Greeks who introduced rigorous mathematical proof to the Islamic Golden Age scholars who preserved and extended Greek mathematical knowledge and developed algebra to the European Renaissance mathematicians who laid the foundations for calculus and modern analysis",
]


def load_eager(path, device):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map=device,
        local_files_only=True, attn_implementation='eager',
    )
    model.eval()
    return model, tok


def measure_entanglement_entropy(model, tok, prompt, device):
    """Measure SVD entropy of cross-attention at the midpoint split."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    split = seq_len // 2
    if split < 2 or split >= seq_len - 1:
        return None

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    entropies_per_layer = []
    for attn in out.attentions:
        a = attn[0].float()  # (heads, seq, seq)
        cross = a[:, split:, :split].mean(dim=0).cpu().numpy()
        try:
            _, s, _ = np.linalg.svd(cross, full_matrices=False)
            s_n = s / (s.sum() + 1e-10)
            s_n = s_n[s_n > 1e-10]
            ent = -float((s_n * np.log(s_n)).sum())
        except Exception:
            ent = 0.0
        entropies_per_layer.append(ent)

    return {
        'seq_len': seq_len,
        'log_seq_len': float(np.log(seq_len)),
        'mean_entropy': float(np.mean(entropies_per_layer)),
        'max_entropy': float(np.max(entropies_per_layer)),
        'entropies': entropies_per_layer,
    }


def main():
    print("=" * 70)
    print("Phase 279: Central Charge from Area Law")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for name, path, hidden, n_heads, n_layers in MODELS:
        print(f"\n=== {name} (h={hidden}, heads={n_heads}, layers={n_layers}) ===")
        model, tok = load_eager(path, device)

        data_points = []
        for prompt in PROMPTS:
            r = measure_entanglement_entropy(model, tok, prompt, device)
            if r:
                data_points.append(r)
                print(f"  L={r['seq_len']:3d}, logL={r['log_seq_len']:.2f}, "
                      f"S={r['mean_entropy']:.4f}")

        # Fit S = (c/3) * log(L) + const
        log_ls = np.array([d['log_seq_len'] for d in data_points])
        entropies = np.array([d['mean_entropy'] for d in data_points])

        slope, intercept, r_val, p_val, _ = stats.linregress(log_ls, entropies)
        central_charge = 3 * slope  # c = 3 * (dS/d(logL))

        # Also fit with max entropy
        max_ents = np.array([d['max_entropy'] for d in data_points])
        slope_max, _, r_max, _, _ = stats.linregress(log_ls, max_ents)
        c_max = 3 * slope_max

        all_results[name] = {
            'hidden_dim': hidden,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'data_points': data_points,
            'central_charge_mean': round(float(central_charge), 4),
            'central_charge_max': round(float(c_max), 4),
            'R2': round(float(r_val**2), 4),
            'R2_max': round(float(r_max**2), 4),
            'slope': round(float(slope), 6),
            'intercept': round(float(intercept), 4),
            'c_over_heads': round(float(central_charge / n_heads), 4),
            'c_over_hidden': round(float(central_charge / hidden), 6),
        }

        print(f"  Central charge c = {central_charge:.4f} (R2={r_val**2:.4f})")
        print(f"  c/n_heads = {central_charge/n_heads:.4f}")
        print(f"  c_max = {c_max:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) S vs log(L) with fit lines
    for name, data in all_results.items():
        dp = data['data_points']
        x = [d['log_seq_len'] for d in dp]
        y = [d['mean_entropy'] for d in dp]
        axes[0, 0].scatter(x, y, c=colors.get(name, '#999'), s=60, zorder=5)
        x_fit = np.linspace(min(x), max(x), 50)
        y_fit = data['slope'] * x_fit + data['intercept']
        axes[0, 0].plot(x_fit, y_fit, '--', color=colors.get(name, '#999'),
                       label=f"{name}: c={data['central_charge_mean']:.2f}")
    axes[0, 0].set_xlabel('log(L)')
    axes[0, 0].set_ylabel('Entanglement Entropy S')
    axes[0, 0].set_title('(a) S = (c/3)*log(L) + const', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Central charge vs model size
    sizes = [data['hidden_dim'] for data in all_results.values()]
    charges = [data['central_charge_mean'] for data in all_results.values()]
    axes[0, 1].bar(list(all_results.keys()), charges,
                  color=[colors.get(n, '#999') for n in all_results.keys()])
    axes[0, 1].set_ylabel('Central Charge c')
    axes[0, 1].set_title('(b) Central Charge vs Model Size', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # (c) c/n_heads ratio
    ratios = [data['c_over_heads'] for data in all_results.values()]
    axes[0, 2].bar(list(all_results.keys()), ratios,
                  color=[colors.get(n, '#999') for n in all_results.keys()])
    axes[0, 2].set_ylabel('c / n_heads')
    axes[0, 2].set_title('(c) Normalized Central Charge', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # (d) Entropy profile per layer (longest prompt)
    for name, data in all_results.items():
        dp = data['data_points']
        longest = max(dp, key=lambda d: d['seq_len'])
        axes[1, 0].plot(longest['entropies'], '-', color=colors.get(name, '#999'),
                       lw=2, label=name)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Entanglement Entropy')
    axes[1, 0].set_title('(d) Entropy per Layer (longest prompt)', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) R^2 comparison
    r2s = [data['R2'] for data in all_results.values()]
    axes[1, 1].bar(list(all_results.keys()), r2s,
                  color=[colors.get(n, '#999') for n in all_results.keys()])
    axes[1, 1].set_ylabel('R^2')
    axes[1, 1].set_ylim(0.9, 1.0)
    axes[1, 1].set_title('(e) Area Law Fit Quality', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "CENTRAL CHARGE EXTRACTION\n"
    txt += "S = (c/3) * log(L) + const\n\n"
    for name, data in all_results.items():
        txt += f"{name}:\n"
        txt += f"  c = {data['central_charge_mean']:.3f}\n"
        txt += f"  R2 = {data['R2']:.4f}\n"
        txt += f"  c/heads = {data['c_over_heads']:.3f}\n\n"
    known = "Known CFT:\n  Ising: c=0.5\n  Free boson: c=1\n  Free fermion: c=0.5"
    txt += known
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 279: Central Charge from Area Law",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase279_central_charge')
    plt.close()

    save_results('phase279_central_charge', {
        'experiment': 'Central Charge from Area Law',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
