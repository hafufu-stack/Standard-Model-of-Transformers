# -*- coding: utf-8 -*-
"""
Phase 283: ER=EPR Cross-Validation
=====================================
S-Qubit Q305 tested ER=EPR (wormholes = entanglement), finding avg effect=0.411.
Standard Model P273 found Area Law (R2=0.987).

Cross-validation: measure both holographic entropy AND wormhole shortcuts
on the same model and prompts to test if they correlate.
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
_SNAP_0B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                         "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")
_SNAP_1B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")

PROMPTS = [
    "The cat sat on the mat",
    "Quantum mechanics describes particles as waves",
    "The stock market crashed in two thousand and eight",
    "Neural networks learn by adjusting weights through backpropagation",
    "The theory of evolution explains the diversity of life on Earth",
    "Water molecules consist of two hydrogen atoms and one oxygen atom",
    "The speed of light is constant in all reference frames",
    "Machine learning algorithms can classify images with high accuracy",
]


def load_eager(path, device):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map=device,
        local_files_only=True, attn_implementation='eager',
    )
    model.eval()
    return model, tok


def measure_holographic_entropy(model, tok, prompt, device):
    """Measure SVD entropy from cross-attention (Area Law metric)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    split = seq_len // 2
    if split < 2:
        return 0.0

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    entropies = []
    for attn in out.attentions:
        a = attn[0].float()
        cross = a[:, split:, :split].mean(dim=0).cpu().numpy()
        try:
            _, s, _ = np.linalg.svd(cross, full_matrices=False)
            s_n = s / (s.sum() + 1e-10)
            s_n = s_n[s_n > 1e-10]
            ent = -float((s_n * np.log(s_n)).sum())
        except Exception:
            ent = 0.0
        entropies.append(ent)
    return float(np.mean(entropies))


def measure_wormhole_effect(model, tok, prompt, device):
    """Measure attention shortcuts (wormhole effect).
    A wormhole exists when attention from a late layer connects
    early-position tokens directly to the last token, bypassing
    the sequential propagation.
    """
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    if seq_len < 4:
        return 0.0

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    # Measure: does attention at later layers reach back to position 0?
    wormhole_scores = []
    n_layers = len(out.attentions)
    for li in range(n_layers // 2, n_layers):  # only look at later layers
        attn = out.attentions[li][0].float()  # (heads, seq, seq)
        # Attention from last token to first token
        attn_to_first = attn[:, -1, 0].mean().item()
        # Attention from last token to nearby tokens (expected)
        nearby_range = max(1, seq_len // 4)
        attn_to_nearby = attn[:, -1, -nearby_range:].mean().item()
        # Wormhole = long-range attention / short-range attention
        wormhole = attn_to_first / (attn_to_nearby + 1e-10)
        wormhole_scores.append(wormhole)

    return float(np.mean(wormhole_scores))


def measure_info_shortcut(model, tok, prompt, device):
    """Measure information shortcut via hidden state similarity jump.
    If information takes a shortcut (wormhole), we'd see the hidden state
    at the last position suddenly become similar to early-position states.
    """
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Track similarity between position 0 and last position across layers
    sims = []
    for hs in out.hidden_states:
        h = hs[0].float()
        if h.shape[0] < 2:
            sims.append(0.0)
            continue
        sim = torch.nn.functional.cosine_similarity(h[0:1], h[-1:], dim=-1).item()
        sims.append(sim)

    # Shortcut = sudden increase in similarity
    jumps = [sims[i+1] - sims[i] for i in range(len(sims)-1)]
    max_jump = float(max(jumps)) if jumps else 0.0
    return max_jump, sims


def main():
    print("=" * 70)
    print("Phase 283: ER=EPR Cross-Validation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for name, path in [('0.5B', _SNAP_0B5), ('1.5B', _SNAP_1B5)]:
        print(f"\n=== {name} ===")
        model, tok = load_eager(path, device)

        prompt_data = []
        for prompt in PROMPTS:
            holo_s = measure_holographic_entropy(model, tok, prompt, device)
            worm_s = measure_wormhole_effect(model, tok, prompt, device)
            shortcut, sims = measure_info_shortcut(model, tok, prompt, device)
            prompt_data.append({
                'prompt': prompt[:50],
                'holographic_entropy': round(holo_s, 4),
                'wormhole_effect': round(worm_s, 4),
                'info_shortcut': round(shortcut, 4),
                'sim_profile': [round(s, 4) for s in sims],
            })
            print(f"  S={holo_s:.3f}, Worm={worm_s:.3f}, Shortcut={shortcut:.3f}")

        # Correlations
        holo_arr = np.array([d['holographic_entropy'] for d in prompt_data])
        worm_arr = np.array([d['wormhole_effect'] for d in prompt_data])
        short_arr = np.array([d['info_shortcut'] for d in prompt_data])

        r_hw, p_hw = stats.pearsonr(holo_arr, worm_arr)
        r_hs, p_hs = stats.pearsonr(holo_arr, short_arr)
        r_ws, p_ws = stats.pearsonr(worm_arr, short_arr)

        all_results[name] = {
            'prompt_data': prompt_data,
            'corr_holo_worm': {'r': round(float(r_hw), 4), 'p': round(float(p_hw), 6)},
            'corr_holo_short': {'r': round(float(r_hs), 4), 'p': round(float(p_hs), 6)},
            'corr_worm_short': {'r': round(float(r_ws), 4), 'p': round(float(p_ws), 6)},
            'er_epr_consistent': abs(r_hw) > 0.3,
        }
        print(f"  Corr(holo, worm) = {r_hw:.3f} (p={p_hw:.4f})")
        print(f"  ER=EPR {'CONSISTENT' if abs(r_hw) > 0.3 else 'WEAK'}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Holographic entropy vs wormhole effect
    for name, data in all_results.items():
        h = [d['holographic_entropy'] for d in data['prompt_data']]
        w = [d['wormhole_effect'] for d in data['prompt_data']]
        axes[0, 0].scatter(h, w, c=colors[name], s=60, label=name, alpha=0.7)
    axes[0, 0].set_xlabel('Holographic Entropy')
    axes[0, 0].set_ylabel('Wormhole Effect')
    axes[0, 0].set_title('(a) ER=EPR: Entropy vs Wormholes', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Info shortcut profiles
    for name, data in all_results.items():
        for pd in data['prompt_data'][:3]:
            axes[0, 1].plot(pd['sim_profile'], '-', color=colors[name], alpha=0.4, lw=1)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Cosine Similarity (pos 0 <-> last)')
    axes[0, 1].set_title('(b) Information Shortcut Profiles', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # (c) Correlation matrix
    for i, (name, data) in enumerate(all_results.items()):
        metrics = ['Holo-Worm', 'Holo-Short', 'Worm-Short']
        vals = [data['corr_holo_worm']['r'], data['corr_holo_short']['r'],
                data['corr_worm_short']['r']]
        x = np.arange(3) + i * 0.35
        axes[0, 2].bar(x, vals, 0.3, color=colors[name], label=name)
    axes[0, 2].set_xticks(np.arange(3) + 0.15)
    axes[0, 2].set_xticklabels(['Holo-Worm', 'Holo-Short', 'Worm-Short'])
    axes[0, 2].set_ylabel('Pearson r')
    axes[0, 2].set_title('(c) Cross-Metric Correlations', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d-f)
    for idx in range(3):
        axes[1, idx].axis('off')

    txt = "ER=EPR CROSS-VALIDATION\n\n"
    txt += "Holographic entropy (P273 Area Law)\n"
    txt += "  x  Wormhole shortcuts (Q305 ER=EPR)\n\n"
    for name, data in all_results.items():
        txt += f"{name}:\n"
        txt += f"  r(holo,worm)  = {data['corr_holo_worm']['r']:.3f}\n"
        txt += f"  r(holo,short) = {data['corr_holo_short']['r']:.3f}\n"
        txt += f"  ER=EPR: {'YES' if data['er_epr_consistent'] else 'NO'}\n\n"
    axes[1, 1].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')

    fig.suptitle("Phase 283: ER=EPR Cross-Validation",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase283_er_epr')
    plt.close()

    save_results('phase283_er_epr', {
        'experiment': 'ER=EPR Cross-Validation',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
