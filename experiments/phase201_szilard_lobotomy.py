# -*- coding: utf-8 -*-
"""
Phase 201: Szilard Engine Lobotomy
====================================
Identify the 8 attention heads that act as "Szilard engines" (highest
information gain + positive work) from Phase 196, then causally test
their importance by masking them vs masking random heads.

Compare: eta, entropy change, and output quality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import save_results, save_figure, RESULTS_DIR

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
]

L0 = 21


def _load_eager_model(device):
    """Load model with eager attention for output_attentions=True."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    _SNAP = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
    tok = AutoTokenizer.from_pretrained(_SNAP, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        _SNAP, torch_dtype=torch.float16, device_map=device,
        local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    return model, tok


def identify_szilard_heads(model, tok, device, n_heads):
    """Re-run Szilard analysis to identify the top-8 engine heads."""
    n_layers = len(model.model.layers)
    all_info = np.zeros((n_layers, n_heads))
    all_work = np.zeros((n_layers, n_heads))

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)
        for li in range(n_layers):
            attn = out.attentions[li][0].float()
            for hi in range(n_heads):
                a = attn[hi, -1, :]
                ent = -(a * torch.log(a + 1e-10)).sum().item()
                ig = np.log(a.shape[0]) - ent
                if np.isnan(ig):
                    ig = 0
                all_info[li, hi] += ig

            h_in = out.hidden_states[li][0, -1, :].float()
            h_out = out.hidden_states[li + 1][0, -1, :].float()
            dU = (h_out.norm() - h_in.norm()).item()
            total_ig = all_info[li, :].sum() + 1e-10
            for hi in range(n_heads):
                all_work[li, hi] += dU * (all_info[li, hi] / total_ig)

    all_info /= len(PROMPTS)
    all_work /= len(PROMPTS)

    # Score: heads with high info AND positive work
    scores = all_info * np.maximum(all_work, 0)
    # Flatten, find top-8
    flat_idx = np.argsort(scores.flatten())[::-1][:8]
    szilard_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in flat_idx]
    return szilard_heads, all_info, all_work


def measure_with_mask(model, tok, device, masked_heads, n_heads):
    """Measure thermodynamics with specific heads masked (zeroed)."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Create mask hooks
    mask_set = set(masked_heads)
    hooks = []

    def make_attn_mask_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # output is a tuple: (attn_output, attn_weights, past_kv)
            if isinstance(output, tuple) and len(output) >= 1:
                attn_out = output[0]  # (batch, seq, hidden)
                # Zero out the masked heads' contribution
                head_dim = attn_out.shape[-1] // n_heads
                for hi in range(n_heads):
                    if (layer_idx, hi) in mask_set:
                        start = hi * head_dim
                        end = (hi + 1) * head_dim
                        attn_out[:, :, start:end] = 0
                return (attn_out,) + output[1:]
            return output
        return hook

    for li in range(n_layers):
        has_masked = any((li, hi) in mask_set for hi in range(n_heads))
        if has_masked:
            h = model.model.layers[li].self_attn.register_forward_hook(
                make_attn_mask_hook(li), with_kwargs=True
            )
            hooks.append(h)

    # Measure thermodynamics
    all_U = []
    all_S = []
    all_T = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_list, S_list, T_list = [], [], []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_list.append(-(h_prob * torch.log(h_prob + 1e-10)).sum().item())

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T_val if not np.isnan(T_val) else 0)

        all_U.append(U_list)
        all_S.append(S_list)
        all_T.append(T_list)

    # Remove hooks
    for h in hooks:
        h.remove()

    U_mean = np.mean(all_U, axis=0).tolist()
    S_mean = np.mean(all_S, axis=0).tolist()
    T_mean = np.mean(all_T, axis=0).tolist()

    # Calculate eta
    T_hot = max(T_mean[1:])
    T_cold = min(T_mean[1:])
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0
    dS = S_mean[-1] - S_mean[0]

    return {'U': U_mean, 'S': S_mean, 'T': T_mean,
            'eta': eta, 'dS': dS, 'T_hot': T_hot, 'T_cold': T_cold}


def main():
    print("=" * 70)
    print("Phase 201: Szilard Engine Lobotomy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = _load_eager_model(device)
    n_heads = model.config.num_attention_heads
    n_layers = len(model.model.layers)

    # Step 1: Identify Szilard heads
    print("\n[1/4] Identifying Szilard engine heads...")
    szilard_heads, info_map, work_map = identify_szilard_heads(model, tok, device, n_heads)
    print(f"  Top-8 Szilard heads (layer, head): {szilard_heads}")

    # Step 2: Baseline (no masking)
    print("\n[2/4] Baseline measurement...")
    baseline = measure_with_mask(model, tok, device, [], n_heads)
    print(f"  Baseline eta: {baseline['eta']:.4f}, dS: {baseline['dS']:.4f}")

    # Step 3: Mask Szilard heads
    print("\n[3/4] Masking Szilard engine heads...")
    szilard_masked = measure_with_mask(model, tok, device, szilard_heads, n_heads)
    print(f"  Szilard-masked eta: {szilard_masked['eta']:.4f}, dS: {szilard_masked['dS']:.4f}")

    # Step 4: Mask random heads (control)
    print("\n[4/4] Masking random heads (control)...")
    np.random.seed(42)
    random_heads = []
    while len(random_heads) < 8:
        rl = np.random.randint(0, n_layers)
        rh = np.random.randint(0, n_heads)
        if (rl, rh) not in random_heads and (rl, rh) not in szilard_heads:
            random_heads.append((rl, rh))
    random_masked = measure_with_mask(model, tok, device, random_heads, n_heads)
    print(f"  Random-masked eta: {random_masked['eta']:.4f}, dS: {random_masked['dS']:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    layers = np.arange(len(baseline['U']))

    # (a) U comparison
    for data, label, color in [(baseline, 'Baseline', '#2ecc71'),
                                (szilard_masked, 'Szilard Masked', '#e74c3c'),
                                (random_masked, 'Random Masked', '#3498db')]:
        axes[0, 0].plot(layers, data['U'], 'o-', label=label, color=color,
                        markersize=3, linewidth=1.5)
    axes[0, 0].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Internal Energy U')
    axes[0, 0].set_title('(a) Internal Energy')
    axes[0, 0].legend(fontsize=8)

    # (b) S comparison
    for data, label, color in [(baseline, 'Baseline', '#2ecc71'),
                                (szilard_masked, 'Szilard Masked', '#e74c3c'),
                                (random_masked, 'Random Masked', '#3498db')]:
        axes[0, 1].plot(layers, data['S'], 'o-', label=label, color=color,
                        markersize=3, linewidth=1.5)
    axes[0, 1].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Hidden Entropy S')
    axes[0, 1].set_title('(b) Entropy Profile')
    axes[0, 1].legend(fontsize=8)

    # (c) T comparison
    for data, label, color in [(baseline, 'Baseline', '#2ecc71'),
                                (szilard_masked, 'Szilard Masked', '#e74c3c'),
                                (random_masked, 'Random Masked', '#3498db')]:
        axes[0, 2].plot(layers, data['T'], 'o-', label=label, color=color,
                        markersize=3, linewidth=1.5)
    axes[0, 2].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Temperature T')
    axes[0, 2].set_title('(c) Temperature Profile')
    axes[0, 2].legend(fontsize=8)

    # (d) Szilard head locations on info-work map
    axes[1, 0].imshow(info_map * np.maximum(work_map, 0), aspect='auto',
                      cmap='hot', extent=[0, n_heads, n_layers, 0])
    for (sl, sh) in szilard_heads:
        axes[1, 0].plot(sh + 0.5, sl + 0.5, 'co', markersize=8, markeredgecolor='cyan')
    axes[1, 0].set_xlabel('Head Index')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_title('(d) Szilard Head Locations')

    # (e) eta comparison bar chart
    conditions = ['Baseline', 'Szilard\nMasked', 'Random\nMasked']
    etas = [baseline['eta'], szilard_masked['eta'], random_masked['eta']]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = axes[1, 1].bar(conditions, etas, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 1].set_ylabel('Carnot Efficiency eta')
    axes[1, 1].set_title('(e) Efficiency Comparison')
    for bar, val in zip(bars, etas):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=9)

    # (f) Summary
    eta_drop_sz = (baseline['eta'] - szilard_masked['eta']) / (baseline['eta'] + 1e-10) * 100
    eta_drop_rnd = (baseline['eta'] - random_masked['eta']) / (baseline['eta'] + 1e-10) * 100
    summary_text = (
        f"Szilard Engine Lobotomy\n\n"
        f"Baseline eta:       {baseline['eta']:.4f}\n"
        f"Szilard-masked eta: {szilard_masked['eta']:.4f}\n"
        f"Random-masked eta:  {random_masked['eta']:.4f}\n\n"
        f"eta drop (Szilard): {eta_drop_sz:.1f}%\n"
        f"eta drop (Random):  {eta_drop_rnd:.1f}%\n\n"
        f"dS Baseline:  {baseline['dS']:.3f}\n"
        f"dS Szilard:   {szilard_masked['dS']:.3f}\n"
        f"dS Random:    {random_masked['dS']:.3f}\n\n"
        f"Szilard heads: {len(szilard_heads)}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 201: Szilard Engine Lobotomy", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase201_lobotomy')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"eta drop (Szilard): {eta_drop_sz:.1f}% vs Random: {eta_drop_rnd:.1f}%")
    print(f"{'=' * 70}")

    save_results('phase201_lobotomy', {
        'experiment': 'Szilard Engine Lobotomy',
        'szilard_heads': szilard_heads,
        'random_heads': random_heads,
        'baseline': {'eta': baseline['eta'], 'dS': baseline['dS'],
                     'T_hot': baseline['T_hot'], 'T_cold': baseline['T_cold']},
        'szilard_masked': {'eta': szilard_masked['eta'], 'dS': szilard_masked['dS'],
                           'T_hot': szilard_masked['T_hot'], 'T_cold': szilard_masked['T_cold']},
        'random_masked': {'eta': random_masked['eta'], 'dS': random_masked['dS'],
                          'T_hot': random_masked['T_hot'], 'T_cold': random_masked['T_cold']},
        'summary': {
            'eta_drop_szilard_pct': eta_drop_sz,
            'eta_drop_random_pct': eta_drop_rnd,
        }
    })


if __name__ == '__main__':
    main()
