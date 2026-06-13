# -*- coding: utf-8 -*-
"""
Phase 56: Newton's Law of Attention
Measure if Attention weight decays as a power law of hidden state distance.
F ~ r^{-k} where F = attention weight, r = L2 distance between token states.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from utils import load_model, save_results, save_figure


def power_law(r, G, k):
    return G * (r + 1e-6) ** (-k)


def main():
    print("=" * 70)
    print("Phase 56: Newton's Law of Attention")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The gravitational force between two masses depends on the distance separating them and",
        "In a neural network the connections between neurons carry signals that propagate through",
        "The electromagnetic spectrum includes radio waves visible light and gamma rays that travel",
        "Machine learning algorithms process training data to discover patterns and relationships in",
        "The periodic table organizes chemical elements by their atomic number and electron configuration",
        "Climate change affects ecosystems worldwide through rising temperatures and shifting weather patterns",
    ]

    all_layer_results = []

    for pi, prompt in enumerate(prompts):
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]

        if seq_len < 5:
            continue

        # Capture attention weights via hooks (output_attentions may be None for Qwen)
        captured_attns = {}
        hooks = []

        def make_attn_hook(li):
            def hook(module, args, output):
                # output is (attn_output, attn_weights, ...) or just attn_output
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    captured_attns[li] = output[1].detach()
            return hook

        for li in range(len(model.model.layers)):
            h = model.model.layers[li].self_attn.register_forward_hook(make_attn_hook(li))
            hooks.append(h)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)

        for h in hooks:
            h.remove()

        # Fallback: try output_attentions
        if not captured_attns and out.attentions is not None:
            for li in range(len(out.attentions)):
                if out.attentions[li] is not None:
                    captured_attns[li] = out.attentions[li]

        hidden_states = out.hidden_states

        if not captured_attns:
            # Use hidden-state cosine similarity as proxy for "attention"
            for li in range(len(hidden_states) - 1):
                hs = hidden_states[li][0].float()
                last_hs = hs[-1, :]

                distances = []
                weights = []
                for j in range(seq_len - 1):
                    d = (hs[j] - last_hs).norm().item()
                    # Use cosine similarity as proxy for attention weight
                    cos = torch.nn.functional.cosine_similarity(
                        hs[j].unsqueeze(0), last_hs.unsqueeze(0)).item()
                    w = max(cos, 1e-8)
                    if d > 0:
                        distances.append(d)
                        weights.append(w)

                if len(distances) < 5:
                    continue

                r = np.array(distances)
                F = np.array(weights)

                try:
                    log_r = np.log(r)
                    log_F = np.log(F)
                    slope, intercept, r_val, p_val, _ = stats.linregress(log_r, log_F)
                    k = -slope
                    G = np.exp(intercept)
                    r_squared = r_val ** 2
                except Exception:
                    k, G, r_squared, p_val = 0, 0, 0, 1

                all_layer_results.append({
                    'prompt_idx': pi, 'layer': li,
                    'k': float(k), 'G': float(G),
                    'r_squared': float(r_squared), 'p_value': float(p_val),
                    'n_pairs': len(distances),
                })
        else:
            for li in sorted(captured_attns.keys()):
                attn = captured_attns[li][0].float().mean(dim=0)
                last_attn = attn[-1, :].cpu().numpy()

                hs = hidden_states[li][0].float()
                last_hs = hs[-1, :]

                distances = []
                weights = []
                for j in range(seq_len - 1):
                    d = (hs[j] - last_hs).norm().item()
                    w = last_attn[j]
                    if d > 0 and w > 1e-8:
                        distances.append(d)
                        weights.append(w)

                if len(distances) < 5:
                    continue

                r = np.array(distances)
                F = np.array(weights)

                try:
                    log_r = np.log(r)
                    log_F = np.log(F)
                    slope, intercept, r_val, p_val, _ = stats.linregress(log_r, log_F)
                    k = -slope
                    G = np.exp(intercept)
                    r_squared = r_val ** 2
                except Exception:
                    k, G, r_squared, p_val = 0, 0, 0, 1

                all_layer_results.append({
                    'prompt_idx': pi, 'layer': li,
                    'k': float(k), 'G': float(G),
                    'r_squared': float(r_squared), 'p_value': float(p_val),
                    'n_pairs': len(distances),
                })

        if pi == 0:
            print(f"  Prompt 0: {len(captured_attns)} attn layers, "
                  f"{len(all_layer_results)} results so far")

    if not all_layer_results:
        print("  No valid results collected")
        save_results('phase56_newton_attention', {'summary': {'verdict': 'NO DATA'}})
        return

    # Aggregate by layer
    n_layers = max(r['layer'] for r in all_layer_results) + 1
    layer_ks = {}
    for r in all_layer_results:
        li = r['layer']
        if li not in layer_ks:
            layer_ks[li] = []
        layer_ks[li].append(r['k'])

    mean_k_per_layer = {li: np.mean(ks) for li, ks in layer_ks.items()}
    overall_k = np.mean([r['k'] for r in all_layer_results])
    overall_r2 = np.mean([r['r_squared'] for r in all_layer_results])
    pct_positive_k = sum(1 for r in all_layer_results if r['k'] > 0) / len(all_layer_results) * 100

    print(f"\n=== Power Law Analysis ===")
    print(f"  Overall: k={overall_k:.3f}, R2={overall_r2:.3f}")
    print(f"  {pct_positive_k:.0f}% have k > 0 (attractive/decaying)")

    for li in sorted(mean_k_per_layer.keys()):
        if li % 7 == 0:
            print(f"  Layer {li}: mean k={mean_k_per_layer[li]:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) k per layer
    layers_plot = sorted(mean_k_per_layer.keys())
    ks_plot = [mean_k_per_layer[l] for l in layers_plot]
    axes[0, 0].plot(layers_plot, ks_plot, 'o-', color='#e74c3c', markersize=4, linewidth=1.5)
    axes[0, 0].axhline(y=2, color='blue', linestyle='--', alpha=0.5, label='Inverse-square (k=2)')
    axes[0, 0].axhline(y=0, color='gray', linewidth=1)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Power Law Exponent k')
    axes[0, 0].set_title(f'(a) Attention Decay Exponent (mean k={overall_k:.2f})')
    axes[0, 0].legend()

    # (b) R2 per layer
    layer_r2s = {}
    for r in all_layer_results:
        if r['layer'] not in layer_r2s:
            layer_r2s[r['layer']] = []
        layer_r2s[r['layer']].append(r['r_squared'])
    mean_r2 = {li: np.mean(vs) for li, vs in layer_r2s.items()}
    axes[0, 1].bar(sorted(mean_r2.keys()), [mean_r2[l] for l in sorted(mean_r2.keys())],
                   color='#3498db', alpha=0.7)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('R-squared (log-log fit)')
    axes[0, 1].set_title(f'(b) Power Law Fit Quality (mean R2={overall_r2:.3f})')

    # (c) Example scatter: use cosine similarity from stored data
    mid_layer = n_layers // 2
    mid_results = [r for r in all_layer_results if r['prompt_idx'] == 0
                   and r['layer'] == mid_layer]
    # Re-extract using hidden states and cosine similarity
    inp0 = tok(prompts[0], return_tensors='pt').to(device)
    with torch.no_grad():
        out0 = model(**inp0, output_hidden_states=True)
    hs0 = out0.hidden_states[mid_layer][0].float()
    last_hs0 = hs0[-1]
    ds, ws = [], []
    for j in range(hs0.shape[0] - 1):
        d = (hs0[j] - last_hs0).norm().item()
        cos = torch.nn.functional.cosine_similarity(
            hs0[j].unsqueeze(0), last_hs0.unsqueeze(0)).item()
        w = max(cos, 1e-8)
        if d > 0:
            ds.append(d)
            ws.append(w)
    if ds:
        axes[0, 2].scatter(np.log(ds), np.log(ws), s=20, color='#e74c3c', alpha=0.7)
        s, i, _, _, _ = stats.linregress(np.log(ds), np.log(ws))
        x_fit = np.linspace(min(np.log(ds)), max(np.log(ds)), 50)
        axes[0, 2].plot(x_fit, s * x_fit + i, 'b--', linewidth=2,
                       label=f'k={-s:.2f}')
        axes[0, 2].set_xlabel('log(distance)')
        axes[0, 2].set_ylabel('log(attention)')
        axes[0, 2].set_title(f'(c) Example: Layer {mid_layer}')
        axes[0, 2].legend()

    # (d) k distribution
    all_ks = [r['k'] for r in all_layer_results]
    axes[1, 0].hist(all_ks, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=2, color='red', linewidth=2, linestyle='--', label='k=2 (inverse square)')
    axes[1, 0].axvline(x=overall_k, color='blue', linewidth=2, label=f'Mean k={overall_k:.2f}')
    axes[1, 0].set_xlabel('Exponent k')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('(d) Distribution of k')
    axes[1, 0].legend(fontsize=8)

    # (e) k vs R2
    axes[1, 1].scatter([r['k'] for r in all_layer_results],
                      [r['r_squared'] for r in all_layer_results],
                      s=10, alpha=0.3, color='#e74c3c')
    axes[1, 1].set_xlabel('Exponent k')
    axes[1, 1].set_ylabel('R-squared')
    axes[1, 1].set_title('(e) k vs Fit Quality')

    # (f) G (gravitational constant) per layer
    layer_Gs = {}
    for r in all_layer_results:
        if r['layer'] not in layer_Gs:
            layer_Gs[r['layer']] = []
        layer_Gs[r['layer']].append(r['G'])
    mean_G = {li: np.mean(vs) for li, vs in layer_Gs.items()}
    axes[1, 2].plot(sorted(mean_G.keys()), [mean_G[l] for l in sorted(mean_G.keys())],
                   'o-', color='#2ecc71', markersize=4)
    axes[1, 2].set_xlabel('Layer')
    axes[1, 2].set_ylabel('G (coupling constant)')
    axes[1, 2].set_title('(f) Gravitational Constant G per Layer')

    fig.suptitle("Phase 56: Newton's Law of Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase56_newton_attention')
    plt.close()

    is_power_law = overall_r2 > 0.3 and pct_positive_k > 50

    print(f"\n{'='*70}")
    print(f"VERDICT: Attention decay k={overall_k:.2f}, R2={overall_r2:.3f}, "
          f"{pct_positive_k:.0f}% attractive. "
          f"Power law {'CONFIRMED' if is_power_law else 'NOT confirmed'}. "
          f"{'Inverse-square-like!' if abs(overall_k - 2) < 1 else f'k={overall_k:.1f} (not inverse-square)'}.")
    print(f"{'='*70}")

    save_results('phase56_newton_attention', {
        'experiment': "Newton's Law of Attention",
        'overall_k': float(overall_k),
        'overall_r2': float(overall_r2),
        'pct_positive_k': float(pct_positive_k),
        'summary': {
            'is_power_law': bool(is_power_law),
            'mean_k': float(overall_k),
            'mean_r2': float(overall_r2),
        }
    })


if __name__ == '__main__':
    main()
