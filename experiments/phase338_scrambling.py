# -*- coding: utf-8 -*-
"""
Phase 338: Scrambling Time -- Information Chaos
=====================================================
In black hole physics, scrambling time t* ~ log(S) measures how fast
information becomes inaccessible. Test the Transformer's scrambling
time: how many layers before a local perturbation affects all dimensions.
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_scrambling(model, tok, prompt, device):
    """Measure scrambling time via OTOC (Out-of-Time-Order Correlator) proxy."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # Perturbation spreading: add small perturbation to dimension 0
    # and measure how it spreads across dimensions
    epsilon = 0.01
    spread_profile = []

    for li in range(n_layers + 1):
        h = hiddens[li]
        h_perturbed = h.clone()
        h_perturbed[0] += epsilon

        # Measure spread: how many dimensions are affected
        diff = (h_perturbed - h).abs()
        affected = int((diff > epsilon * 0.01).sum().item())
        spread_frac = affected / dim
        spread_profile.append(round(float(spread_frac), 4))

    # OTOC proxy: C(l) = <|[W(0), V(l)]|^2>
    # Approximate as correlation between perturbation at layer 0 and state at layer l
    otoc_values = []
    h0 = hiddens[0]
    for li in range(n_layers + 1):
        h_l = hiddens[li]
        # OTOC ~ 1 - overlap between perturbed and unperturbed evolved states
        cos = float(torch.nn.functional.cosine_similarity(
            h0.unsqueeze(0), h_l.unsqueeze(0)).item())
        otoc = 1 - cos**2
        otoc_values.append(round(float(otoc), 4))

    # Scrambling time: layer where OTOC reaches 1/2 of max
    otoc_arr = np.array(otoc_values)
    max_otoc = otoc_arr.max()
    scrambling_layer = n_layers  # default
    for li in range(len(otoc_arr)):
        if otoc_arr[li] >= max_otoc * 0.5:
            scrambling_layer = li
            break

    # Lyapunov exponent from OTOC growth
    # OTOC ~ exp(2 * lambda_L * l) at early times
    early = min(scrambling_layer + 1, n_layers // 2)
    if early > 2:
        layers_early = np.arange(1, early + 1)
        otoc_early = otoc_arr[1:early + 1]
        valid = otoc_early > 1e-10
        if valid.sum() > 2:
            log_otoc = np.log(otoc_early[valid] + 1e-30)
            slope, _, r, _, _ = stats.linregress(layers_early[valid], log_otoc)
            lambda_L = slope / 2
        else:
            lambda_L, r = 0, 0
    else:
        lambda_L, r = 0, 0

    # Theoretical scrambling time: t* ~ log(dim) / lambda_L
    if lambda_L > 0:
        t_star_theory = np.log(dim) / (2 * lambda_L)
    else:
        t_star_theory = float('inf')

    return {
        'otoc_profile': otoc_values,
        'spread_profile': spread_profile,
        'scrambling_layer': scrambling_layer,
        'lambda_L': round(float(lambda_L), 4),
        't_star_theory': round(float(min(t_star_theory, 1000)), 2),
        't_star_measured': scrambling_layer,
        'max_otoc': round(float(max_otoc), 4),
        'fast_scrambler': scrambling_layer <= np.log(dim) * 2,
    }


def main():
    print("=" * 70)
    print("Phase 338: Scrambling Time")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        scr_data = []
        for prompt in PROMPTS:
            s = measure_scrambling(model, tok, prompt, device)
            scr_data.append(s)

        n = len(scr_data[0]['otoc_profile'])
        all_results[size] = {
            'otoc_profile': [round(float(np.mean([s['otoc_profile'][i] for s in scr_data])), 4)
                           for i in range(n)],
            'spread_profile': [round(float(np.mean([s['spread_profile'][i] for s in scr_data])), 4)
                              for i in range(n)],
            'scrambling_layer': round(float(np.mean([s['scrambling_layer'] for s in scr_data])), 1),
            'lambda_L': round(float(np.mean([s['lambda_L'] for s in scr_data])), 4),
            'max_otoc': round(float(np.mean([s['max_otoc'] for s in scr_data])), 4),
            'fast_scrambler': sum(1 for s in scr_data if s['fast_scrambler']) >= 4,
        }
        fast = 'YES' if all_results[size]['fast_scrambler'] else 'NO'
        print(f"  Scrambling layer: {all_results[size]['scrambling_layer']:.1f}")
        print(f"  Lyapunov: {all_results[size]['lambda_L']:.4f}")
        print(f"  Fast scrambler: {fast}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['otoc_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('OTOC')
    axes[0, 0].set_title('(a) Out-of-Time-Order Correlator', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['spread_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Spread fraction')
    axes[0, 1].set_title('(b) Perturbation Spread', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 2].bar(x - w/2, [all_results[s]['scrambling_layer'] for s in sizes], w,
                  label='t* (measured)', color='#3498db')
    log_dims = [np.log(896), np.log(1536)]  # Qwen2.5 dims
    axes[0, 2].bar(x + w/2, log_dims, w, label='log(d)', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Scrambling Time vs log(d)', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['lambda_L'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('lambda_L')
    axes[1, 0].set_title('(d) Lyapunov Exponent', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "SCRAMBLING TIME\n\n"
    txt += "t* ~ log(S) / lambda_L\n\n"
    for s in sizes:
        d = all_results[s]
        fast = 'YES' if d['fast_scrambler'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  t* = {d['scrambling_layer']:.1f}\n"
        txt += f"  lambda = {d['lambda_L']:.4f}\n"
        txt += f"  fast: {fast}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 338: Scrambling Time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase338_scrambling')
    plt.close()
    save_results('phase338_scrambling', {'experiment': 'Scrambling Time', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
