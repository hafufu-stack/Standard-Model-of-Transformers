# -*- coding: utf-8 -*-
"""
Phase 339: Chaos Bound -- MSS (Maldacena-Shenker-Stanford)
=====================================================
The MSS bound: lambda_L <= 2*pi*T/hbar is the fundamental upper bound
on Lyapunov exponent in quantum systems. Black holes saturate this bound.
Test whether the Transformer's information dynamics respects this bound.
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


def measure_mss(model, tok, prompt, device):
    """Measure MSS chaos bound."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]

    # Measure temperature at each layer
    temperatures = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        e = h.abs()
        T = float(torch.var(e).item()) / (float(torch.mean(e).item()) + 1e-10)
        temperatures.append(T)

    # Measure Lyapunov exponent from hidden state divergence
    # Perturb initial state and measure exponential growth of difference
    epsilon = 1e-4
    h0_perturbed = hiddens[0].clone()
    h0_perturbed[0] += epsilon

    divergences = []
    for li in range(n_layers + 1):
        diff = torch.norm(hiddens[li] - h0_perturbed).item() if li == 0 else 0
        # Use OTOC proxy for divergence at each layer
        cos = float(torch.nn.functional.cosine_similarity(
            hiddens[0].unsqueeze(0), hiddens[li].unsqueeze(0)).item())
        div = max(0, 1 - cos)
        divergences.append(div)

    # Lyapunov from early growth
    div_arr = np.array(divergences)
    layers = np.arange(len(div_arr))
    valid = div_arr > 1e-10
    if valid.sum() > 3:
        log_div = np.log(div_arr[valid] + 1e-30)
        slope, _, r, _, _ = stats.linregress(layers[valid], log_div)
        lambda_L = float(slope)
    else:
        lambda_L = 0.0

    # MSS bound: lambda_L <= 2*pi*T (with hbar=1)
    T_avg = float(np.mean(temperatures[1:]))  # Exclude embedding
    mss_bound = 2 * np.pi * T_avg
    ratio = lambda_L / (mss_bound + 1e-10)
    bound_satisfied = lambda_L <= mss_bound * 1.1  # 10% margin

    # Per-layer check
    layer_ratios = []
    for li in range(1, n_layers + 1):
        local_bound = 2 * np.pi * temperatures[li]
        layer_ratios.append(round(float(lambda_L / (local_bound + 1e-10)), 4))

    return {
        'lambda_L': round(float(lambda_L), 4),
        'T_avg': round(float(T_avg), 4),
        'mss_bound': round(float(mss_bound), 4),
        'ratio': round(float(ratio), 4),
        'bound_satisfied': bool(bound_satisfied),
        'temperature_profile': [round(t, 4) for t in temperatures],
        'divergence_profile': [round(d, 6) for d in divergences],
        'layer_ratios': layer_ratios,
    }


def main():
    print("=" * 70)
    print("Phase 339: MSS Chaos Bound")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        mss_data = []
        for prompt in PROMPTS:
            m = measure_mss(model, tok, prompt, device)
            mss_data.append(m)

        n = len(mss_data[0]['temperature_profile'])
        all_results[size] = {
            'lambda_L': round(float(np.mean([m['lambda_L'] for m in mss_data])), 4),
            'T_avg': round(float(np.mean([m['T_avg'] for m in mss_data])), 4),
            'mss_bound': round(float(np.mean([m['mss_bound'] for m in mss_data])), 4),
            'ratio': round(float(np.mean([m['ratio'] for m in mss_data])), 4),
            'bound_satisfied': sum(1 for m in mss_data if m['bound_satisfied']) >= 4,
            'temperature_profile': [round(float(np.mean([m['temperature_profile'][i] for m in mss_data])), 4)
                                   for i in range(n)],
        }
        sat = 'YES' if all_results[size]['bound_satisfied'] else 'NO'
        print(f"  lambda_L: {all_results[size]['lambda_L']:.4f}")
        print(f"  MSS bound: {all_results[size]['mss_bound']:.4f}")
        print(f"  Ratio: {all_results[size]['ratio']:.4f}")
        print(f"  Satisfied: {sat}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['temperature_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) Temperature Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 1].bar(x - w/2, [all_results[s]['lambda_L'] for s in sizes], w,
                  label='lambda_L', color='#3498db')
    axes[0, 1].bar(x + w/2, [all_results[s]['mss_bound'] for s in sizes], w,
                  label='MSS bound', color='#e74c3c')
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(sizes)
    axes[0, 1].set_title('(b) Lyapunov vs MSS Bound', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(sizes, [all_results[s]['ratio'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].axhline(1.0, color='gold', ls='--', lw=2, label='MSS limit')
    axes[0, 2].set_ylabel('lambda/bound')
    axes[0, 2].set_title('(c) Saturation Ratio', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "MSS CHAOS BOUND\n\n"
    txt += "lambda_L <= 2*pi*T\n\n"
    for s in sizes:
        d = all_results[s]
        sat = 'YES' if d['bound_satisfied'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  lambda = {d['lambda_L']:.4f}\n"
        txt += f"  bound = {d['mss_bound']:.4f}\n"
        txt += f"  ratio = {d['ratio']:.4f}\n"
        txt += f"  OK: {sat}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 339: MSS Chaos Bound", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase339_mss')
    plt.close()
    save_results('phase339_mss', {'experiment': 'MSS Chaos Bound', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
