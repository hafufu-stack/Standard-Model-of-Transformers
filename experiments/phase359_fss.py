# -*- coding: utf-8 -*-
"""
Phase 359: Finite-Size Scaling at Phase Transition
=====================================================
Precise measurement of the phase transition at L0~21.
Use finite-size scaling (FSS) to extract critical exponents
and verify the 2D XY universality class (beta=0.161).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
    "Darwin proposed that natural selection drives adaptation",
    "The periodic table organizes elements by atomic number",
    "Neural networks learn hierarchical representations of data",
    "Photosynthesis converts sunlight into chemical energy",
]


def detect_transition(T_profile):
    """Find the phase transition point using susceptibility peak."""
    T = np.array(T_profile)
    n = len(T)
    if n < 5:
        return 0, 0, T
    
    # Compute "susceptibility" = variance in local T fluctuations
    window = 3
    susceptibility = []
    for i in range(window, n - window):
        local = T[i-window:i+window+1]
        susceptibility.append(float(np.var(local)))
    
    if len(susceptibility) == 0:
        return 0, 0, T
    
    # Peak = critical point
    peak_idx = int(np.argmax(susceptibility))
    L0 = peak_idx + window  # actual layer index
    
    # Order parameter: T deviation from mean
    T_mean = np.mean(T)
    order_param = np.abs(T - T_mean)
    
    return L0, float(np.max(susceptibility)), order_param


def measure_critical_exponents(model, tok, prompts, device):
    """Measure critical exponents from multiple prompts."""
    all_T = []
    all_U = []
    
    for prompt in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        all_T.append([t['T'] for t in thermo])
        all_U.append([t['U'] for t in thermo])
    
    n = min(len(x) for x in all_T)
    T_mean = np.mean([[x[i] for i in range(n)] for x in all_T], axis=0)
    T_std = np.std([[x[i] for i in range(n)] for x in all_T], axis=0)
    U_mean = np.mean([[x[i] for i in range(n)] for x in all_U], axis=0)
    
    # Find transition
    L0, chi_max, order_param = detect_transition(T_mean)
    
    # Critical exponent beta: order_parameter ~ |L - L0|^beta near L0
    layers = np.arange(n)
    dist_from_L0 = np.abs(layers - L0)
    
    # Use points near but not at the transition
    mask = (dist_from_L0 > 0) & (dist_from_L0 < n//3) & (order_param > 1e-6)
    if np.sum(mask) > 3:
        log_d = np.log(dist_from_L0[mask])
        log_op = np.log(order_param[mask])
        sl, inter, r, p, _ = stats.linregress(log_d, log_op)
        beta = round(sl, 4)
        beta_r2 = round(r**2, 4)
    else:
        beta = 0.0
        beta_r2 = 0.0
    
    # Critical exponent nu: correlation length ~ |L - L0|^(-nu)
    # Correlation length from T autocorrelation
    T_centered = T_mean - np.mean(T_mean)
    autocorr = np.correlate(T_centered, T_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    
    # Find correlation length (first zero crossing or 1/e)
    xi = 0
    for i in range(1, len(autocorr)):
        if autocorr[i] < 1/np.e:
            xi = i
            break
    
    # Exponent gamma: susceptibility ~ |L - L0|^(-gamma)
    window = 2
    chi_profile = []
    for i in range(window, n - window):
        local_var = float(np.var(T_mean[max(0,i-window):min(n,i+window+1)]))
        chi_profile.append(local_var)
    
    chi_profile = np.array(chi_profile)
    chi_layers = np.arange(window, n - window)
    dist_chi = np.abs(chi_layers - L0)
    mask_chi = (dist_chi > 1) & (dist_chi < n//3) & (chi_profile > 1e-8)
    
    if np.sum(mask_chi) > 3:
        log_dc = np.log(dist_chi[mask_chi])
        log_chi = np.log(chi_profile[mask_chi])
        sl_g, _, r_g, _, _ = stats.linregress(log_dc, log_chi)
        gamma = round(-sl_g, 4)  # chi ~ |t|^{-gamma}
        gamma_r2 = round(r_g**2, 4)
    else:
        gamma = 0.0
        gamma_r2 = 0.0
    
    # Hyperscaling check: gamma = nu * (2 - eta)
    # In 2D XY: beta = 0.23 (or BKT), gamma ~ 1.3, nu ~ 0.67
    
    return {
        'L0': int(L0),
        'chi_max': round(chi_max, 6),
        'beta': beta,
        'beta_r2': beta_r2,
        'xi': int(xi),
        'gamma': gamma,
        'gamma_r2': gamma_r2,
        'T_profile': [round(float(x), 4) for x in T_mean],
        'T_std': [round(float(x), 4) for x in T_std],
        'order_param': [round(float(x), 4) for x in order_param],
        'chi_profile': [round(float(x), 6) for x in chi_profile],
    }


def main():
    print("=" * 70)
    print("Phase 359: Finite-Size Scaling at Phase Transition")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    
    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device, size=size)
        data = measure_critical_exponents(model, tok, PROMPTS, device)
        all_results[size] = data
        print(f"  L0 = {data['L0']}, beta = {data['beta']:.4f} (R2={data['beta_r2']:.3f})")
        print(f"  xi = {data['xi']}, gamma = {data['gamma']:.4f} (R2={data['gamma_r2']:.3f})")
        print(f"  chi_max = {data['chi_max']:.6f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # TinyLlama
    print("\n=== TinyLlama-1.1B ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=dtype, device_map=device, local_files_only=True)
    model.eval()
    
    # Need to patch measure_full_thermodynamics for TinyLlama
    # TinyLlama also has model.model.norm and model.lm_head
    data = measure_critical_exponents(model, tok, PROMPTS, device)
    all_results['TinyLlama-1.1B'] = data
    print(f"  L0 = {data['L0']}, beta = {data['beta']:.4f} (R2={data['beta_r2']:.3f})")
    print(f"  xi = {data['xi']}, gamma = {data['gamma']:.4f} (R2={data['gamma_r2']:.3f})")
    
    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', 'TinyLlama-1.1B': '#2ecc71'}
    
    # (a) Temperature profiles with transition marked
    for name, data in all_results.items():
        T = data['T_profile']
        ax = axes[0, 0]
        ax.plot(T, '-', color=colors[name], lw=2, label=name)
        ax.axvline(data['L0'], color=colors[name], ls='--', alpha=0.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Temperature')
    ax.set_title('(a) Temperature Profile', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    
    # (b) Susceptibility
    for name, data in all_results.items():
        axes[0, 1].plot(data['chi_profile'], '-', color=colors[name], lw=2, label=name)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Susceptibility')
    axes[0, 1].set_title('(b) Susceptibility Peak', fontweight='bold')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)
    
    # (c) Order parameter
    for name, data in all_results.items():
        axes[0, 2].plot(data['order_param'], '-', color=colors[name], lw=2, label=name)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('|T - <T>|')
    axes[0, 2].set_title('(c) Order Parameter', fontweight='bold')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)
    
    # (d) Critical exponent beta comparison
    names = list(all_results.keys())
    betas = [all_results[n]['beta'] for n in names]
    ax = axes[1, 0]
    bars = ax.bar(names, betas, color=[colors[n] for n in names], alpha=0.8)
    ax.axhline(0.161, color='gray', ls='--', label='2D XY (0.161)')
    ax.axhline(0.3265, color='orange', ls='--', label='3D Ising (0.327)')
    ax.set_title('(d) Critical Exponent beta', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # (e) Transition layer comparison
    L0s = [all_results[n]['L0'] for n in names]
    n_layers = [len(all_results[n]['T_profile'])-1 for n in names]
    L0_frac = [l/nl for l, nl in zip(L0s, n_layers)]
    ax = axes[1, 1]
    ax.bar(names, L0_frac, color=[colors[n] for n in names], alpha=0.8)
    ax.set_title('(e) Transition Point (fraction)', fontweight='bold')
    ax.set_ylabel('L0 / n_layers'); ax.grid(alpha=0.3)
    
    # (f) Summary
    axes[1, 2].axis('off')
    txt = "CRITICAL EXPONENTS\n\n"
    for name in names:
        d = all_results[name]
        txt += f"{name}:\n"
        txt += f"  L0 = {d['L0']}\n"
        txt += f"  beta = {d['beta']:.3f}\n"
        txt += f"  gamma = {d['gamma']:.3f}\n"
        txt += f"  xi = {d['xi']}\n\n"
    txt += "2D XY: beta=0.161\n"
    txt += "3D Ising: beta=0.327"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    
    fig.suptitle("Phase 359: Finite-Size Scaling at Phase Transition",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase359_fss')
    plt.close()
    
    save_results('phase359_fss', {
        'experiment': 'Finite-Size Scaling',
        'results': {k: {kk: vv for kk, vv in v.items() 
                       if kk not in ['T_profile', 'T_std', 'order_param', 'chi_profile']}
                   for k, v in all_results.items()},
        'full_data': all_results,
    })
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
