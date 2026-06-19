# -*- coding: utf-8 -*-
"""
Phase 358: Scaling Analysis
============================
How do the physical constants scale with model size?
Test 0.5B, 1.1B (TinyLlama), 1.5B to find power-law relationships:
  constant ~ N^alpha  (N = number of parameters)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]

MODEL_SIZES = {
    'Qwen2.5-0.5B': 0.5e9,
    'TinyLlama-1.1B': 1.1e9,
    'Qwen2.5-1.5B': 1.5e9,
}


def load_any(name, device):
    if 'TinyLlama' in name:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mid = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        dtype = torch.float16 if device == 'cuda' else torch.float32
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=dtype, device_map=device, local_files_only=True)
        model.eval()
        return model, tok
    else:
        size = '0.5B' if '0.5B' in name else '1.5B'
        return load_model(device, size=size)


def measure_constants(model, tok, prompts, device):
    """Measure physical constants from thermodynamic profiles."""
    all_U, all_T, all_PR = [], [], []
    
    for prompt in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        all_U.append([t['U'] for t in thermo])
        all_T.append([t['T'] for t in thermo])
        all_PR.append([t['PR'] for t in thermo])
    
    # Average across prompts
    n = min(len(x) for x in all_U)
    U = np.mean([[x[i] for i in range(n)] for x in all_U], axis=0)
    T = np.mean([[x[i] for i in range(n)] for x in all_T], axis=0)
    PR = np.mean([[x[i] for i in range(n)] for x in all_PR], axis=0)
    
    # Speed of sound
    c_s = float(np.std(np.diff(U)))
    
    # Mach number (average)
    mach = np.abs(np.diff(U)) / (c_s + 1e-10)
    mach_mean = float(np.mean(mach))
    
    # Carnot efficiency
    t_hot = float(np.max(T[1:]))
    t_cold = float(np.min(T[1:]))
    eta = 1.0 - t_cold / (t_hot + 1e-10)
    
    # Free energy amplification
    S = np.array([float(-np.sum(p * np.log(p + 1e-30)) 
                  if np.sum(p) > 0 else 0) 
                  for p in [np.array([1.0/pr]*int(pr)) if pr > 1 else np.array([1.0]) 
                            for pr in PR]])
    F = U - T * S
    fe_ratio = float(F[-1] / (F[1] + 1e-10))
    
    # Temperature range
    t_range = t_hot - t_cold
    
    # PR final
    pr_final = float(PR[-1])
    
    # Energy scale
    u_mean = float(np.mean(U[1:]))
    
    # Number of layers
    n_layers = n - 1
    
    # String tension proxy (from Regge-like analysis)
    cov = np.cov(np.column_stack([U[1:], T[1:]]).T)
    eigs = np.linalg.eigvalsh(cov)
    sigma = float(np.max(eigs) / (np.min(eigs) + 1e-10))
    
    return {
        'c_s': round(c_s, 4),
        'mach_mean': round(mach_mean, 4),
        'carnot_eta': round(eta, 4),
        'fe_ratio': round(fe_ratio, 4),
        't_range': round(t_range, 4),
        'pr_final': round(pr_final, 2),
        'u_mean': round(u_mean, 4),
        'n_layers': n_layers,
        'sigma_proxy': round(sigma, 4),
    }


def main():
    print("=" * 70)
    print("Phase 358: Scaling Analysis")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    
    for name, n_params in MODEL_SIZES.items():
        print(f"\n=== {name} ({n_params/1e9:.1f}B) ===")
        model, tok = load_any(name, device)
        constants = measure_constants(model, tok, PROMPTS, device)
        constants['n_params'] = n_params
        all_results[name] = constants
        print(f"  c_s={constants['c_s']:.2f}, Mach={constants['mach_mean']:.2f}, "
              f"eta={constants['carnot_eta']:.3f}, sigma={constants['sigma_proxy']:.1f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Scaling analysis: fit constant ~ N^alpha
    N = np.array([all_results[m]['n_params'] for m in MODEL_SIZES])
    log_N = np.log(N)
    
    scaling = {}
    metrics_to_scale = ['c_s', 'mach_mean', 'carnot_eta', 'u_mean', 'pr_final', 'sigma_proxy']
    metric_labels = ['Speed of Sound', 'Mach Number', 'Carnot Efficiency', 
                    'Mean Energy', 'Final PR', 'String Tension']
    
    print("\n" + "=" * 70)
    print("SCALING LAWS: constant ~ N^alpha")
    print("=" * 70)
    for metric, label in zip(metrics_to_scale, metric_labels):
        vals = np.array([all_results[m][metric] for m in MODEL_SIZES])
        if np.all(vals > 0):
            log_v = np.log(vals)
            sl, inter, r, p, _ = stats.linregress(log_N, log_v)
            scaling[metric] = {'alpha': round(sl, 4), 'r2': round(r**2, 4), 'p': round(p, 6)}
            print(f"  {label}: alpha = {sl:.3f}, R2 = {r**2:.3f}, p = {p:.4f}")
        else:
            scaling[metric] = {'alpha': 0, 'r2': 0, 'p': 1}
            print(f"  {label}: non-positive values, skipping")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    for idx, (metric, label) in enumerate(zip(metrics_to_scale, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        vals = [all_results[m][metric] for m in MODEL_SIZES]
        sizes = [MODEL_SIZES[m] / 1e9 for m in MODEL_SIZES]
        
        ax.scatter(sizes, vals, s=100, color='#e74c3c', zorder=5)
        for s, v, m in zip(sizes, vals, MODEL_SIZES):
            ax.annotate(m.split('-')[-1], (s, v), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)
        
        # Power law fit line
        if scaling[metric]['r2'] > 0:
            x_fit = np.linspace(min(sizes)*0.8, max(sizes)*1.2, 50)
            alpha = scaling[metric]['alpha']
            y_fit = np.exp(np.log(vals[0]) + alpha * (np.log(x_fit) - np.log(sizes[0])))
            ax.plot(x_fit, y_fit, '--', color='#3498db', alpha=0.7,
                   label=f'alpha={alpha:.2f}, R2={scaling[metric]["r2"]:.2f}')
        
        ax.set_xlabel('Model Size (B params)')
        ax.set_ylabel(label)
        ax.set_title(f'({chr(97+idx)}) {label}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Phase 358: Scaling Analysis of Physical Constants", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase358_scaling')
    plt.close()
    
    save_results('phase358_scaling', {
        'experiment': 'Scaling Analysis',
        'results': all_results,
        'scaling_laws': scaling,
    })
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
