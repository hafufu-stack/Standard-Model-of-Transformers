# -*- coding: utf-8 -*-
"""
Phase 356: Prompt Independence Test
====================================
Test whether the top 5 universal laws hold across diverse prompt categories:
  (1) Boltzmann distribution  (2) Negative Cv  (3) P1*T conservation
  (4) Mach convergence  (5) Carnot efficiency
If these hold for science, literature, code, nonsense, and multilingual prompts,
they are truly structural properties of the model, not input artifacts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPT_CATEGORIES = {
    'science': [
        "The second law of thermodynamics states that entropy",
        "In quantum mechanics the wave function describes",
        "The speed of light in vacuum is approximately",
    ],
    'literature': [
        "It was the best of times it was the worst of times",
        "Call me Ishmael some years ago never mind how long",
        "All happy families are alike each unhappy family",
    ],
    'code': [
        "def fibonacci(n): if n <= 1: return n else: return",
        "import numpy as np; x = np.linspace(0, 2*np.pi",
        "class NeuralNetwork(nn.Module): def __init__(self",
    ],
    'nonsense': [
        "purple elephant seventeen calculator moon whisper",
        "the the the the the the the the the the the the",
        "asdfjkl qwerty zxcvbn poiuytrewq mnbvcxz lkjhgfdsa",
    ],
    'multilingual': [
        "Die Quantenmechanik beschreibt die physikalischen",
        "La relativite generale est une theorie geometrique",
        "El universo se expande a una velocidad cada vez mayor",
    ],
}


def measure_laws(model, tok, prompt, device):
    """Measure the 5 core laws for a single prompt."""
    thermo, out = measure_full_thermodynamics(model, tok, prompt, device)
    n = len(thermo)
    
    U = np.array([t['U'] for t in thermo])
    T = np.array([t['T'] for t in thermo])
    PR = np.array([t['PR'] for t in thermo])
    PRT = np.array([t['PRT'] for t in thermo])
    
    # (1) Boltzmann: log(P(E)) ~ -E/kT => energy histogram R^2
    energies = U[1:]  # skip embedding
    if len(energies) > 3 and np.std(energies) > 0:
        log_E = np.log(energies + 1e-10)
        slope, intercept, r_boltz, _, _ = stats.linregress(
            np.arange(len(log_E)), log_E)
        r2_boltz = r_boltz**2
    else:
        r2_boltz = 0.0
    
    # (2) Negative Cv: dU/dT < 0
    dU = np.diff(U)
    dT = np.diff(T)
    valid = np.abs(dT) > 1e-6
    if np.sum(valid) > 2:
        cv_local = dU[valid] / dT[valid]
        cv_mean = float(np.mean(cv_local))
        cv_negative = cv_mean < 0
    else:
        cv_mean = 0
        cv_negative = False
    
    # (3) P1*T conservation
    # P1 = first principal component loading
    # Approximate with PR normalization
    P1_proxy = 1.0 / PR  # inverse PR as P1 proxy
    P1T = P1_proxy * T
    if len(P1T) > 2:
        cv_p1t = float(np.std(P1T[1:]) / (np.mean(P1T[1:]) + 1e-10))
    else:
        cv_p1t = 1.0
    p1t_conserved = cv_p1t < 0.3
    
    # (4) Mach convergence: M -> 1.0
    # M = |delta_U| / c_s where c_s ~ std(U)
    if len(U) > 3:
        c_s = float(np.std(np.diff(U)))
        if c_s > 1e-6:
            mach = np.abs(np.diff(U)) / c_s
            mach_final = float(np.mean(mach[-3:]))
        else:
            mach_final = 0
    else:
        mach_final = 0
    
    # (5) Carnot efficiency: eta = 1 - T_cold/T_hot
    T_hot = float(np.max(T[1:]))
    T_cold = float(np.min(T[1:]))
    if T_hot > 1e-6:
        eta = 1.0 - T_cold / T_hot
    else:
        eta = 0
    
    return {
        'r2_boltzmann': round(r2_boltz, 4),
        'cv_mean': round(cv_mean, 4),
        'cv_negative': bool(cv_negative),
        'p1t_cv': round(cv_p1t, 4),
        'p1t_conserved': bool(p1t_conserved),
        'mach_final': round(mach_final, 4),
        'carnot_eta': round(eta, 4),
    }


def main():
    print("=" * 70)
    print("Phase 356: Prompt Independence Test")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    
    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device, size=size)
        
        size_data = {}
        for cat, prompts in PROMPT_CATEGORIES.items():
            cat_results = []
            for prompt in prompts:
                r = measure_laws(model, tok, prompt, device)
                cat_results.append(r)
            
            # Average across prompts in category
            avg = {}
            for key in cat_results[0]:
                vals = [r[key] for r in cat_results]
                if isinstance(vals[0], bool):
                    avg[key] = sum(vals) / len(vals)  # fraction True
                else:
                    avg[key] = round(float(np.mean(vals)), 4)
            size_data[cat] = avg
            
            print(f"  {cat}: Boltz R2={avg['r2_boltzmann']:.3f}, "
                  f"Cv<0={avg['cv_negative']:.0%}, "
                  f"P1T CV={avg['p1t_cv']:.3f}, "
                  f"Mach={avg['mach_final']:.2f}, "
                  f"eta={avg['carnot_eta']:.3f}")
        
        all_results[size] = size_data
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Visualization
    categories = list(PROMPT_CATEGORIES.keys())
    laws = ['r2_boltzmann', 'cv_negative', 'p1t_conserved', 'mach_final', 'carnot_eta']
    law_names = ['Boltzmann R2', 'Neg. Cv', 'P1T Cons.', 'Mach Final', 'Carnot eta']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_cat = {'science': '#e74c3c', 'literature': '#3498db', 
                  'code': '#2ecc71', 'nonsense': '#9b59b6', 'multilingual': '#f39c12'}
    
    for idx, (law, lname) in enumerate(zip(laws, law_names)):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(categories))
        w = 0.35
        for si, size in enumerate(['0.5B', '1.5B']):
            vals = [all_results[size][cat][law] for cat in categories]
            bars = ax.bar(x + si*w - w/2, vals, w, label=size,
                         color=['#3498db', '#e74c3c'][si], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
        ax.set_title(f'({chr(97+idx)}) {lname}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Summary panel
    axes[1, 2].axis('off')
    txt = "PROMPT INDEPENDENCE\n\n"
    for size in ['0.5B', '1.5B']:
        boltz_vals = [all_results[size][c]['r2_boltzmann'] for c in categories]
        cv_vals = [all_results[size][c]['cv_negative'] for c in categories]
        txt += f"{size}:\n"
        txt += f"  Boltz R2: {np.mean(boltz_vals):.3f} +/- {np.std(boltz_vals):.3f}\n"
        txt += f"  Cv<0: {np.mean(cv_vals):.0%} of categories\n"
        txt += f"  Verdict: {'UNIVERSAL' if np.mean(boltz_vals) > 0.5 else 'NOT UNIVERSAL'}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    
    fig.suptitle("Phase 356: Prompt Independence of Universal Laws", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase356_prompt_independence')
    plt.close()
    save_results('phase356_prompt_independence', {
        'experiment': 'Prompt Independence',
        'results': all_results
    })
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
