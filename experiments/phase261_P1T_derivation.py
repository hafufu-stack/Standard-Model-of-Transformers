# -*- coding: utf-8 -*-
"""
Phase 261: Analytical Derivation of P1 x T Conservation
=========================================================
Phase 257 discovered: P1 * T_sm (max probability x output entropy)
is conserved across layers with CV=0.14.

This phase investigates WHETHER and WHY this is a mathematical necessity:

1. Analytical: For softmax distributions parameterized by sharpness beta,
   compute P1(beta) * T(beta) and check if it's approximately constant.
2. Empirical: Map the (P1, T) trajectory across layers and compare to
   the analytical prediction.
3. Synthetic: Generate random logit distributions with varying sharpness
   and measure the P1*T product.

If P1*T = const follows from softmax geometry alone, it's a THEOREM.
If it requires transformer-specific logit structure, it's a PHYSICAL LAW.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "General relativity describes gravity as spacetime curvature",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Colorless green ideas sleep furiously in",
]


def analytical_P1T(V=151936):
    """
    Analytical: For a softmax distribution with one dominant logit,
    compute P1 * T as a function of the sharpness parameter.
    
    Model: z = [Delta, 0, 0, ..., 0] (one hot + uniform background)
    Then P1 = exp(Delta) / (exp(Delta) + V-1)
    T = -P1*log(P1) - (1-P1)*log((1-P1)/(V-1))
    """
    # Sweep Delta from 0 (uniform) to large (peaked)
    deltas = np.linspace(0, 20, 500)
    P1_vals, T_vals, PRT_vals = [], [], []
    
    for delta in deltas:
        # Exact softmax
        exp_delta = np.exp(min(delta, 500))  # prevent overflow
        P1 = exp_delta / (exp_delta + V - 1)
        
        # Entropy
        p_rest = (1 - P1) / (V - 1) if V > 1 else 0
        T = -P1 * np.log(P1 + 1e-15)
        if p_rest > 1e-15:
            T -= (V - 1) * p_rest * np.log(p_rest + 1e-15)
        
        P1_vals.append(P1)
        T_vals.append(T)
        PRT_vals.append(P1 * T)
    
    return {
        'deltas': deltas.tolist(),
        'P1': P1_vals,
        'T': T_vals,
        'PRT': PRT_vals,
    }


def analytical_zipf(V=151936):
    """
    Alternative model: Zipf-like logit distribution z_i ~ log(V/i).
    Scale by beta (inverse temperature) to vary sharpness.
    """
    ranks = np.arange(1, min(V, 5000) + 1)
    base_logits = np.log(V / ranks)  # Zipf-like
    
    betas = np.linspace(0.01, 5.0, 200)
    P1_vals, T_vals, PRT_vals = [], [], []
    
    for beta in betas:
        logits = base_logits * beta
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        
        P1 = float(probs[0])
        T = float(-np.sum(probs[probs > 1e-15] * np.log(probs[probs > 1e-15])))
        PRT_vals.append(P1 * T)
        P1_vals.append(P1)
        T_vals.append(T)
    
    return {
        'betas': betas.tolist(),
        'P1': P1_vals,
        'T': T_vals,
        'PRT': PRT_vals,
    }


def empirical_P1T(model, tok, device, model_name):
    """Measure actual P1*T trajectory across layers."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    
    all_P1, all_T, all_PRT = [], [], []
    
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        
        P1_l, T_l, PRT_l = [], [], []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            T_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T_sm): T_sm = 0
            P1_l.append(P1)
            T_l.append(T_sm)
            PRT_l.append(P1 * T_sm)
        
        all_P1.append(P1_l)
        all_T.append(T_l)
        all_PRT.append(PRT_l)
    
    n = min(len(p) for p in all_P1)
    avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
    
    mean_P1 = avg(all_P1)
    mean_T = avg(all_T)
    mean_PRT = avg(all_PRT)
    
    # CV of PRT (skip embedding)
    cv = float(np.std(mean_PRT[1:]) / (np.mean(mean_PRT[1:]) + 1e-10))
    
    return {
        'model': model_name,
        'mean_P1': mean_P1,
        'mean_T': mean_T,
        'mean_PRT': mean_PRT,
        'PRT_cv': round(cv, 4),
        'PRT_mean': round(float(np.mean(mean_PRT[1:])), 4),
    }


def synthetic_random(V=151936, n_samples=500):
    """
    Random logit vectors with varying effective dimensionality.
    If P1*T = const holds for RANDOM distributions, it's purely a
    softmax geometric property. If not, it requires structured logits.
    """
    P1_vals, T_vals, PRT_vals = [], [], []
    
    for i in range(n_samples):
        # Varying effective rank: sample logits from distribution with
        # different numbers of significant components
        k = max(1, int(np.random.exponential(100)))  # effective rank
        logits = np.zeros(min(V, 10000))
        logits[:k] = np.random.randn(k) * np.random.uniform(0.5, 5.0)
        np.random.shuffle(logits)
        
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        
        P1 = float(probs.max())
        T = float(-np.sum(probs[probs > 1e-15] * np.log(probs[probs > 1e-15])))
        
        P1_vals.append(P1)
        T_vals.append(T)
        PRT_vals.append(P1 * T)
    
    cv = float(np.std(PRT_vals) / (np.mean(PRT_vals) + 1e-10))
    return {
        'P1': P1_vals,
        'T': T_vals,
        'PRT': PRT_vals,
        'cv': round(cv, 4),
    }


def main():
    print("=" * 70)
    print("Phase 261: Analytical Derivation of P1 x T Conservation")
    print("=" * 70)
    
    # === Part 1: Analytical (one-hot model) ===
    print("\n--- Analytical (one-hot + uniform) ---")
    ana_hot = analytical_P1T()
    prt = ana_hot['PRT']
    # Find the range where P1 is in [0.01, 0.95]
    valid = [(p, t, pt) for p, t, pt in zip(ana_hot['P1'], ana_hot['T'], prt) if 0.01 < p < 0.95]
    if valid:
        cv_hot = float(np.std([x[2] for x in valid]) / (np.mean([x[2] for x in valid]) + 1e-10))
        print(f"  One-hot model: CV(P1*T) = {cv_hot:.4f} over P1 in [0.01, 0.95]")
    
    # === Part 2: Analytical (Zipf model) ===
    print("\n--- Analytical (Zipf-like logits) ---")
    ana_zipf = analytical_zipf()
    valid_z = [(p, t, pt) for p, t, pt in zip(ana_zipf['P1'], ana_zipf['T'], ana_zipf['PRT']) if 0.01 < p < 0.95]
    if valid_z:
        cv_zipf = float(np.std([x[2] for x in valid_z]) / (np.mean([x[2] for x in valid_z]) + 1e-10))
        print(f"  Zipf model: CV(P1*T) = {cv_zipf:.4f} over P1 in [0.01, 0.95]")
    
    # === Part 3: Synthetic random ===
    print("\n--- Synthetic random logits ---")
    syn = synthetic_random()
    print(f"  Random logits: CV(P1*T) = {syn['cv']:.4f}")
    
    # === Part 4: Empirical transformer ===
    print("\n--- Empirical transformer ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for size in ['0.5B', '1.5B']:
        model, tok = load_model(device, size=size)
        emp = empirical_P1T(model, tok, device, size)
        results[size] = emp
        print(f"  {size}: CV(P1*T) = {emp['PRT_cv']:.4f}, mean = {emp['PRT_mean']:.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # === Verdict ===
    cv_hot_val = cv_hot if valid else 999
    cv_zipf_val = cv_zipf if valid_z else 999
    cv_syn = syn['cv']
    cv_emp_avg = np.mean([r['PRT_cv'] for r in results.values()])
    
    if cv_hot_val < 0.2 and cv_zipf_val < 0.2:
        verdict = "THEOREM: P1*T ~ const follows from softmax geometry alone"
        theorem_type = "geometric"
    elif cv_emp_avg < cv_syn * 0.5:
        verdict = "PHYSICAL LAW: P1*T conservation requires transformer-specific logit structure"
        theorem_type = "physical"
    else:
        verdict = "COINCIDENCE: P1*T conservation is not significantly better than random"
        theorem_type = "coincidence"
    
    print(f"\n  VERDICT: {verdict}")
    print(f"  CV comparison: one-hot={cv_hot_val:.3f}, zipf={cv_zipf_val:.3f}, random={cv_syn:.3f}, transformer={cv_emp_avg:.3f}")
    
    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # (a) One-hot: P1*T vs Delta
    axes[0, 0].plot(ana_hot['deltas'], ana_hot['PRT'], '-', color='#e74c3c', lw=2)
    axes[0, 0].set_xlabel('Delta (sharpness)')
    axes[0, 0].set_ylabel('P1 x T')
    axes[0, 0].set_title(f'(a) One-Hot Model (CV={cv_hot_val:.3f})', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # (b) Zipf: P1*T vs beta
    axes[0, 1].plot(ana_zipf['betas'], ana_zipf['PRT'], '-', color='#3498db', lw=2)
    axes[0, 1].set_xlabel('Beta (inverse temperature)')
    axes[0, 1].set_ylabel('P1 x T')
    axes[0, 1].set_title(f'(b) Zipf Model (CV={cv_zipf_val:.3f})', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # (c) P1 vs T for all models
    axes[0, 2].scatter(syn['P1'], syn['T'], c='gray', s=5, alpha=0.3, label=f'Random (CV={cv_syn:.2f})')
    for size, emp in results.items():
        axes[0, 2].plot(emp['mean_P1'], emp['mean_T'], 'o-', markersize=4, lw=2,
                       label=f'{size} (CV={emp["PRT_cv"]:.2f})')
    axes[0, 2].set_xlabel('P1 (max probability)')
    axes[0, 2].set_ylabel('T (entropy)')
    axes[0, 2].set_title('(c) P1 vs T: Transformer vs Random', fontweight='bold')
    axes[0, 2].legend(fontsize=7); axes[0, 2].grid(alpha=0.3)
    
    # (d) Empirical PRT profiles
    for size, emp in results.items():
        axes[1, 0].plot(range(len(emp['mean_PRT'])), emp['mean_PRT'], '-o', markersize=3,
                       lw=2, label=f'{size} (CV={emp["PRT_cv"]:.3f})')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('P1 x T')
    axes[1, 0].set_title('(d) P1*T Across Layers', fontweight='bold')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)
    
    # (e) P1*T histogram comparison
    axes[1, 1].hist(syn['PRT'], bins=30, alpha=0.4, color='gray', density=True, label='Random')
    for size, emp in results.items():
        axes[1, 1].hist(emp['mean_PRT'][1:], bins=10, alpha=0.5, density=True, label=size)
    axes[1, 1].set_xlabel('P1 x T')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('(e) P1*T Distribution', fontweight='bold')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)
    
    # (f) Summary
    summary = f"P1 x T CONSERVATION ANALYSIS\n\n"
    summary += f"One-hot model: CV = {cv_hot_val:.4f}\n"
    summary += f"Zipf model:    CV = {cv_zipf_val:.4f}\n"
    summary += f"Random:        CV = {cv_syn:.4f}\n"
    for size, emp in results.items():
        summary += f"Transformer {size}: CV = {emp['PRT_cv']:.4f}\n"
    summary += f"\nVERDICT: {theorem_type.upper()}\n"
    summary += verdict
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Verdict')
    
    fig.suptitle("Phase 261: Is P1*T Conservation a Theorem or a Physical Law?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase261_P1T_derivation')
    plt.close()
    
    save_results('phase261_P1T_derivation', {
        'experiment': 'P1*T Analytical Derivation',
        'analytical_one_hot_cv': round(cv_hot_val, 4),
        'analytical_zipf_cv': round(cv_zipf_val, 4),
        'synthetic_random_cv': syn['cv'],
        'empirical': {size: {'cv': r['PRT_cv'], 'mean': r['PRT_mean']} for size, r in results.items()},
        'verdict': verdict,
        'theorem_type': theorem_type,
    })


if __name__ == '__main__':
    main()
