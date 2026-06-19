# -*- coding: utf-8 -*-
"""
Phase 357: TinyLlama Universality Test
========================================
Test whether the 30 universal laws hold on TinyLlama-1.1B,
a completely different architecture from Qwen2.5.
Focus on the top 10 most important laws.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def load_tinyllama(device):
    """Load TinyLlama with local_files_only."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=dtype, device_map=device, local_files_only=True)
    model.eval()
    return model, tok


def measure_all_laws(model, tok, prompt, device):
    """Measure top 10 laws for a single prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    
    n_layers = len(out.hidden_states) - 1
    norm_layer = model.model.norm
    lm_head = model.lm_head
    
    U, T_vals, PR_vals, S_vals = [], [], [], []
    for li, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        u = h.norm().item()
        U.append(u)
        
        # PR
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        pr = 1.0 / (h_prob ** 2).sum().item()
        PR_vals.append(pr)
        
        # Temperature from logits
        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(t_val): t_val = 0.0
        T_vals.append(t_val)
        
        # Hidden state entropy
        s = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
        S_vals.append(s)
    
    U = np.array(U)
    T = np.array(T_vals)
    PR = np.array(PR_vals)
    S = np.array(S_vals)
    F = U - T * S  # Free energy
    
    results = {}
    
    # (1) Boltzmann distribution: R^2 of log-linear energy fit
    log_E = np.log(U[1:] + 1e-10)
    if np.std(log_E) > 0:
        _, _, r, _, _ = stats.linregress(np.arange(len(log_E)), log_E)
        results['boltzmann_r2'] = round(r**2, 4)
    else:
        results['boltzmann_r2'] = 0.0
    
    # (2) Negative specific heat
    dU = np.diff(U)
    dT = np.diff(T)
    valid = np.abs(dT) > 1e-6
    if np.sum(valid) > 2:
        cv = dU[valid] / dT[valid]
        results['cv_mean'] = round(float(np.mean(cv)), 4)
        results['cv_negative'] = bool(np.mean(cv) < 0)
    else:
        results['cv_mean'] = 0.0
        results['cv_negative'] = False
    
    # (3) Inverse radiation law: L ~ T^alpha
    layers = np.arange(1, len(T))
    t_valid = T[1:] > 0
    if np.sum(t_valid) > 3:
        log_l = np.log(layers[t_valid])
        log_t = np.log(T[1:][t_valid])
        sl, _, r, _, _ = stats.linregress(log_t, log_l)
        results['radiation_alpha'] = round(sl, 4)
        results['radiation_r2'] = round(r**2, 4)
    else:
        results['radiation_alpha'] = 0.0
        results['radiation_r2'] = 0.0
    
    # (4) Carnot efficiency
    t_hot = float(np.max(T[1:]))
    t_cold = float(np.min(T[1:]))
    results['carnot_eta'] = round(1.0 - t_cold / (t_hot + 1e-10), 4)
    
    # (5) Anti-FEP: free energy increases
    results['free_energy_ratio'] = round(float(F[-1] / (F[1] + 1e-10)), 4)
    results['anti_fep'] = bool(F[-1] > F[1])
    
    # (6) P1T conservation
    P1_proxy = 1.0 / PR
    P1T = P1_proxy * T
    results['p1t_cv'] = round(float(np.std(P1T[1:]) / (np.mean(P1T[1:]) + 1e-10)), 4)
    
    # (7) Mach convergence
    c_s = float(np.std(np.diff(U)))
    if c_s > 1e-6:
        mach = np.abs(np.diff(U)) / c_s
        results['mach_final'] = round(float(np.mean(mach[-3:])), 4)
        results['mach_mean'] = round(float(np.mean(mach)), 4)
    else:
        results['mach_final'] = 0.0
        results['mach_mean'] = 0.0
    
    # (8) Speed of sound
    results['speed_of_sound'] = round(c_s, 4)
    
    # (9) Information amplification: PR_last / PR_first
    results['info_amp'] = round(float(PR[-1] / (PR[0] + 1e-10)), 4)
    
    # (10) Phase transition detection
    # Look for maximum gradient in T
    if len(T) > 5:
        grad_T = np.abs(np.diff(T))
        transition_layer = int(np.argmax(grad_T))
        results['transition_layer'] = transition_layer
        results['transition_strength'] = round(float(np.max(grad_T) / (np.mean(grad_T) + 1e-10)), 4)
    else:
        results['transition_layer'] = 0
        results['transition_strength'] = 0.0
    
    return results


def main():
    print("=" * 70)
    print("Phase 357: TinyLlama Universality Test")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    
    # TinyLlama
    print("\n=== TinyLlama-1.1B ===")
    model, tok = load_tinyllama(device)
    tl_data = []
    for prompt in PROMPTS:
        r = measure_all_laws(model, tok, prompt, device)
        tl_data.append(r)
    
    # Average
    avg = {}
    for key in tl_data[0]:
        vals = [d[key] for d in tl_data]
        if isinstance(vals[0], bool):
            avg[key] = sum(vals) / len(vals)
        else:
            avg[key] = round(float(np.mean(vals)), 4)
    all_results['TinyLlama-1.1B'] = avg
    
    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Qwen for comparison
    from utils import load_model
    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device, size=size)
        qw_data = []
        for prompt in PROMPTS:
            r = measure_all_laws(model, tok, prompt, device)
            qw_data.append(r)
        avg = {}
        for key in qw_data[0]:
            vals = [d[key] for d in qw_data]
            if isinstance(vals[0], bool):
                avg[key] = sum(vals) / len(vals)
            else:
                avg[key] = round(float(np.mean(vals)), 4)
        all_results[f'Qwen2.5-{size}'] = avg
        del model, tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "=" * 70)
    print("UNIVERSALITY COMPARISON")
    print("=" * 70)
    laws_to_check = [
        ('boltzmann_r2', 'Boltzmann R2', lambda v: v > 0.5),
        ('cv_negative', 'Negative Cv', lambda v: v > 0.5),
        ('carnot_eta', 'Carnot eta', lambda v: v > 0.3),
        ('anti_fep', 'Anti-FEP', lambda v: v > 0.5),
        ('mach_mean', 'Mach Mean', lambda v: v > 0.3),
    ]
    for key, name, check in laws_to_check:
        print(f"\n  {name}:")
        all_pass = True
        for mname, data in all_results.items():
            val = data[key]
            passed = check(val)
            status = "PASS" if passed else "FAIL"
            print(f"    {mname}: {val:.4f} [{status}]")
            if not passed: all_pass = False
        print(f"    -> {'UNIVERSAL' if all_pass else 'NOT UNIVERSAL'}")
    
    # Visualization
    models = list(all_results.keys())
    metrics = ['boltzmann_r2', 'carnot_eta', 'mach_mean', 'p1t_cv', 'speed_of_sound', 'info_amp']
    metric_names = ['Boltzmann R2', 'Carnot eta', 'Mach Mean', 'P1T CV', 'Speed of Sound', 'Info Amp']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_m = {'TinyLlama-1.1B': '#2ecc71', 'Qwen2.5-0.5B': '#3498db', 'Qwen2.5-1.5B': '#e74c3c'}
    
    for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        vals = [all_results[m][metric] for m in models]
        bars = ax.bar(models, vals, color=[colors_m[m] for m in models], alpha=0.8)
        ax.set_title(f'({chr(97+idx)}) {mname}', fontweight='bold')
        ax.grid(alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle("Phase 357: TinyLlama Universality Test", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase357_universality')
    plt.close()
    save_results('phase357_universality', {
        'experiment': 'TinyLlama Universality',
        'results': all_results
    })
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
