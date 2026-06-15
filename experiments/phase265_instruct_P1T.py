# -*- coding: utf-8 -*-
"""
Phase 265: Instruction Tuning Effect on P1*T Conservation
============================================================
Law 5 states: Instruction tuning (SFT/RLHF) reduces final-layer T.
Question: Does it also change the P1*T conservation law?

Compare:
  - Qwen2.5-0.5B (base) vs Qwen2.5-0.5B-Instruct (tuned)
  - Qwen2.5-1.5B (base) vs Qwen2.5-1.5B-Instruct (tuned)

Predictions:
  H1: RLHF preserves P1*T (changes T and P1 but not their product)
  H2: RLHF changes the P1*T constant (like a phase transition)
  H3: RLHF improves conservation (lower CV)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_any_model, get_model_internals, save_results, save_figure

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


def measure_thermodynamics(model, tok, device, model_name, norm_layer=None, lm_head=None):
    """Full thermodynamic profile: P1, T, PRT, arrow."""
    if norm_layer is None:
        internals = get_model_internals(model)
        norm_layer = internals['norm']
        lm_head = internals['lm_head']
    
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
    
    cv = float(np.std(mean_PRT[1:]) / (np.mean(mean_PRT[1:]) + 1e-10))
    rho_T, _ = stats.spearmanr(range(n), mean_T)
    
    return {
        'model': model_name,
        'n_layers': n,
        'mean_P1': mean_P1,
        'mean_T': mean_T,
        'mean_PRT': mean_PRT,
        'PRT_cv': round(cv, 4),
        'PRT_mean': round(float(np.mean(mean_PRT[1:])), 4),
        'PRT_final': round(float(mean_PRT[-1]), 4),
        'T_final': round(float(mean_T[-1]), 4),
        'P1_final': round(float(mean_P1[-1]), 4),
        'arrow_rho': round(float(rho_T), 4),
    }


def main():
    print("=" * 70)
    print("Phase 265: Instruction Tuning Effect on P1*T")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    model_ids = [
        ('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'),
        ('Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
        ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B'),
        ('Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'),
    ]
    
    for label, model_id in model_ids:
        print(f"\n--- {label} ---")
        try:
            model, tok = load_any_model(model_id, device=device)
            internals = get_model_internals(model)
            r = measure_thermodynamics(model, tok, device, label,
                                       norm_layer=internals['norm'],
                                       lm_head=internals['lm_head'])
            results[label] = r
            print(f"  P1*T: CV={r['PRT_cv']:.4f}, mean={r['PRT_mean']:.3f}")
            print(f"  T_final={r['T_final']:.3f}, P1_final={r['P1_final']:.3f}")
            print(f"  Arrow rho = {r['arrow_rho']:.4f}")
            del model, tok
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"  SKIP ({e})")
    
    if len(results) < 2:
        print("Need at least 2 models!")
        return
    
    # === Analysis ===
    # Compare base vs instruct
    pairs = []
    for size in ['0.5B', '1.5B']:
        base_key = f"Qwen2.5-{size}"
        inst_key = f"Qwen2.5-{size}-Instruct"
        if base_key in results and inst_key in results:
            pairs.append({
                'size': size,
                'base': results[base_key],
                'inst': results[inst_key],
                'dCV': results[inst_key]['PRT_cv'] - results[base_key]['PRT_cv'],
                'dPRT': results[inst_key]['PRT_mean'] - results[base_key]['PRT_mean'],
                'dT': results[inst_key]['T_final'] - results[base_key]['T_final'],
                'dP1': results[inst_key]['P1_final'] - results[base_key]['P1_final'],
            })
    
    for p in pairs:
        print(f"\n  {p['size']} Base->Instruct:")
        print(f"    dCV(P1*T) = {p['dCV']:+.4f}")
        print(f"    dPRT_mean = {p['dPRT']:+.4f}")
        print(f"    dT_final  = {p['dT']:+.4f}")
        print(f"    dP1_final = {p['dP1']:+.4f}")
    
    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    colors = {
        'Qwen2.5-0.5B': '#3498db',
        'Qwen2.5-0.5B-Instruct': '#2980b9',
        'Qwen2.5-1.5B': '#e74c3c',
        'Qwen2.5-1.5B-Instruct': '#c0392b',
    }
    styles = {}
    for k in results:
        styles[k] = '--' if 'Instruct' in k else '-'
    
    # (a) P1*T profiles
    for name, r in results.items():
        c = colors.get(name, '#333')
        ls = styles.get(name, '-')
        x = np.linspace(0, 1, len(r['mean_PRT']))
        axes[0, 0].plot(x, r['mean_PRT'], ls, color=c, lw=2,
                       label=f"{name} (CV={r['PRT_cv']:.3f})")
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('P1 x T')
    axes[0, 0].set_title('(a) P1*T: Base vs Instruct', fontweight='bold')
    axes[0, 0].legend(fontsize=6); axes[0, 0].grid(alpha=0.3)
    
    # (b) T profiles
    for name, r in results.items():
        c = colors.get(name, '#333')
        ls = styles.get(name, '-')
        x = np.linspace(0, 1, len(r['mean_T']))
        axes[0, 1].plot(x, r['mean_T'], ls, color=c, lw=2, label=name)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('T_sm')
    axes[0, 1].set_title('(b) Temperature: Base vs Instruct', fontweight='bold')
    axes[0, 1].legend(fontsize=6); axes[0, 1].grid(alpha=0.3)
    
    # (c) P1 profiles
    for name, r in results.items():
        c = colors.get(name, '#333')
        ls = styles.get(name, '-')
        x = np.linspace(0, 1, len(r['mean_P1']))
        axes[0, 2].plot(x, r['mean_P1'], ls, color=c, lw=2, label=name)
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) Max Probability: Base vs Instruct', fontweight='bold')
    axes[0, 2].legend(fontsize=6); axes[0, 2].grid(alpha=0.3)
    
    # (d) CV comparison bar
    names = list(results.keys())
    cvs = [results[n]['PRT_cv'] for n in names]
    c_list = [colors.get(n, '#333') for n in names]
    bars = axes[1, 0].bar(range(len(names)), cvs, color=c_list, edgecolor='black', alpha=0.8)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels([n.replace('Qwen2.5-', '') for n in names], fontsize=7, rotation=15)
    axes[1, 0].set_ylabel('CV(P1*T)')
    axes[1, 0].set_title('(d) Conservation Quality', fontweight='bold')
    for bar, cv in zip(bars, cvs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{cv:.3f}', ha='center', fontsize=8, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # (e) PRT mean comparison
    prts = [results[n]['PRT_mean'] for n in names]
    bars2 = axes[1, 1].bar(range(len(names)), prts, color=c_list, edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels([n.replace('Qwen2.5-', '') for n in names], fontsize=7, rotation=15)
    axes[1, 1].set_ylabel('Mean P1*T')
    axes[1, 1].set_title('(e) P1*T Constant Value', fontweight='bold')
    for bar, prt in zip(bars2, prts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{prt:.3f}', ha='center', fontsize=8, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    # (f) Summary
    summary = "INSTRUCTION TUNING EFFECT\n\n"
    for p in pairs:
        summary += f"{p['size']}:\n"
        summary += f"  dCV  = {p['dCV']:+.4f}\n"
        summary += f"  dPRT = {p['dPRT']:+.4f}\n"
        summary += f"  dT   = {p['dT']:+.4f}\n"
        summary += f"  dP1  = {p['dP1']:+.4f}\n\n"
    
    # Verdict
    if all(p['dCV'] < 0 for p in pairs):
        verdict = "RLHF IMPROVES conservation"
    elif all(p['dCV'] > 0 for p in pairs):
        verdict = "RLHF WEAKENS conservation"
    else:
        verdict = "MIXED effect"
    summary += f"Verdict: {verdict}"
    
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')
    
    fig.suptitle("Phase 265: Does Instruction Tuning Change P1*T Conservation?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase265_instruct_P1T')
    plt.close()
    
    save_results('phase265_instruct_P1T', {
        'experiment': 'Instruction Tuning Effect on P1*T',
        'results': {k: {kk: vv for kk, vv in v.items() 
                       if kk not in ('mean_P1', 'mean_T', 'mean_PRT')}
                   for k, v in results.items()},
        'pairs': [{k: v for k, v in p.items() if k not in ('base', 'inst')} for p in pairs],
    })


if __name__ == '__main__':
    main()
