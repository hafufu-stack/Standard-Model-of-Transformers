# -*- coding: utf-8 -*-
"""
Phase 264: P1*T Conservation and the Arrow of Time — Causal Link?
===================================================================
Phase 262 revealed:
  Qwen (Arrow: rho=-0.36, P1*T CV=0.14) - strong conservation, strong arrow
  GPT2  (Arrow: rho=+0.65, P1*T CV=0.21) - moderate conservation, REVERSED arrow
  Llama (Arrow: rho=+0.27, P1*T CV=0.59) - weak conservation, weak arrow

This phase investigates:
1. Is there a systematic relationship between P1*T CV and Arrow strength?
2. Per-prompt analysis: do individual prompts also conserve P1*T?
3. Semantic vs nonsensical: does prompt coherence affect conservation?
4. Layer-resolved: WHERE does conservation break down in Llama?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import save_results, save_figure

SEMANTIC_PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
]

NONSENSICAL_PROMPTS = [
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Colorless green ideas sleep furiously in",
    "The moon decided to become a professional",
    "Random words create unpredictable sequences when",
    "Seven abstract thoughts collided creating new",
]

ALL_PROMPTS = SEMANTIC_PROMPTS + NONSENSICAL_PROMPTS

MODELS = {
    'GPT2': {
        'name': 'gpt2',
        'norm_path': lambda m: m.transformer.ln_f,
        'head_path': lambda m: m.lm_head,
    },
    'Llama-3.2-1B': {
        'name': 'meta-llama/Llama-3.2-1B',
        'norm_path': lambda m: m.model.norm,
        'head_path': lambda m: m.lm_head,
    },
    'Qwen2.5-0.5B': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'norm_path': lambda m: m.model.norm,
        'head_path': lambda m: m.lm_head,
    },
    'Qwen2.5-1.5B': {
        'name': 'Qwen/Qwen2.5-1.5B',
        'norm_path': lambda m: m.model.norm,
        'head_path': lambda m: m.lm_head,
    },
}


def measure_per_prompt(model, tok, device, norm_layer, lm_head, prompts):
    """Measure P1*T profile for each prompt individually."""
    per_prompt = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        
        P1_l, T_l = [], []
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
        
        PRT = [p * t for p, t in zip(P1_l, T_l)]
        cv = float(np.std(PRT[1:]) / (np.mean(PRT[1:]) + 1e-10))
        rho_T, _ = stats.spearmanr(range(len(T_l)), T_l)
        
        per_prompt.append({
            'prompt': prompt[:40],
            'P1': P1_l,
            'T': T_l,
            'PRT': PRT,
            'cv': round(cv, 4),
            'arrow_rho': round(float(rho_T), 4),
            'PRT_mean': round(float(np.mean(PRT[1:])), 4),
        })
    
    return per_prompt


def main():
    print("=" * 70)
    print("Phase 264: P1*T Conservation and Arrow of Time")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    
    for arch_name, cfg in MODELS.items():
        print(f"\n--- {arch_name} ---")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg['name'], torch_dtype=torch.float16,
                device_map=device, local_files_only=True)
            tok = AutoTokenizer.from_pretrained(cfg['name'], local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
            norm_layer = cfg['norm_path'](model)
            lm_head = cfg['head_path'](model)
            
            per_prompt = measure_per_prompt(model, tok, device, norm_layer, lm_head, ALL_PROMPTS)
            
            # Aggregate
            sem_cvs = [p['cv'] for p in per_prompt[:len(SEMANTIC_PROMPTS)]]
            non_cvs = [p['cv'] for p in per_prompt[len(SEMANTIC_PROMPTS):]]
            all_cvs = [p['cv'] for p in per_prompt]
            all_arrows = [p['arrow_rho'] for p in per_prompt]
            
            # Mean P1*T profile across all prompts
            n = min(len(p['PRT']) for p in per_prompt)
            mean_PRT = [float(np.mean([per_prompt[j]['PRT'][i] for j in range(len(per_prompt))])) for i in range(n)]
            mean_T = [float(np.mean([per_prompt[j]['T'][i] for j in range(len(per_prompt))])) for i in range(n)]
            
            agg_cv = float(np.std(mean_PRT[1:]) / (np.mean(mean_PRT[1:]) + 1e-10))
            agg_arrow, _ = stats.spearmanr(range(n), mean_T)
            
            # Correlation between per-prompt CV and arrow
            r_cv_arrow, p_cv_arrow = stats.pearsonr(all_cvs, all_arrows) if len(all_cvs) > 3 else (0, 1)
            
            result = {
                'arch': arch_name,
                'agg_cv': round(agg_cv, 4),
                'agg_arrow': round(float(agg_arrow), 4),
                'semantic_cv_mean': round(float(np.mean(sem_cvs)), 4),
                'nonsensical_cv_mean': round(float(np.mean(non_cvs)), 4),
                'r_cv_arrow': round(float(r_cv_arrow), 4),
                'p_cv_arrow': round(float(p_cv_arrow), 4),
                'per_prompt_cvs': all_cvs,
                'per_prompt_arrows': all_arrows,
                'mean_PRT': mean_PRT,
                'mean_T': mean_T,
            }
            all_results[arch_name] = result
            
            print(f"  Aggregate: CV={agg_cv:.4f}, Arrow rho={agg_arrow:.4f}")
            print(f"  Semantic prompts: mean CV={np.mean(sem_cvs):.4f}")
            print(f"  Nonsensical prompts: mean CV={np.mean(non_cvs):.4f}")
            print(f"  r(CV, Arrow) = {r_cv_arrow:.4f} (p={p_cv_arrow:.4f})")
            
            del model, tok
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"  SKIP ({e})")
    
    if not all_results:
        print("No models loaded!")
        return
    
    # === Cross-model: CV vs Arrow ===
    arch_cvs = [r['agg_cv'] for r in all_results.values()]
    arch_arrows = [r['agg_arrow'] for r in all_results.values()]
    if len(arch_cvs) >= 3:
        r_cross, p_cross = stats.pearsonr(arch_cvs, arch_arrows)
    else:
        r_cross, p_cross = 0, 1
    
    print(f"\n  Cross-model r(CV, Arrow) = {r_cross:.4f}")
    
    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'GPT2': '#e67e22', 'Llama-3.2-1B': '#9b59b6',
              'Qwen2.5-0.5B': '#3498db', 'Qwen2.5-1.5B': '#e74c3c'}
    
    # (a) P1*T profiles
    for arch, r in all_results.items():
        c = colors.get(arch, '#333')
        x = np.linspace(0, 1, len(r['mean_PRT']))
        axes[0, 0].plot(x, r['mean_PRT'], '-o', color=c, markersize=3, lw=2,
                       label=f"{arch} (CV={r['agg_cv']:.3f})")
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('P1 x T')
    axes[0, 0].set_title('(a) P1*T Profiles', fontweight='bold')
    axes[0, 0].legend(fontsize=6); axes[0, 0].grid(alpha=0.3)
    
    # (b) CV vs Arrow scatter (cross-model)
    for arch, r in all_results.items():
        c = colors.get(arch, '#333')
        axes[0, 1].scatter(r['agg_arrow'], r['agg_cv'], c=c, s=100,
                          zorder=5, label=arch, edgecolor='black')
    axes[0, 1].set_xlabel('Arrow of Time (rho)')
    axes[0, 1].set_ylabel('P1*T CV (lower = better)')
    axes[0, 1].set_title(f'(b) Conservation vs Arrow (r={r_cross:.2f})', fontweight='bold')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)
    
    # (c) Semantic vs Nonsensical CV comparison
    archs = list(all_results.keys())
    x_pos = np.arange(len(archs))
    sem_means = [all_results[a]['semantic_cv_mean'] for a in archs]
    non_means = [all_results[a]['nonsensical_cv_mean'] for a in archs]
    axes[0, 2].bar(x_pos - 0.2, sem_means, 0.35, color='#3498db', alpha=0.8,
                  label='Semantic', edgecolor='black')
    axes[0, 2].bar(x_pos + 0.2, non_means, 0.35, color='#e74c3c', alpha=0.8,
                  label='Nonsensical', edgecolor='black')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(archs, fontsize=7, rotation=15)
    axes[0, 2].set_ylabel('Mean CV(P1*T)')
    axes[0, 2].set_title('(c) Semantic vs Nonsensical', fontweight='bold')
    axes[0, 2].legend(fontsize=7); axes[0, 2].grid(alpha=0.3, axis='y')
    
    # (d) Per-prompt CV distribution
    for arch, r in all_results.items():
        c = colors.get(arch, '#333')
        axes[1, 0].hist(r['per_prompt_cvs'], bins=8, alpha=0.4, color=c,
                       edgecolor='black', label=arch, density=True)
    axes[1, 0].set_xlabel('Per-Prompt CV(P1*T)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('(d) Per-Prompt CV Distribution', fontweight='bold')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)
    
    # (e) Temperature profiles
    for arch, r in all_results.items():
        c = colors.get(arch, '#333')
        x = np.linspace(0, 1, len(r['mean_T']))
        axes[1, 1].plot(x, r['mean_T'], '-', color=c, lw=2, label=arch)
    axes[1, 1].set_xlabel('Normalized Depth')
    axes[1, 1].set_ylabel('T_sm')
    axes[1, 1].set_title('(e) Temperature Profiles', fontweight='bold')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)
    
    # (f) Summary
    summary = "P1*T vs ARROW OF TIME\n\n"
    for arch, r in all_results.items():
        summary += f"{arch:20s}:\n"
        summary += f"  CV={r['agg_cv']:.3f}, Arrow={r['agg_arrow']:+.3f}\n"
        summary += f"  Sem={r['semantic_cv_mean']:.3f}, Non={r['nonsensical_cv_mean']:.3f}\n"
    summary += f"\nCross-model r = {r_cross:.3f}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')
    
    fig.suptitle("Phase 264: P1*T Conservation and the Arrow of Time",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase264_P1T_arrow')
    plt.close()
    
    save_results('phase264_P1T_arrow', {
        'experiment': 'P1*T Conservation vs Arrow of Time',
        'results': {k: {kk: vv for kk, vv in v.items() 
                       if kk not in ('mean_PRT', 'mean_T', 'per_prompt_cvs', 'per_prompt_arrows')}
                   for k, v in all_results.items()},
        'cross_model_r': round(float(r_cross), 4),
        'cross_model_p': round(float(p_cross), 4),
    })


if __name__ == '__main__':
    main()
