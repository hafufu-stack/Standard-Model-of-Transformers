# -*- coding: utf-8 -*-
"""
Phase 239: Autoregressive Generation Dynamics
================================================
Track thermodynamic variables as the model generates tokens autoregressively.
Question: How does T, P1, U evolve over the *time* axis (generation steps)?
Is there a thermodynamic signature of coherent vs incoherent generation?
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

SEEDS = [
    "The meaning of life is",
    "In the beginning there was",
    "The most important scientific discovery is",
    "Once upon a time in a distant galaxy",
    "The future of artificial intelligence will",
]

MAX_TOKENS = 50


def generation_dynamics(model, tok, device, model_name):
    """Track thermodynamic evolution during autoregressive generation."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_runs = []
    for seed in SEEDS:
        input_ids = tok(seed, return_tensors='pt').input_ids.to(device)
        
        step_data = []
        for step in range(MAX_TOKENS):
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            
            # Measure at final hidden state (last layer)
            final_hs = out.hidden_states[-1]
            h = final_hs[0, -1, :].float()
            U = h.norm().item()
            
            # Get logit-space measurements
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T = float(S) if not np.isnan(S) else 0
            
            # Also measure at intermediate layer (~50%)
            mid_idx = len(out.hidden_states) // 2
            mid_hs = out.hidden_states[mid_idx]
            with torch.no_grad():
                mid_normed = norm_layer(mid_hs[:, -1:, :])
                mid_logits = lm_head(mid_normed).squeeze().float()
            mid_probs = torch.softmax(mid_logits, dim=-1)
            mid_T = -(mid_probs * torch.log(mid_probs + 1e-10)).sum().item()
            
            # Sample next token (greedy)
            next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
            token_text = tok.decode(next_token[0])
            
            step_data.append({
                'step': step,
                'token': token_text.strip(),
                'T': T,
                'P1': P1,
                'U': U,
                'T_mid': float(mid_T) if not np.isnan(mid_T) else 0,
            })
            
            # Append for next step
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Compute generated text
        gen_text = tok.decode(input_ids[0], skip_special_tokens=True)
        
        # Time-series analysis
        T_series = [s['T'] for s in step_data]
        P1_series = [s['P1'] for s in step_data]
        U_series = [s['U'] for s in step_data]
        
        # Autocorrelation of T
        T_arr = np.array(T_series) - np.mean(T_series)
        if np.std(T_arr) > 1e-10:
            autocorr = np.correlate(T_arr, T_arr, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            # Correlation time: first zero crossing
            zero_cross = np.where(autocorr[1:] < 0)[0]
            corr_time = float(zero_cross[0] + 1) if len(zero_cross) > 0 else float(MAX_TOKENS)
        else:
            corr_time = 0.0
            autocorr = np.zeros(MAX_TOKENS)
        
        # Trend
        rho_T, _ = stats.spearmanr(range(MAX_TOKENS), T_series)
        rho_P1, _ = stats.spearmanr(range(MAX_TOKENS), P1_series)
        
        all_runs.append({
            'seed': seed[:40],
            'gen_text': gen_text[:120],
            'step_data': step_data,
            'T_mean': float(np.mean(T_series)),
            'T_std': float(np.std(T_series)),
            'P1_mean': float(np.mean(P1_series)),
            'corr_time': corr_time,
            'rho_T': float(rho_T),
            'rho_P1': float(rho_P1),
            'autocorr': autocorr[:20].tolist() if hasattr(autocorr, 'tolist') else [],
        })
    
    return {
        'model': model_name,
        'max_tokens': MAX_TOKENS,
        'runs': all_runs,
    }


def main():
    print("=" * 70)
    print("Phase 239: Autoregressive Generation Dynamics")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = generation_dynamics(model, tok, device, size)
        results[size] = r
        for run in r['runs']:
            print(f"  {run['seed'][:30]}...")
            print(f"    T_mean={run['T_mean']:.2f}+/-{run['T_std']:.2f}, "
                  f"corr_time={run['corr_time']:.1f}, rho_T={run['rho_T']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_run = plt.cm.Set2(np.linspace(0, 1, len(SEEDS)))
    
    # (a) T over generation steps (1.5B)
    r15 = results.get('1.5B', results.get('0.5B'))
    for ri, run in enumerate(r15['runs']):
        T_vals = [s['T'] for s in run['step_data']]
        axes[0, 0].plot(range(MAX_TOKENS), T_vals, '-', color=colors_run[ri],
                       lw=1.5, alpha=0.7, label=run['seed'][:20])
    axes[0, 0].set_xlabel('Generation Step')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) T During Generation')
    axes[0, 0].legend(fontsize=5, loc='upper right')
    
    # (b) P1 over generation steps
    for ri, run in enumerate(r15['runs']):
        P1_vals = [s['P1'] for s in run['step_data']]
        axes[0, 1].plot(range(MAX_TOKENS), P1_vals, '-', color=colors_run[ri],
                       lw=1.5, alpha=0.7, label=run['seed'][:20])
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('P1')
    axes[0, 1].set_title('(b) P1 During Generation')
    axes[0, 1].legend(fontsize=5, loc='upper right')
    
    # (c) T vs T_mid (final vs middle layer during generation)
    for ri, run in enumerate(r15['runs']):
        T_vals = [s['T'] for s in run['step_data']]
        T_mid = [s['T_mid'] for s in run['step_data']]
        axes[0, 2].scatter(T_mid, T_vals, c=[colors_run[ri]]*MAX_TOKENS,
                          s=15, alpha=0.5)
    axes[0, 2].plot([0, 8], [0, 8], 'k--', alpha=0.3, label='y=x')
    axes[0, 2].set_xlabel('T (middle layer)')
    axes[0, 2].set_ylabel('T (final layer)')
    axes[0, 2].set_title('(c) Mid vs Final Temperature')
    axes[0, 2].legend(fontsize=7)
    
    # (d) Autocorrelation
    for ri, run in enumerate(r15['runs']):
        ac = run['autocorr']
        if ac:
            axes[1, 0].plot(range(len(ac)), ac, '-', color=colors_run[ri],
                           lw=1.5, alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_title('(d) T Autocorrelation')
    
    # (e) Size comparison: T statistics
    sizes = list(results.keys())
    T_means = [[run['T_mean'] for run in results[s]['runs']] for s in sizes]
    axes[1, 1].boxplot(T_means, labels=sizes)
    axes[1, 1].set_ylabel('Mean T per run')
    axes[1, 1].set_title('(e) Generation T by Size')
    
    # (f) Summary
    summary = "GENERATION DYNAMICS\n\n"
    for size, r in results.items():
        avg_T = np.mean([run['T_mean'] for run in r['runs']])
        avg_corr = np.mean([run['corr_time'] for run in r['runs']])
        avg_rho = np.mean([run['rho_T'] for run in r['runs']])
        summary += f"{size}:\n"
        summary += f"  <T> = {avg_T:.2f}\n"
        summary += f"  corr_time = {avg_corr:.1f}\n"
        summary += f"  rho(T,step) = {avg_rho:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')
    
    fig.suptitle("Phase 239: Autoregressive Generation Dynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase239_generation')
    plt.close()
    save_results('phase239_generation', {
        'experiment': 'Generation Dynamics',
        'results': results,
    })


if __name__ == '__main__':
    main()
