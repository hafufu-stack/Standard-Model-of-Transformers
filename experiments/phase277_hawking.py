# -*- coding: utf-8 -*-
"""
Phase 277: Hawking Radiation from Black Hole Collapse
=======================================================
Phase 57 showed T->0 singularity through iterative token feeding.
Black holes in physics radiate at T_H = hbar*c^3 / (8*pi*G*M*k_B).

Question: Even at "T->0", do residual temperature fluctuations remain?
This would be the LLM analogue of Hawking radiation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from utils import load_model, save_results, save_figure

SEED = "Hello world"
N_COLLAPSE_STEPS = 20
N_RADIATION_SAMPLES = 50  # measure T many times at each collapse step


def collapse_and_measure_radiation(model, tok, seed_text, device):
    """Iteratively collapse while measuring T fluctuations at each step."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    current_text = seed_text
    step_data = []

    for step in range(N_COLLAPSE_STEPS):
        inp = tok(current_text, return_tensors='pt', truncation=True,
                  max_length=512).to(device)

        # Measure temperature N times with slightly different subsequences
        t_samples = []
        p1_samples = []

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure at the last position
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1_main = probs.max().item()
        t_main = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Measure at multiple positions for "radiation" statistics
        seq_len = inp['input_ids'].shape[1]
        for pos in range(max(0, seq_len - N_RADIATION_SAMPLES), seq_len):
            hs = out.hidden_states[-1][:, pos:pos+1, :]
            with torch.no_grad():
                normed = norm_layer(hs)
                pos_logits = lm_head(normed).squeeze().float()
            pos_probs = torch.softmax(pos_logits, dim=-1)
            t_val = -(pos_probs * torch.log(pos_probs + 1e-10)).sum().item()
            p1_val = pos_probs.max().item()
            if not np.isnan(t_val):
                t_samples.append(t_val)
                p1_samples.append(p1_val)

        mean_t = float(np.mean(t_samples)) if t_samples else t_main
        std_t = float(np.std(t_samples)) if t_samples else 0
        min_t = float(np.min(t_samples)) if t_samples else t_main

        # Hawking temperature: residual fluctuation at minimum
        t_hawking = std_t  # fluctuation as analogue

        step_data.append({
            'step': step,
            'seq_len': seq_len,
            'T_main': round(t_main, 4),
            'T_mean': round(mean_t, 4),
            'T_std': round(std_t, 6),
            'T_min': round(min_t, 4),
            'P1_main': round(p1_main, 4),
            'T_hawking': round(t_hawking, 6),
            'n_samples': len(t_samples),
            'text_preview': current_text[:60],
        })

        print(f"  Step {step}: T={t_main:.3f}, T_std={std_t:.4f}, "
              f"P1={p1_main:.3f}, seq_len={seq_len}")

        # Generate next tokens and feed back
        with torch.no_grad():
            gen = model.generate(inp['input_ids'], max_new_tokens=20,
                                do_sample=False)
        current_text = tok.decode(gen[0], skip_special_tokens=True)

    # Fit exponential decay: T(step) = T_0 * exp(-gamma * step) + T_H
    steps = np.array([d['step'] for d in step_data])
    temps = np.array([d['T_main'] for d in step_data])

    try:
        def decay_model(x, T0, gamma, TH):
            return T0 * np.exp(-gamma * x) + TH
        popt, _ = optimize.curve_fit(decay_model, steps, temps,
                                     p0=[temps[0], 0.1, temps[-1]],
                                     maxfev=5000)
        T0_fit, gamma_fit, TH_fit = popt
        fit_success = True
    except Exception:
        T0_fit, gamma_fit, TH_fit = temps[0], 0, temps[-1]
        fit_success = False

    return {
        'step_data': step_data,
        'T0_fit': round(float(T0_fit), 4),
        'gamma_fit': round(float(gamma_fit), 4),
        'TH_fit': round(float(TH_fit), 4),
        'fit_success': fit_success,
        'final_T_std': step_data[-1]['T_std'],
    }


def main():
    print("=" * 70)
    print("Phase 277: Hawking Radiation from Black Hole Collapse")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        r = collapse_and_measure_radiation(model, tok, SEED, device)
        all_results[size] = r
        print(f"  Decay: T(n) = {r['T0_fit']:.2f} * exp(-{r['gamma_fit']:.3f} * n) "
              f"+ {r['TH_fit']:.4f}")
        print(f"  Hawking T = {r['TH_fit']:.4f} (residual)")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        c = colors[size]
        sd = data['step_data']
        steps = [d['step'] for d in sd]

        # (a) Temperature collapse
        axes[0, 0].plot(steps, [d['T_main'] for d in sd], 'o-', color=c,
                       lw=2, label=size)
        # Fit curve
        if data['fit_success']:
            x_fit = np.linspace(0, max(steps), 100)
            y_fit = data['T0_fit'] * np.exp(-data['gamma_fit'] * x_fit) + data['TH_fit']
            axes[0, 0].plot(x_fit, y_fit, '--', color=c, alpha=0.5)
            axes[0, 0].axhline(data['TH_fit'], color=c, ls=':', alpha=0.3)

    axes[0, 0].set_xlabel('Collapse Step')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Temperature Collapse to T_H', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) T fluctuations (Hawking radiation)
    for size, data in all_results.items():
        sd = data['step_data']
        axes[0, 1].plot([d['step'] for d in sd], [d['T_std'] for d in sd],
                       'o-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Collapse Step')
    axes[0, 1].set_ylabel('T Fluctuation (std)')
    axes[0, 1].set_title('(b) Hawking Radiation (T Fluctuations)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) P1 trajectory
    for size, data in all_results.items():
        sd = data['step_data']
        axes[0, 2].plot([d['step'] for d in sd], [d['P1_main'] for d in sd],
                       'o-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Collapse Step')
    axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) P1 During Collapse', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) T_min at each step
    for size, data in all_results.items():
        sd = data['step_data']
        axes[1, 0].plot([d['step'] for d in sd], [d['T_min'] for d in sd],
                       'o-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Collapse Step')
    axes[1, 0].set_ylabel('T_min')
    axes[1, 0].set_title('(d) Minimum Temperature at Each Step', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) log(T) vs step (exponential decay test)
    for size, data in all_results.items():
        sd = data['step_data']
        t_vals = [max(d['T_main'], 1e-6) for d in sd]
        axes[1, 1].plot([d['step'] for d in sd], np.log(t_vals),
                       'o-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Collapse Step')
    axes[1, 1].set_ylabel('log(T)')
    axes[1, 1].set_title('(e) Log-T Decay', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "HAWKING RADIATION\n\n"
    summary += "T(n) = T0 * exp(-gamma*n) + T_H\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  T0 = {data['T0_fit']:.3f}\n"
        summary += f"  gamma = {data['gamma_fit']:.3f}\n"
        summary += f"  T_H = {data['TH_fit']:.4f}\n"
        summary += f"  Final std = {data['final_T_std']:.4f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 277: Hawking Radiation from Black Hole Collapse",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase277_hawking')
    plt.close()

    save_results('phase277_hawking', {
        'experiment': 'Hawking Radiation',
        'seed': SEED,
        'n_collapse_steps': N_COLLAPSE_STEPS,
        'results': all_results,
    })


if __name__ == '__main__':
    main()
