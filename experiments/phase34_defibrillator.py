# -*- coding: utf-8 -*-
"""
Phase 34: Thermodynamic Defibrillator (Season 5)
===================================================
When the firewall detects anomalous PR*T variance (hallucination onset),
dynamically suppress FFN output to beta_c=0.57 to pull the model back
to its stable attractor. Tests both detection AND real-time correction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 34: Thermodynamic Defibrillator")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # PRT variance threshold - adaptive based on calibration window
    PRT_WINDOW = 10  # sliding window for variance
    BETA_DEFIB = 0.57  # critical suppression point from Phase 33
    DEFIB_DURATION = 5  # steps to maintain suppression
    CALIBRATION_STEPS = 15  # initial steps for baseline calibration

    test_cases = [
        ("factual", "The capital of France is Paris. The capital of Japan is"),
        ("factual", "Water boils at 100 degrees Celsius. Ice melts at"),
        ("ambiguous", "The ancient prophecy states that when the stars align, the"),
        ("ambiguous", "Scientists recently discovered that consciousness is actually"),
    ]

    # FFN suppression hook
    ffn_hooks = []
    ffn_scale = [1.0]  # mutable container for dynamic scaling

    def make_ffn_hook(layer_idx):
        def hook(module, input, output):
            if ffn_scale[0] >= 1.0:
                return output
            h = output[0] if isinstance(output, tuple) else output
            # Scale down MLP output (dark energy suppression)
            h_scaled = h * ffn_scale[0]
            if isinstance(output, tuple):
                return (h_scaled,) + output[1:]
            return h_scaled
        return hook

    all_results = []
    for label, prompt in test_cases:
        print(f"\n--- {label}: '{prompt[:50]}...' ---")

        # Run twice: without and with defibrillator
        for mode in ['normal', 'defib']:
            # Reset hooks
            for h in ffn_hooks:
                h.remove()
            ffn_hooks.clear()

            if mode == 'defib':
                # Install FFN hooks on all MLP layers
                for i, layer in enumerate(model.model.layers):
                    handle = layer.mlp.register_forward_hook(make_ffn_hook(i))
                    ffn_hooks.append(handle)

            inp = tok(prompt, return_tensors='pt').to(device)
            input_ids = inp['input_ids']
            past_kv = None
            prt_history = []
            trace = []
            generated = []
            defib_active = 0
            defib_count = 0

            for t in range(80):
                if past_kv is None:
                    curr_input = input_ids
                else:
                    curr_input = next_token_id

                with torch.no_grad():
                    out = model(input_ids=curr_input, past_key_values=past_kv,
                               use_cache=True, output_hidden_states=True)

                past_kv = out.past_key_values
                h_last = out.hidden_states[-1][0, -1, :].float()
                U = h_last.norm().item()

                logits = out.logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                PR = 1.0 / (probs ** 2).sum().item()
                PRT = PR * T

                prt_history.append(PRT)

                # Defibrillator logic: adaptive threshold
                if mode == 'defib' and len(prt_history) >= PRT_WINDOW:
                    recent_std = np.std(prt_history[-PRT_WINDOW:])
                    # Adaptive threshold: use calibration window stats
                    if len(prt_history) <= CALIBRATION_STEPS:
                        prt_threshold = float('inf')  # don't trigger during calibration
                    else:
                        cal_stds = [np.std(prt_history[max(0,j-PRT_WINDOW):j])
                                   for j in range(PRT_WINDOW, CALIBRATION_STEPS)]
                        if cal_stds:
                            prt_threshold = np.mean(cal_stds) + 1.5 * np.std(cal_stds)
                        else:
                            prt_threshold = recent_std * 2

                    if recent_std > prt_threshold and defib_active <= 0:
                        ffn_scale[0] = BETA_DEFIB
                        defib_active = DEFIB_DURATION
                        defib_count += 1
                        if defib_count <= 5:
                            print(f"  [DEFIB] t={t}: PRT_std={recent_std:.1f} > {prt_threshold:.1f} -> beta={BETA_DEFIB}")
                    elif defib_active > 0:
                        defib_active -= 1
                        if defib_active == 0:
                            ffn_scale[0] = 1.0
                else:
                    ffn_scale[0] = 1.0

                trace.append({
                    't': t, 'U': U, 'T': T, 'PR': PR, 'PRT': PRT,
                    'beta': ffn_scale[0], 'defib_active': defib_active > 0
                })

                next_token_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
                generated.append(tok.decode(next_token_id[0, 0].item()))

            # Remove hooks
            for h in ffn_hooks:
                h.remove()
            ffn_hooks.clear()
            ffn_scale[0] = 1.0

            prt_std = np.std([s['PRT'] for s in trace])
            text = ''.join(generated[:30])
            print(f"  [{mode}] PRT_std={prt_std:.1f}, defib_count={defib_count}, text='{text[:40]}...'")

            all_results.append({
                'label': label, 'mode': mode, 'prompt': prompt[:50],
                'prt_std': prt_std, 'defib_count': defib_count,
                'trace': trace, 'generated': text
            })

    # === Visualization ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'normal': '#e74c3c', 'defib': '#2ecc71'}

    # Group by prompt
    prompts_seen = []
    for r in all_results:
        key = r['prompt']
        if key not in prompts_seen:
            prompts_seen.append(key)

    for i, prompt_key in enumerate(prompts_seen[:4]):
        ax = axes[i // 2][i % 2]
        for r in all_results:
            if r['prompt'] == prompt_key:
                steps = [s['t'] for s in r['trace']]
                prts = [s['PRT'] for s in r['trace']]
                ax.plot(steps, prts, '-', color=colors[r['mode']], lw=1.5,
                        alpha=0.8, label=f"{r['mode']} (std={r['prt_std']:.0f})")
                # Mark defibrillation events
                if r['mode'] == 'defib':
                    defib_steps = [s['t'] for s in r['trace'] if s['defib_active']]
                    defib_prts = [s['PRT'] for s in r['trace'] if s['defib_active']]
                    if defib_steps:
                        ax.scatter(defib_steps, defib_prts, color='gold', s=30,
                                   zorder=5, marker='*', label='defib active')
        ax.set_title(f"{'[F]' if 'capital' in prompt_key or 'Water' in prompt_key else '[A]'} {prompt_key[:35]}...", fontsize=9)
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('PR x T')
        ax.legend(fontsize=7)

    # Compute improvement metrics
    normal_ambig_stds = [r['prt_std'] for r in all_results if r['label'] == 'ambiguous' and r['mode'] == 'normal']
    defib_ambig_stds = [r['prt_std'] for r in all_results if r['label'] == 'ambiguous' and r['mode'] == 'defib']
    normal_fact_stds = [r['prt_std'] for r in all_results if r['label'] == 'factual' and r['mode'] == 'normal']
    defib_fact_stds = [r['prt_std'] for r in all_results if r['label'] == 'factual' and r['mode'] == 'defib']

    n_a = np.mean(normal_ambig_stds) if normal_ambig_stds else 0
    d_a = np.mean(defib_ambig_stds) if defib_ambig_stds else 0
    n_f = np.mean(normal_fact_stds) if normal_fact_stds else 0
    d_f = np.mean(defib_fact_stds) if defib_fact_stds else 0
    reduction = (1 - d_a / (n_a + 1e-10)) * 100

    fig.suptitle(
        f"Phase 34: Thermodynamic Defibrillator\n"
        f"Ambig PRT_std: {n_a:.0f} (normal) -> {d_a:.0f} (defib) [{reduction:.0f}% reduction]\n"
        f"Factual PRT_std: {n_f:.0f} (normal) -> {d_f:.0f} (defib)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase34_defibrillator")
    plt.close()

    verdict = (f"Defibrillator reduces ambiguous PRT variance by {reduction:.0f}%. "
               f"Normal={n_a:.0f}, Defib={d_a:.0f}. "
               f"Factual impact: {n_f:.0f}->{d_f:.0f}.")
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase34_defibrillator", {
        'name': 'Phase 34: Thermodynamic Defibrillator',
        'summary': {
            'verdict': verdict,
            'normal_ambig_std': n_a, 'defib_ambig_std': d_a,
            'normal_factual_std': n_f, 'defib_factual_std': d_f,
            'variance_reduction_pct': reduction,
            'beta_c': BETA_DEFIB, 'threshold': 'adaptive (mean+1.5*std of calibration)',
        }
    })


if __name__ == '__main__':
    main()
