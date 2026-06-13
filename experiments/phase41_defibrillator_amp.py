# -*- coding: utf-8 -*-
"""
Phase 41: Defibrillator Amplification
Combine FFN cooling (Phase 34) with vector steering toward factual registers.
500+ token generation with adaptive PRT monitoring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 41: Defibrillator Amplification (Cooling + Steering)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    BETA_DEFIB = 0.57
    DEFIB_DURATION = 5
    CALIBRATION_STEPS = 20
    PRT_WINDOW = 10
    GEN_LENGTH = 100  # tokens per test
    STEER_ALPHA = 0.15  # steering strength

    test_cases = [
        ("factual", "Albert Einstein was born in 1879 in Ulm, Germany. He developed the theory of"),
        ("factual", "The speed of light in a vacuum is approximately 299,792,458 meters per second. This constant is"),
        ("ambiguous", "The lost city of Atlantis was discovered in 2024 by deep sea researchers who found evidence that"),
        ("ambiguous", "New research proves that humans can photosynthesize like plants when exposed to"),
        ("factual", "The chemical formula for water is H2O. Each molecule consists of two hydrogen atoms and"),
        ("hallucination_bait", "The President of Mars announced in his speech yesterday that the interplanetary"),
    ]

    all_results = []

    for cat, prompt in test_cases:
        print(f"\n--- {cat}: '{prompt[:55]}...' ---")
        input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)

        results_by_mode = {}

        for mode in ['normal', 'cooling_only', 'cooling_steering']:
            ffn_scale = [1.0]
            steer_active = [False]
            hooks = []

            # Capture factual prototype from calibration
            factual_prototypes = {}

            def make_ffn_hook(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if ffn_scale[0] < 1.0:
                        h = h * ffn_scale[0]
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook

            def make_steer_hook(layer_idx):
                def hook(module, input, output):
                    if not steer_active[0]:
                        return output
                    h = output[0] if isinstance(output, tuple) else output
                    if layer_idx in factual_prototypes:
                        proto = factual_prototypes[layer_idx]
                        if proto.shape == h[0, -1, :].shape:
                            # Nudge toward prototype
                            direction = proto - h[0, -1, :].float()
                            direction = direction / (direction.norm() + 1e-10)
                            h_mod = h.clone()
                            h_mod[0, -1, :] += (STEER_ALPHA * direction * h[0, -1, :].float().norm()).to(h.dtype)
                            if isinstance(output, tuple):
                                return (h_mod,) + output[1:]
                            return h_mod
                    return output
                return hook

            if mode in ('cooling_only', 'cooling_steering'):
                for li in range(len(model.model.layers)):
                    h = model.model.layers[li].mlp.register_forward_hook(make_ffn_hook(li))
                    hooks.append(h)

            if mode == 'cooling_steering':
                # Steering hooks on key layers (L10, L16 - fact extraction registers)
                for li in [10, 16]:
                    if li < len(model.model.layers):
                        h = model.model.layers[li].register_forward_hook(make_steer_hook(li))
                        hooks.append(h)

            # Calibration: run initial prompt to get prototypes
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                for li in [10, 16]:
                    if li < len(out.hidden_states):
                        factual_prototypes[li] = out.hidden_states[li][0, -1, :].float().clone()

            # Generate tokens one by one with monitoring
            current_ids = input_ids.clone()
            prt_history = []
            defib_count = 0
            defib_active = 0
            generated_tokens = []

            for t in range(GEN_LENGTH):
                with torch.no_grad():
                    out = model(current_ids, output_hidden_states=True)
                    logits = out.logits[0, -1, :].float()

                # Measure PRT
                probs = torch.softmax(logits, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T
                prt_history.append(PRT)

                # Defibrillator logic
                if mode != 'normal' and len(prt_history) >= PRT_WINDOW:
                    recent_std = np.std(prt_history[-PRT_WINDOW:])
                    if len(prt_history) > CALIBRATION_STEPS:
                        cal_vals = prt_history[:CALIBRATION_STEPS]
                        cal_stds = [np.std(cal_vals[max(0,j-PRT_WINDOW):j])
                                   for j in range(PRT_WINDOW, len(cal_vals))]
                        if cal_stds:
                            threshold = np.mean(cal_stds) + 1.5 * np.std(cal_stds)
                        else:
                            threshold = recent_std * 2

                        if recent_std > threshold and defib_active <= 0:
                            ffn_scale[0] = BETA_DEFIB
                            steer_active[0] = True
                            defib_active = DEFIB_DURATION
                            defib_count += 1
                        elif defib_active > 0:
                            defib_active -= 1
                            if defib_active == 0:
                                ffn_scale[0] = 1.0
                                steer_active[0] = False
                    else:
                        ffn_scale[0] = 1.0
                        steer_active[0] = False

                # Sample next token (greedy)
                next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                generated_tokens.append(next_id.item())
                current_ids = torch.cat([current_ids, next_id], dim=1)

                # Truncate context if too long
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            for h in hooks:
                h.remove()

            text = tok.decode(generated_tokens, skip_special_tokens=True)
            prt_std = np.std(prt_history) if prt_history else 0
            prt_mean = np.mean(prt_history) if prt_history else 0

            safe_text = text.encode('ascii', errors='replace').decode('ascii')[:60]
            print(f"  [{mode}] PRT_std={prt_std:.1f}, defib={defib_count}, "
                  f"text='{safe_text}...'")

            results_by_mode[mode] = {
                'prt_std': prt_std, 'prt_mean': prt_mean,
                'defib_count': defib_count,
                'text': text[:200],
                'prt_history': [float(v) for v in prt_history],
            }

        all_results.append({
            'category': cat, 'prompt': prompt[:80],
            **results_by_mode
        })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, result in enumerate(all_results[:6]):
        ax = axes[idx // 3][idx % 3]
        for mode, color in [('normal', '#e74c3c'), ('cooling_only', '#3498db'), ('cooling_steering', '#2ecc71')]:
            if mode in result and 'prt_history' in result[mode]:
                ax.plot(result[mode]['prt_history'], color=color, alpha=0.7, label=mode, linewidth=0.8)
        ax.set_title(f"{result['category']}: ...{result['prompt'][-30:]}", fontsize=8)
        ax.set_xlabel('Token')
        ax.set_ylabel('PRT')
        ax.legend(fontsize=6)

    fig.suptitle('Phase 41: Defibrillator Amplification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase41_defibrillator_amp')
    plt.close()

    # === Verdict ===
    normal_stds = [r['normal']['prt_std'] for r in all_results if 'normal' in r]
    cooling_stds = [r['cooling_only']['prt_std'] for r in all_results if 'cooling_only' in r]
    steer_stds = [r['cooling_steering']['prt_std'] for r in all_results if 'cooling_steering' in r]

    n_std = np.mean(normal_stds) if normal_stds else 0
    c_std = np.mean(cooling_stds) if cooling_stds else 0
    s_std = np.mean(steer_stds) if steer_stds else 0
    cool_pct = (1 - c_std / (n_std + 1e-10)) * 100
    steer_pct = (1 - s_std / (n_std + 1e-10)) * 100

    print(f"\n{'='*70}")
    print(f"VERDICT: Normal PRT_std={n_std:.1f}, Cooling={c_std:.1f} ({cool_pct:.0f}% reduction), "
          f"Cooling+Steering={s_std:.1f} ({steer_pct:.0f}% reduction).")
    print(f"{'='*70}")

    save_results('phase41_defibrillator_amp', {
        'experiment': 'Defibrillator Amplification',
        'results': all_results,
        'summary': {
            'normal_prt_std': n_std, 'cooling_prt_std': c_std,
            'steering_prt_std': s_std,
            'cooling_reduction_pct': cool_pct,
            'steering_reduction_pct': steer_pct,
        }
    })


if __name__ == '__main__':
    main()
