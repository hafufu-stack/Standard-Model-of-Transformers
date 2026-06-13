# -*- coding: utf-8 -*-
"""
Phase 60: Unified Defibrillator v3
Combine Phase 43's velocity detection with Phase 41b's FFN cooling.
Detect anomalous dPRT/dt -> auto-apply cooling -> continue generation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 60: Unified Defibrillator v3")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    BETA_C = 0.57
    COOLING_ALPHA = 0.3
    VELOCITY_TRIGGER = 2.0  # multiples of running std
    WINDOW = 5
    GEN_LENGTH = 80

    prompts = [
        "The process of photosynthesis in plants converts carbon dioxide and water into glucose",
        "Albert Einstein developed the theory of general relativity which describes how gravity",
        "The human nervous system consists of the central and peripheral systems that coordinate",
        "According to recently leaked classified documents, the real reason for the moon missions was",
        "Scientists at a secret laboratory have discovered that consciousness is actually",
        "The lost civilization of Atlantis was recently found and its technology proves that",
    ]

    all_results = []

    for prompt in prompts:
        # Run 3 modes: normal, detect-only, detect+heal
        for mode in ['normal', 'detect', 'detect_heal']:
            input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
            current_ids = input_ids.clone()

            prt_trace = []
            velocity_trace = []
            heals_applied = []
            tokens = []

            # FFN cooling hooks (only active in detect_heal mode when triggered)
            cooling_active = [False]
            hooks = []

            def make_cooling_hook(li):
                def hook(module, input, output):
                    if not cooling_active[0]:
                        return output
                    h = output[0] if isinstance(output, tuple) else output
                    scale = 1.0 - COOLING_ALPHA * BETA_C
                    h_scaled = h * scale
                    if isinstance(output, tuple):
                        return (h_scaled,) + output[1:]
                    return h_scaled
                return hook

            if mode == 'detect_heal':
                for li in range(n_layers):
                    h = model.model.layers[li].mlp.register_forward_hook(make_cooling_hook(li))
                    hooks.append(h)

            for t_step in range(GEN_LENGTH):
                with torch.no_grad():
                    out = model(current_ids)
                    logits = out.logits[0, -1, :].float()

                probs = torch.softmax(logits, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T_val
                if np.isnan(PRT):
                    PRT = 0
                prt_trace.append(PRT)

                # Velocity detection
                if len(prt_trace) >= 2:
                    vel = abs(prt_trace[-1] - prt_trace[-2])
                    velocity_trace.append(vel)

                    if len(velocity_trace) >= WINDOW:
                        recent_vel = velocity_trace[-WINDOW:]
                        mean_v = np.mean(recent_vel)
                        std_v = np.std(recent_vel) + 1e-6
                        z_score = (vel - mean_v) / std_v

                        if mode in ['detect', 'detect_heal'] and z_score > VELOCITY_TRIGGER:
                            if mode == 'detect_heal':
                                cooling_active[0] = True
                                heals_applied.append(t_step)
                        else:
                            cooling_active[0] = False

                next_id = logits.argmax().item()
                tokens.append(next_id)
                next_tensor = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            for h in hooks:
                h.remove()

            text = tok.decode(tokens, skip_special_tokens=True)

            vel_std = float(np.std(velocity_trace)) if velocity_trace else 0

            all_results.append({
                'prompt': prompt[:60],
                'mode': mode,
                'prt_trace': [float(v) for v in prt_trace],
                'velocity_trace': [float(v) for v in velocity_trace],
                'velocity_std': vel_std,
                'n_heals': len(heals_applied),
                'heal_positions': heals_applied,
                'text': text[:200],
            })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    mode_colors = {'normal': '#3498db', 'detect': '#f39c12', 'detect_heal': '#e74c3c'}

    # (a) Velocity std comparison
    mode_vels = {}
    for r in all_results:
        if r['mode'] not in mode_vels:
            mode_vels[r['mode']] = []
        mode_vels[r['mode']].append(r['velocity_std'])
    modes = ['normal', 'detect', 'detect_heal']
    means = [np.mean(mode_vels.get(m, [0])) for m in modes]
    axes[0, 0].bar(modes, means, color=[mode_colors[m] for m in modes], alpha=0.8)
    axes[0, 0].set_ylabel('Velocity Std')
    axes[0, 0].set_title('(a) Velocity Std by Mode')
    for i, v in enumerate(means):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)

    # (b) PRT traces - one prompt, all 3 modes
    for r in all_results[:3]:  # first prompt's 3 modes
        axes[0, 1].plot(r['prt_trace'], color=mode_colors[r['mode']],
                       linewidth=1.5, label=r['mode'], alpha=0.7)
        for hp in r.get('heal_positions', []):
            axes[0, 1].axvline(x=hp, color='green', alpha=0.3, linewidth=0.5)
    axes[0, 1].set_xlabel('Token')
    axes[0, 1].set_ylabel('PRT')
    axes[0, 1].set_title('(b) PRT (Prompt 1)')
    axes[0, 1].legend(fontsize=8)

    # (c) Heals applied
    heals_per_prompt = [r['n_heals'] for r in all_results if r['mode'] == 'detect_heal']
    axes[0, 2].bar(range(len(heals_per_prompt)), heals_per_prompt, color='#2ecc71', alpha=0.8)
    axes[0, 2].set_xlabel('Prompt Index')
    axes[0, 2].set_ylabel('Healing Events')
    axes[0, 2].set_title('(c) Healing Events per Prompt')

    # (d) Velocity traces - risky prompt
    risky_idx = 3  # "According to recently leaked..."
    for r in all_results[risky_idx*3:(risky_idx+1)*3]:
        axes[1, 0].plot(r['velocity_trace'], color=mode_colors[r['mode']],
                       linewidth=1.5, label=r['mode'], alpha=0.7)
    axes[1, 0].set_xlabel('Token')
    axes[1, 0].set_ylabel('|dPRT/dt|')
    axes[1, 0].set_title('(d) Velocity (Risky Prompt)')
    axes[1, 0].legend(fontsize=8)

    # (e) Normal vs healed velocity comparison for risky prompts
    risky_normal = [r['velocity_std'] for r in all_results
                    if r['mode'] == 'normal' and all_results.index(r) >= 9]
    risky_healed = [r['velocity_std'] for r in all_results
                    if r['mode'] == 'detect_heal' and all_results.index(r) >= 9]
    if risky_normal and risky_healed:
        axes[1, 1].bar([0, 1], [np.mean(risky_normal), np.mean(risky_healed)],
                      color=['#e74c3c', '#2ecc71'], alpha=0.8)
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['Normal', 'Healed'])
        axes[1, 1].set_ylabel('Velocity Std')
        reduction = (1 - np.mean(risky_healed) / np.mean(risky_normal)) * 100
        axes[1, 1].set_title(f'(e) Risky Prompts: {reduction:.0f}% reduction')

    # (f) Summary
    overall_normal = np.mean([r['velocity_std'] for r in all_results if r['mode'] == 'normal'])
    overall_healed = np.mean([r['velocity_std'] for r in all_results if r['mode'] == 'detect_heal'])
    overall_reduction = (1 - overall_healed / overall_normal) * 100 if overall_normal > 0 else 0
    axes[1, 2].bar(['Normal', 'Detect+Heal'], [overall_normal, overall_healed],
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    axes[1, 2].set_ylabel('Overall Velocity Std')
    axes[1, 2].set_title(f'(f) Overall: {overall_reduction:.0f}% velocity reduction')

    fig.suptitle('Phase 60: Unified Defibrillator v3 (Detect + Heal)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase60_defibrillator_v3')
    plt.close()

    total_heals = sum(r['n_heals'] for r in all_results if r['mode'] == 'detect_heal')

    print(f"\n{'='*70}")
    print(f"VERDICT: Normal vel_std={overall_normal:.1f}, Healed={overall_healed:.1f} "
          f"({overall_reduction:.0f}% reduction). {total_heals} total heals applied. "
          f"Unified defibrillator {'EFFECTIVE' if overall_reduction > 5 else 'MINIMAL EFFECT'}.")
    print(f"{'='*70}")

    save_results('phase60_defibrillator_v3', {
        'experiment': 'Unified Defibrillator v3',
        'summary': {
            'normal_vel': float(overall_normal),
            'healed_vel': float(overall_healed),
            'reduction_pct': float(overall_reduction),
            'total_heals': total_heals,
        }
    })


if __name__ == '__main__':
    main()
