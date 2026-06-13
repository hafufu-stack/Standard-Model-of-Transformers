# -*- coding: utf-8 -*-
"""
Phase 41b: Defibrillator v2
Fix: Use running-average prototype for adaptive steering instead of static calibration.
Also track defib activation rate and text quality metrics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 41b: Defibrillator v2 (Adaptive Steering)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    BETA_DEFIB = 0.57
    DEFIB_DURATION = 3
    PRT_WINDOW = 8
    GEN_LENGTH = 80
    STEER_ALPHA = 0.05  # Gentler than v1
    EMA_DECAY = 0.9     # For running average prototype

    test_cases = [
        ("factual", "The periodic table of elements organizes all known chemical elements by their atomic number. The first element is"),
        ("factual", "In computer science, a binary search algorithm works by repeatedly dividing the search interval in half. It requires"),
        ("halluc_bait", "A team of archaeologists discovered a functioning computer from 50,000 BC that was powered by"),
        ("halluc_bait", "Scientists at MIT have confirmed that telepathy is real and have developed a device that"),
        ("mixed", "While renewable energy sources like solar and wind are growing, some experts controversially claim that"),
        ("mixed", "The Amazon rainforest, often called the lungs of the Earth, has been found to contain traces of"),
    ]

    all_results = []

    for cat, prompt in test_cases:
        print(f"\n--- [{cat}] '{prompt[:50]}...' ---")
        input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)

        modes_data = {}

        for mode in ['normal', 'cooling', 'adaptive_steer']:
            # State variables
            ffn_scale = [1.0]
            steer_vec = [None]  # Running average prototype
            steer_active = [False]
            hooks = []

            # FFN cooling hook
            def make_ffn_hook():
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if ffn_scale[0] < 1.0:
                        h = (h.float() * ffn_scale[0]).to(h.dtype)
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook

            # Adaptive steering hook - uses EMA prototype
            def make_adaptive_steer_hook(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    current = h[0, -1, :].float().detach()

                    # Update EMA prototype (always)
                    if steer_vec[0] is None:
                        steer_vec[0] = current.clone()
                    else:
                        steer_vec[0] = EMA_DECAY * steer_vec[0] + (1 - EMA_DECAY) * current

                    # Steer only when active
                    if steer_active[0] and steer_vec[0] is not None:
                        direction = steer_vec[0] - current
                        dir_norm = direction.norm()
                        if dir_norm > 1e-6:
                            direction = direction / dir_norm
                            h_mod = h.clone()
                            nudge = (STEER_ALPHA * direction * current.norm()).to(h.dtype)
                            h_mod[0, -1, :] = h_mod[0, -1, :] + nudge
                            if isinstance(output, tuple):
                                return (h_mod,) + output[1:]
                            return h_mod
                    return output
                return hook

            if mode in ('cooling', 'adaptive_steer'):
                for li in range(len(model.model.layers)):
                    h = model.model.layers[li].mlp.register_forward_hook(make_ffn_hook())
                    hooks.append(h)

            if mode == 'adaptive_steer':
                # Steer at layers 10 and 16 (fact-extraction registers)
                for li in [10, 16]:
                    if li < len(model.model.layers):
                        h = model.model.layers[li].register_forward_hook(
                            make_adaptive_steer_hook(li))
                        hooks.append(h)

            # Generate
            current_ids = input_ids.clone()
            prt_history = []
            defib_count = 0
            defib_active = 0
            generated_tokens = []

            for t in range(GEN_LENGTH):
                with torch.no_grad():
                    out = model(current_ids)
                    logits = out.logits[0, -1, :].float()

                probs = torch.softmax(logits, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T_val
                prt_history.append(PRT)

                # Defibrillator trigger based on velocity (Phase 43 finding!)
                if mode != 'normal' and len(prt_history) >= 3:
                    velocity = abs(prt_history[-1] - prt_history[-2])
                    # Adaptive threshold from recent history
                    if len(prt_history) >= PRT_WINDOW:
                        vels = [abs(prt_history[j] - prt_history[j-1])
                               for j in range(max(1, len(prt_history)-PRT_WINDOW), len(prt_history))]
                        vel_threshold = np.mean(vels) + 2.0 * np.std(vels)
                    else:
                        vel_threshold = velocity * 3

                    if velocity > vel_threshold and defib_active <= 0:
                        ffn_scale[0] = BETA_DEFIB
                        steer_active[0] = True
                        defib_active = DEFIB_DURATION
                        defib_count += 1
                    elif defib_active > 0:
                        defib_active -= 1
                        if defib_active == 0:
                            ffn_scale[0] = 1.0
                            steer_active[0] = False

                next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                generated_tokens.append(next_id.item())
                current_ids = torch.cat([current_ids, next_id], dim=1)
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            for h in hooks:
                h.remove()

            text = tok.decode(generated_tokens, skip_special_tokens=True)
            prt_std = np.std(prt_history) if prt_history else 0
            prt_mean = np.mean(prt_history) if prt_history else 0

            # Compute velocity std (the key metric from Phase 43)
            if len(prt_history) > 1:
                velocity_std = np.std(np.diff(prt_history))
            else:
                velocity_std = 0

            safe_text = text.encode('ascii', errors='replace').decode('ascii')[:50]
            print(f"  [{mode}] PRT_std={prt_std:.1f}, vel_std={velocity_std:.1f}, "
                  f"defib={defib_count}, '{safe_text}...'")

            modes_data[mode] = {
                'prt_std': float(prt_std), 'prt_mean': float(prt_mean),
                'velocity_std': float(velocity_std),
                'defib_count': defib_count,
                'text': text[:200],
                'prt_history': [float(v) for v in prt_history],
            }

        all_results.append({
            'category': cat, 'prompt': prompt[:80],
            **modes_data,
        })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, result in enumerate(all_results[:6]):
        ax = axes[idx // 3][idx % 3]
        for mode, color, ls in [('normal', '#e74c3c', '-'),
                                 ('cooling', '#3498db', '--'),
                                 ('adaptive_steer', '#2ecc71', '-')]:
            if mode in result and 'prt_history' in result[mode]:
                ax.plot(result[mode]['prt_history'], color=color, alpha=0.7,
                       label=mode, linewidth=0.8, linestyle=ls)
        short = result['prompt'][-25:].encode('ascii', errors='replace').decode('ascii')
        ax.set_title(f"[{result['category']}] ...{short}", fontsize=8)
        ax.set_xlabel('Token')
        ax.set_ylabel('PRT')
        if idx == 0:
            ax.legend(fontsize=6)

    fig.suptitle('Phase 41b: Defibrillator v2 (Velocity-Triggered + EMA Steering)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase41b_defibrillator')
    plt.close()

    # === Verdict ===
    normal_vel = np.mean([r['normal']['velocity_std'] for r in all_results])
    cooling_vel = np.mean([r['cooling']['velocity_std'] for r in all_results])
    steer_vel = np.mean([r['adaptive_steer']['velocity_std'] for r in all_results])
    cool_pct = (1 - cooling_vel / (normal_vel + 1e-10)) * 100
    steer_pct = (1 - steer_vel / (normal_vel + 1e-10)) * 100

    print(f"\n{'='*70}")
    print(f"VERDICT: Velocity_std: Normal={normal_vel:.1f}, Cooling={cooling_vel:.1f} "
          f"({cool_pct:.0f}% reduction), Adaptive={steer_vel:.1f} ({steer_pct:.0f}% reduction). "
          f"{'SUCCESS' if steer_pct > 10 else 'Marginal effect'}.")
    print(f"{'='*70}")

    save_results('phase41b_defibrillator', {
        'experiment': 'Defibrillator v2',
        'results': all_results,
        'summary': {
            'normal_velocity_std': normal_vel,
            'cooling_velocity_std': cooling_vel,
            'adaptive_velocity_std': steer_vel,
            'cooling_reduction_pct': cool_pct,
            'adaptive_reduction_pct': steer_pct,
        }
    })


if __name__ == '__main__':
    main()
