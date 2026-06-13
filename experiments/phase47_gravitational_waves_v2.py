# -*- coding: utf-8 -*-
"""
Phase 47: Gravitational Waves v2 (Auto-Regressive)
Fix Phase 46: Use auto-regressive generation instead of teacher-forcing
so contradictions propagate through the model's own generation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 47: Gravitational Waves v2 (Auto-Regressive)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    GEN_LENGTH = 40

    # Consistent premises -> generate continuation
    consistent = [
        "Water freezes at zero degrees Celsius. Ice is solid water. Therefore, when the temperature drops, water",
        "The Sun provides light and heat to Earth. Plants need sunlight to grow. In this way, the Sun is essential for",
        "Paris is the capital of France. The Eiffel Tower is in Paris. Many tourists visit Paris to see",
        "Gravity pulls objects toward Earth. Heavier objects experience more gravitational force. When you drop a ball, it",
        "DNA contains the genetic blueprint for living organisms. Genes encode proteins. Mutations in DNA can lead to",
    ]

    # Contradictory premises -> generate continuation
    contradictory = [
        "Water freezes at zero degrees Celsius. Water never freezes at any temperature. Therefore, when the temperature drops, water",
        "The Sun provides light and heat to Earth. The Sun provides no energy whatsoever. In this way, the Sun is essential for",
        "Paris is the capital of France. London is the capital of France. Many tourists visit the French capital to see",
        "Gravity pulls objects toward Earth. Gravity pushes objects away from Earth. When you drop a ball, it",
        "DNA contains the genetic blueprint for living organisms. DNA contains no information at all. Mutations in DNA can lead to",
    ]

    all_results = []

    for prompts, label in [(consistent, 'consistent'), (contradictory, 'contradictory')]:
        for prompt in prompts:
            input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
            current_ids = input_ids.clone()
            prt_trace = []

            for t in range(GEN_LENGTH):
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

                next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                current_ids = torch.cat([current_ids, next_id], dim=1)
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            prt_arr = np.array(prt_trace)
            velocity = np.diff(prt_arr)
            vel_std = float(np.std(velocity)) if len(velocity) > 0 else 0
            prt_std = float(np.std(prt_arr))

            # Oscillation analysis
            if len(prt_arr) > 4:
                detrended = prt_arr - np.mean(prt_arr)
                fft = np.abs(np.fft.rfft(detrended))
                total_power = float(np.sum(fft[1:] ** 2))
                peaks, _ = find_peaks(prt_arr, prominence=np.std(prt_arr) * 0.3)
                n_peaks = len(peaks)
            else:
                total_power = 0
                n_peaks = 0

            # Max velocity (strongest perturbation)
            vel_max = float(np.max(np.abs(velocity))) if len(velocity) > 0 else 0

            safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:45]
            print(f"  [{label}] '{safe_p}...' vel_std={vel_std:.1f}, "
                  f"prt_std={prt_std:.1f}, power={total_power:.0f}")

            all_results.append({
                'label': label,
                'prompt': prompt[:80],
                'prt_trace': [float(v) for v in prt_trace],
                'velocity_std': vel_std,
                'velocity_max': vel_max,
                'prt_std': prt_std,
                'total_power': total_power,
                'n_peaks': n_peaks,
                'is_contradiction': 1 if label == 'contradictory' else 0,
            })

    # ROC
    y_true = [r['is_contradiction'] for r in all_results]
    feature_aucs = {}
    for feat in ['velocity_std', 'velocity_max', 'prt_std', 'total_power', 'n_peaks']:
        scores = [r[feat] for r in all_results]
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.5
        feature_aucs[feat] = auc

    best_feat = max(feature_aucs, key=feature_aucs.get)
    best_auc = feature_aucs[best_feat]

    print(f"\nFeature AUCs (auto-regressive):")
    for f, a in sorted(feature_aucs.items(), key=lambda x: -x[1]):
        print(f"  {f}: {a:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) PRT traces
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'consistent' else '#e74c3c'
        axes[0].plot(r['prt_trace'], color=c, alpha=0.4, linewidth=0.8)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0],[0],color='#2ecc71',label='Consistent'),
                            Line2D([0],[0],color='#e74c3c',label='Contradictory')])
    axes[0].set_xlabel('Generated Token')
    axes[0].set_ylabel('PRT')
    axes[0].set_title('(a) Auto-Regressive PRT Traces')

    # (b) Velocity comparison
    cons_vels = [r['velocity_std'] for r in all_results if r['label'] == 'consistent']
    cont_vels = [r['velocity_std'] for r in all_results if r['label'] == 'contradictory']
    axes[1].boxplot([cons_vels, cont_vels], labels=['Consistent', 'Contradictory'])
    axes[1].set_ylabel('Velocity Std (dPRT/dt)')
    axes[1].set_title(f'(b) Velocity Separation')

    # (c) AUCs
    feats = sorted(feature_aucs.keys(), key=lambda x: feature_aucs[x])
    aucs_s = [feature_aucs[f] for f in feats]
    colors = ['#e74c3c' if f == best_feat else '#3498db' for f in feats]
    axes[2].barh(feats, aucs_s, color=colors, alpha=0.8)
    axes[2].axvline(x=0.5, color='gray', linestyle='--')
    axes[2].set_xlabel('ROC AUC')
    axes[2].set_title(f'(c) Detection AUCs (best={best_auc:.3f})')

    fig.suptitle('Phase 47: Gravitational Waves v2 (Auto-Regressive)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase47_gravitational_waves_v2')
    plt.close()

    # Verdict
    p46_best = 0.60  # Phase 46 best AUC
    improvement = best_auc - p46_best

    print(f"\n{'='*70}")
    print(f"VERDICT: Best='{best_feat}' AUC={best_auc:.3f} (vs Phase 46: {p46_best:.3f}, "
          f"{improvement:+.3f}). Auto-regressive {'IMPROVES' if improvement > 0.05 else 'comparable to'} "
          f"teacher-forcing for contradiction detection.")
    print(f"{'='*70}")

    save_results('phase47_gravitational_waves_v2', {
        'experiment': 'Gravitational Waves v2',
        'results': all_results,
        'feature_aucs': feature_aucs,
        'summary': {
            'best_feature': best_feat, 'best_auc': best_auc,
            'p46_best_auc': p46_best, 'improvement': improvement,
        }
    })


if __name__ == '__main__':
    main()
