# -*- coding: utf-8 -*-
"""
Phase 43: Kinematic Firewall
Detect hallucination via d(PRT)/dt (rate of change) instead of PRT variance.
Captures the 'inflationary' signature at hallucination onset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 43: Kinematic Firewall (dPRT/dt Detection)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    GEN_LENGTH = 60

    # Factual (should be stable PRT trajectory)
    factual_prompts = [
        "The Earth revolves around the Sun, completing one orbit every",
        "Water is composed of two hydrogen atoms and one oxygen atom, forming",
        "The speed of sound in air at room temperature is approximately",
        "Photosynthesis is the process by which plants convert sunlight into",
        "The human heart beats approximately 100,000 times per day, pumping",
    ]

    # Hallucination bait (should trigger PRT instability)
    halluc_prompts = [
        "Scientists confirmed yesterday that time travel is now possible because",
        "The secret 13th zodiac sign, Ophiuchus, grants its bearers the ability to",
        "Researchers at CERN accidentally opened a portal to another dimension when",
        "A newly discovered element, Pandemonium, has the unique property of",
        "The ancient civilization of Lemuria left behind technology that proves",
    ]

    all_metrics = []

    for prompts, label in [(factual_prompts, 'factual'), (halluc_prompts, 'hallucination')]:
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
                prt_trace.append(PRT)

                next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                current_ids = torch.cat([current_ids, next_id], dim=1)
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            prt_arr = np.array(prt_trace)

            # Compute kinematic features
            # 1. Velocity: d(PRT)/dt
            velocity = np.diff(prt_arr)
            # 2. Acceleration: d2(PRT)/dt2
            acceleration = np.diff(velocity)
            # 3. Jerk: d3(PRT)/dt3
            jerk = np.diff(acceleration) if len(acceleration) > 1 else np.array([0])

            # Feature extraction
            vel_std = np.std(velocity) if len(velocity) > 0 else 0
            vel_max = np.max(np.abs(velocity)) if len(velocity) > 0 else 0
            acc_std = np.std(acceleration) if len(acceleration) > 0 else 0
            acc_max = np.max(np.abs(acceleration)) if len(acceleration) > 0 else 0
            jerk_std = np.std(jerk) if len(jerk) > 0 else 0

            # PRT variance (baseline method from Phase 29)
            prt_cv = np.std(prt_arr) / (np.mean(prt_arr) + 1e-10)

            # Inflation detector: count sudden jumps in velocity
            vel_threshold = np.mean(np.abs(velocity)) + 2 * np.std(np.abs(velocity)) if len(velocity) > 2 else float('inf')
            inflation_events = int(np.sum(np.abs(velocity) > vel_threshold))

            safe_prompt = prompt.encode('ascii', errors='replace').decode('ascii')[:45]
            print(f"  [{label}] '{safe_prompt}...' vel_std={vel_std:.1f}, "
                  f"acc_std={acc_std:.1f}, inflations={inflation_events}")

            all_metrics.append({
                'label': label,
                'prompt': prompt[:60],
                'prt_trace': [float(v) for v in prt_trace],
                'velocity_std': float(vel_std),
                'velocity_max': float(vel_max),
                'acceleration_std': float(acc_std),
                'acceleration_max': float(acc_max),
                'jerk_std': float(jerk_std),
                'prt_cv': float(prt_cv),
                'inflation_events': inflation_events,
                'is_halluc': 1 if label == 'hallucination' else 0,
            })

    # === ROC Analysis ===
    y_true = [m['is_halluc'] for m in all_metrics]

    feature_aucs = {}
    for feat in ['velocity_std', 'acceleration_std', 'jerk_std', 'prt_cv', 'inflation_events']:
        scores = [m[feat] for m in all_metrics]
        try:
            auc = roc_auc_score(y_true, scores)
            feature_aucs[feat] = auc
        except ValueError:
            feature_aucs[feat] = 0.5
        print(f"  Feature '{feat}': AUC={feature_aucs[feat]:.3f}")

    best_feature = max(feature_aucs, key=feature_aucs.get)
    best_auc = feature_aucs[best_feature]

    # Compare with Phase 29 baseline (prt_cv)
    baseline_auc = feature_aucs.get('prt_cv', 0.5)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) PRT traces: factual vs hallucination
    for m in all_metrics:
        color = '#2ecc71' if m['label'] == 'factual' else '#e74c3c'
        alpha = 0.4
        axes[0, 0].plot(m['prt_trace'], color=color, alpha=alpha, linewidth=0.8)
    axes[0, 0].set_xlabel('Token')
    axes[0, 0].set_ylabel('PRT')
    axes[0, 0].set_title('(a) PRT Traces')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2ecc71', label='Factual'),
                       Line2D([0], [0], color='#e74c3c', label='Hallucination')]
    axes[0, 0].legend(handles=legend_elements)

    # (b) Velocity distributions
    fact_vels = []
    hall_vels = []
    for m in all_metrics:
        vels = np.diff(m['prt_trace'])
        if m['label'] == 'factual':
            fact_vels.extend(vels.tolist())
        else:
            hall_vels.extend(vels.tolist())
    axes[0, 1].hist(fact_vels, bins=30, alpha=0.6, color='#2ecc71', label='Factual', density=True)
    axes[0, 1].hist(hall_vels, bins=30, alpha=0.6, color='#e74c3c', label='Halluc', density=True)
    axes[0, 1].set_xlabel('d(PRT)/dt')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('(b) Velocity Distribution')
    axes[0, 1].legend()

    # (c) Feature AUCs
    feats = list(feature_aucs.keys())
    aucs = [feature_aucs[f] for f in feats]
    colors = ['#e74c3c' if f == best_feature else '#3498db' for f in feats]
    axes[0, 2].barh(feats, aucs, color=colors, alpha=0.8)
    axes[0, 2].axvline(x=0.5, color='gray', linestyle='--')
    axes[0, 2].set_xlabel('ROC AUC')
    axes[0, 2].set_title('(c) Feature Discriminability')

    # (d-f) Example traces with derivatives
    for idx, (m, ax_row) in enumerate(zip(all_metrics[:3], [axes[1, 0], axes[1, 1], axes[1, 2]])):
        prt = np.array(m['prt_trace'])
        vel = np.diff(prt)
        ax_row.plot(prt, color='#3498db', label='PRT', alpha=0.8)
        ax2 = ax_row.twinx()
        ax2.plot(range(1, len(prt)), vel, color='#e74c3c', alpha=0.6, label='dPRT/dt')
        ax_row.set_xlabel('Token')
        ax_row.set_ylabel('PRT', color='#3498db')
        ax2.set_ylabel('Velocity', color='#e74c3c')
        short_prompt = m['prompt'][:25].encode('ascii', errors='replace').decode('ascii')
        ax_row.set_title(f'({"def"[idx]}) {m["label"]}: {short_prompt}...', fontsize=8)

    fig.suptitle('Phase 43: Kinematic Firewall', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase43_kinematic_firewall')
    plt.close()

    # === Verdict ===
    improvement = best_auc - baseline_auc

    print(f"\n{'='*70}")
    print(f"VERDICT: Best kinematic feature='{best_feature}' (AUC={best_auc:.3f}) "
          f"vs baseline prt_cv (AUC={baseline_auc:.3f}). "
          f"{'IMPROVED' if improvement > 0.02 else 'Comparable'} by {improvement:+.3f}.")
    print(f"{'='*70}")

    save_results('phase43_kinematic_firewall', {
        'experiment': 'Kinematic Firewall',
        'metrics': all_metrics,
        'feature_aucs': feature_aucs,
        'summary': {
            'best_feature': best_feature,
            'best_auc': best_auc,
            'baseline_auc': baseline_auc,
            'improvement': improvement,
        }
    })


if __name__ == '__main__':
    main()
