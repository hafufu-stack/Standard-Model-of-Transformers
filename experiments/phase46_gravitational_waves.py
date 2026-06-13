# -*- coding: utf-8 -*-
"""
Phase 46: Gravitational Wave Detection
Detect contradictions in text via PRT oscillation patterns.
Contradictory information creates 'gravitational waves' in the thermodynamic field.
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
    print("Phase 46: Gravitational Wave Detection (Opus)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Consistent statements (no contradiction)
    consistent = [
        "Water freezes at 0 degrees Celsius. Ice forms when the temperature drops below zero. This process is called",
        "The Sun is a star at the center of our solar system. All planets orbit around this star. The closest planet is",
        "Paris is the capital of France. The Eiffel Tower is located in Paris. This landmark was built in",
        "Dogs are mammals that have been domesticated for thousands of years. They are known for their loyalty and",
        "The Pacific Ocean is the largest ocean on Earth. It covers more area than all the land combined and",
    ]

    # Contradictory statements (logical conflict embedded)
    contradictory = [
        "Water freezes at 0 degrees Celsius. Water never freezes at any temperature. This means that ice is",
        "The Sun is a star at the center of our solar system. The Earth is the center of the universe and the Sun orbits us. This shows that",
        "Paris is the capital of France. Berlin is the capital of France. The true capital is definitely",
        "Dogs are mammals. Dogs are actually reptiles with cold blood. This explains why dogs need to",
        "The Pacific Ocean is the largest ocean. The Atlantic is the largest ocean and the Pacific doesn't exist. Swimming in the",
    ]

    all_results = []

    for texts, label in [(consistent, 'consistent'), (contradictory, 'contradictory')]:
        for text in texts:
            input_ids = tok(text, return_tensors='pt')['input_ids'].to(device)
            seq_len = input_ids.shape[1]

            # Measure PRT at each token position
            prt_per_token = []

            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)

            # For each token position, compute PRT from its logits
            for pos in range(seq_len):
                if pos < seq_len - 1:
                    # Use teacher-forcing logits
                    logits = out.logits[0, pos, :].float()
                else:
                    logits = out.logits[0, -1, :].float()

                probs = torch.softmax(logits, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T_val

                if np.isnan(PRT):
                    PRT = 0
                prt_per_token.append(PRT)

            prt_arr = np.array(prt_per_token)

            # Gravitational wave analysis
            # 1. Oscillation: compute FFT of PRT signal
            if len(prt_arr) > 4:
                prt_detrended = prt_arr - np.mean(prt_arr)
                fft_vals = np.abs(np.fft.rfft(prt_detrended))
                fft_freqs = np.fft.rfftfreq(len(prt_detrended))

                # Dominant frequency (exclude DC)
                if len(fft_vals) > 1:
                    dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
                    dominant_freq = fft_freqs[dominant_freq_idx]
                    dominant_power = fft_vals[dominant_freq_idx]
                else:
                    dominant_freq = 0
                    dominant_power = 0

                # Total oscillation power
                total_power = np.sum(fft_vals[1:] ** 2)

                # Peak count in PRT signal
                peaks, peak_props = find_peaks(prt_arr, prominence=np.std(prt_arr) * 0.5)
                n_peaks = len(peaks)
                peak_rate = n_peaks / len(prt_arr)
            else:
                dominant_freq = dominant_power = total_power = 0
                n_peaks = peak_rate = 0

            # 2. Derivative features
            velocity = np.diff(prt_arr)
            vel_std = np.std(velocity) if len(velocity) > 0 else 0
            vel_max = np.max(np.abs(velocity)) if len(velocity) > 0 else 0

            # 3. Local variance (sliding window)
            window = 5
            local_vars = []
            for i in range(len(prt_arr) - window):
                local_vars.append(np.var(prt_arr[i:i+window]))
            max_local_var = max(local_vars) if local_vars else 0
            mean_local_var = np.mean(local_vars) if local_vars else 0

            safe_text = text.encode('ascii', errors='replace').decode('ascii')[:45]
            print(f"  [{label}] '{safe_text}...' "
                  f"power={total_power:.0f}, peaks={n_peaks}, vel_std={vel_std:.1f}")

            all_results.append({
                'label': label,
                'text': text[:80],
                'prt_trace': [float(v) for v in prt_per_token],
                'total_power': float(total_power),
                'dominant_freq': float(dominant_freq),
                'dominant_power': float(dominant_power),
                'n_peaks': int(n_peaks),
                'peak_rate': float(peak_rate),
                'velocity_std': float(vel_std),
                'velocity_max': float(vel_max),
                'max_local_var': float(max_local_var),
                'mean_local_var': float(mean_local_var),
                'prt_std': float(np.std(prt_arr)),
                'is_contradiction': 1 if label == 'contradictory' else 0,
            })

    # === ROC Analysis ===
    y_true = [r['is_contradiction'] for r in all_results]
    feature_aucs = {}
    for feat in ['total_power', 'n_peaks', 'peak_rate', 'velocity_std',
                 'max_local_var', 'prt_std', 'velocity_max']:
        scores = [r[feat] for r in all_results]
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.5
        feature_aucs[feat] = auc

    best_feat = max(feature_aucs, key=feature_aucs.get)
    best_auc = feature_aucs[best_feat]

    print(f"\nFeature AUCs:")
    for f, a in sorted(feature_aucs.items(), key=lambda x: -x[1]):
        print(f"  {f}: {a:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) PRT traces
    for r in all_results:
        color = '#2ecc71' if r['label'] == 'consistent' else '#e74c3c'
        axes[0, 0].plot(r['prt_trace'], color=color, alpha=0.4, linewidth=0.8)
    axes[0, 0].set_xlabel('Token Position')
    axes[0, 0].set_ylabel('PRT')
    axes[0, 0].set_title('(a) PRT Signals')
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0], [0], color='#2ecc71', label='Consistent'),
                 Line2D([0], [0], color='#e74c3c', label='Contradictory')]
    axes[0, 0].legend(handles=legend_el, fontsize=8)

    # (b) FFT power comparison
    cons_power = [r['total_power'] for r in all_results if r['label'] == 'consistent']
    cont_power = [r['total_power'] for r in all_results if r['label'] == 'contradictory']
    axes[0, 1].boxplot([cons_power, cont_power], labels=['Consistent', 'Contradictory'])
    axes[0, 1].set_ylabel('Total Oscillation Power')
    axes[0, 1].set_title('(b) Gravitational Wave Power')

    # (c) Feature AUCs
    feats = sorted(feature_aucs.keys(), key=lambda x: feature_aucs[x])
    aucs_sorted = [feature_aucs[f] for f in feats]
    colors_bar = ['#e74c3c' if f == best_feat else '#3498db' for f in feats]
    axes[0, 2].barh(feats, aucs_sorted, color=colors_bar, alpha=0.8)
    axes[0, 2].axvline(x=0.5, color='gray', linestyle='--')
    axes[0, 2].set_xlabel('ROC AUC')
    axes[0, 2].set_title('(c) Detection AUCs')

    # (d-f) Example FFTs
    for idx in range(min(3, len(all_results))):
        ax = axes[1, idx]
        r = all_results[idx]
        prt = np.array(r['prt_trace'])
        prt_d = prt - np.mean(prt)
        fft = np.abs(np.fft.rfft(prt_d))
        freqs = np.fft.rfftfreq(len(prt_d))
        color = '#2ecc71' if r['label'] == 'consistent' else '#e74c3c'
        ax.plot(freqs[1:], fft[1:], color=color, linewidth=1.5)
        ax.fill_between(freqs[1:], fft[1:], alpha=0.2, color=color)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('FFT Amplitude')
        short = r['text'][:20].encode('ascii', errors='replace').decode('ascii')
        ax.set_title(f'({"def"[idx]}) {r["label"]}: {short}...', fontsize=8)

    fig.suptitle('Phase 46: Gravitational Wave Detection (Opus)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase46_gravitational_waves')
    plt.close()

    # === Verdict ===
    print(f"\n{'='*70}")
    print(f"VERDICT: Best contradiction detector='{best_feat}' (AUC={best_auc:.3f}). "
          f"Gravitational waves {'DETECTED' if best_auc > 0.7 else 'not clearly detected'} "
          f"in contradictory text.")
    print(f"{'='*70}")

    save_results('phase46_gravitational_waves', {
        'experiment': 'Gravitational Wave Detection',
        'results': all_results,
        'feature_aucs': feature_aucs,
        'summary': {
            'best_feature': best_feat,
            'best_auc': best_auc,
        }
    })


if __name__ == '__main__':
    main()
