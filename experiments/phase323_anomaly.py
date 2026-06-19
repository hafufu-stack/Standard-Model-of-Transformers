# -*- coding: utf-8 -*-
"""
Phase 323: Anomaly Cancellation -- Consistency Conditions
==========================================================
In gauge theories, anomaly cancellation ensures consistency.
Anomalies = violations of classical symmetries by quantum effects.
Test if the transformer obeys analogous consistency conditions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_anomaly(model, tok, prompt, device):
    """Test anomaly cancellation in transformer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Anomaly 1: Trace anomaly
    # In CFT, trace of stress-energy tensor T^mu_mu = 0 (conformal)
    # Non-zero trace = anomaly
    trace_anomalies = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0].float()  # (seq, D)
        # "Stress-energy" = h @ h.T (correlation matrix)
        T = h @ h.T  # (seq, seq)
        trace = float(torch.trace(T).item())
        norm = float(torch.norm(T).item())
        trace_anomaly = trace / (norm + 1e-10)
        trace_anomalies.append(float(trace_anomaly))

    # Anomaly 2: Chiral anomaly
    # Left-right asymmetry in hidden state spectrum
    chiral_anomalies = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        D = len(h)
        h_left = h[:D//2]
        h_right = h[D//2:]
        asym = float((h_left.norm() - h_right.norm()) / (h_left.norm() + h_right.norm() + 1e-10))
        chiral_anomalies.append(asym)

    # Anomaly cancellation: do anomalies sum to zero across layers?
    trace_sum = float(np.sum(trace_anomalies))
    chiral_sum = float(np.sum(chiral_anomalies))

    # Alternating sign pattern (anomaly cancellation mechanism)
    trace_alt = sum((-1)**i * t for i, t in enumerate(trace_anomalies))
    chiral_alt = sum((-1)**i * c for i, c in enumerate(chiral_anomalies))

    return {
        'trace_anomalies': [round(t, 4) for t in trace_anomalies],
        'chiral_anomalies': [round(c, 4) for c in chiral_anomalies],
        'trace_sum': round(trace_sum, 4),
        'chiral_sum': round(chiral_sum, 4),
        'trace_alternating': round(float(trace_alt), 4),
        'chiral_alternating': round(float(chiral_alt), 4),
        'trace_cancelled': abs(trace_sum / (len(trace_anomalies) * np.mean(np.abs(trace_anomalies)) + 1e-10)) < 0.1,
        'chiral_cancelled': abs(chiral_sum / (len(chiral_anomalies) * np.mean(np.abs(chiral_anomalies)) + 1e-10)) < 0.1,
    }


def main():
    print("=" * 70)
    print("Phase 323: Anomaly Cancellation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        an_data = []
        for prompt in PROMPTS:
            a = measure_anomaly(model, tok, prompt, device)
            an_data.append(a)

        n = len(an_data[0]['trace_anomalies'])
        avg_trace = [float(np.mean([a['trace_anomalies'][i] for a in an_data])) for i in range(n)]
        avg_chiral = [float(np.mean([a['chiral_anomalies'][i] for a in an_data])) for i in range(n)]

        all_results[size] = {
            'avg_trace_anomaly': [round(t, 4) for t in avg_trace],
            'avg_chiral_anomaly': [round(c, 4) for c in avg_chiral],
            'trace_sum': round(float(np.mean([a['trace_sum'] for a in an_data])), 4),
            'chiral_sum': round(float(np.mean([a['chiral_sum'] for a in an_data])), 4),
            'trace_cancelled': sum(1 for a in an_data if a['trace_cancelled']) >= 3,
            'chiral_cancelled': sum(1 for a in an_data if a['chiral_cancelled']) >= 3,
        }
        tc = 'YES' if all_results[size]['trace_cancelled'] else 'NO'
        cc = 'YES' if all_results[size]['chiral_cancelled'] else 'NO'
        print(f"  Trace cancellation: {tc}")
        print(f"  Chiral cancellation: {cc}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_trace_anomaly'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Trace Anomaly')
    axes[0, 0].set_title('(a) Trace Anomaly', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_chiral_anomaly'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Chiral Anomaly')
    axes[0, 1].set_title('(b) Chiral Anomaly', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.35
    axes[0, 2].bar(x - w/2, [all_results[s]['trace_sum'] for s in sizes], w,
                  label='Trace', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['chiral_sum'] for s in sizes], w,
                  label='Chiral', color='#e74c3c')
    axes[0, 2].axhline(0, color='gold', ls='--', lw=2)
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_ylabel('Sum'); axes[0, 2].set_title('(c) Anomaly Sums', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "ANOMALY CANCELLATION\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        tc = 'YES' if d['trace_cancelled'] else 'NO'
        cc = 'YES' if d['chiral_cancelled'] else 'NO'
        txt += f"  Trace: {tc}\n"
        txt += f"  Chiral: {cc}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 323: Anomaly Cancellation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase323_anomaly')
    plt.close()
    save_results('phase323_anomaly', {'experiment': 'Anomaly Cancellation', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
