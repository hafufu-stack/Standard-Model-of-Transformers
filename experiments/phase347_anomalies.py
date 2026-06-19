# -*- coding: utf-8 -*-
"""
Phase 347: Anomalies -- Chiral and Gravitational
=====================================================
In QFT, anomalies are quantum-mechanical violations of classical
symmetries. The chiral anomaly breaks chiral symmetry; the trace
anomaly gives non-zero trace of the stress-energy tensor.
Test whether the Transformer exhibits analogous anomalies.
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

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_anomalies(model, tok, prompt, device):
    """Measure quantum anomalies in Transformer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Chiral anomaly: split hidden state into "left" and "right" sectors
    # Test if L-R symmetry is broken at the quantum level
    chiral_asymmetries = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        half = dim // 2
        h_L = h[:half]
        h_R = h[half:2*half]

        # L-R asymmetry
        norm_L = float(torch.norm(h_L).item())
        norm_R = float(torch.norm(h_R).item())
        asymmetry = abs(norm_L - norm_R) / (norm_L + norm_R + 1e-10)
        chiral_asymmetries.append(round(float(asymmetry), 4))

    # Chiral current: J5 = <psi_L|psi_L> - <psi_R|psi_R>
    chiral_currents = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        half = dim // 2
        J5 = float(torch.sum(h[:half]**2).item() - torch.sum(h[half:2*half]**2).item())
        J5_normalized = J5 / (float(torch.sum(h**2).item()) + 1e-10)
        chiral_currents.append(round(float(J5_normalized), 4))

    # Non-conservation of chiral current: dJ5/dl != 0
    dJ5 = []
    for li in range(len(chiral_currents) - 1):
        dj = chiral_currents[li + 1] - chiral_currents[li]
        dJ5.append(round(float(dj), 6))

    chiral_anomaly_strength = float(np.std(dJ5)) if dJ5 else 0

    # 2. Trace anomaly: Tr(T_mu^mu) != 0
    # In CFT, trace anomaly = c/12 * R (Ricci scalar)
    # Use the stress-energy trace computed from energy-momentum
    trace_anomalies = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Trace of "stress tensor": sum of diagonal elements of rho
        rho_diag = h**2 / (torch.sum(h**2) + 1e-30)
        trace = float(torch.sum(rho_diag).item())  # Should be 1
        # Anomaly is deviation from expected scaling
        expected = 1.0 / dim
        anomaly = float(torch.var(rho_diag).item()) * dim  # Rescaled variance
        trace_anomalies.append(round(float(anomaly), 6))

    # 3. ABJ anomaly: rate of change of topological charge
    # Q = (1/2pi) * integral F, where F is Berry curvature
    half = min(dim // 2, 256)
    topological_charges = []
    for li in range(n_layers):
        z1 = hiddens[li][:half].numpy() + 1j * hiddens[li][half:2*half].numpy()
        z2 = hiddens[li + 1][:half].numpy() + 1j * hiddens[li + 1][half:2*half].numpy()
        inner = np.sum(z1.conj() * z2)
        phase = float(np.angle(inner))
        topological_charges.append(round(phase / (2 * np.pi), 4))

    # Total topological charge
    Q_total = float(np.sum(topological_charges))

    return {
        'chiral_asymmetries': chiral_asymmetries,
        'chiral_currents': chiral_currents,
        'dJ5': dJ5,
        'chiral_anomaly_strength': round(float(chiral_anomaly_strength), 6),
        'trace_anomalies': trace_anomalies,
        'avg_trace_anomaly': round(float(np.mean(trace_anomalies)), 6),
        'topological_charges': topological_charges,
        'Q_total': round(float(Q_total), 4),
        'chiral_anomaly_present': chiral_anomaly_strength > 0.001,
    }


def main():
    print("=" * 70)
    print("Phase 347: Anomalies")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        anom_data = []
        for prompt in PROMPTS:
            a = measure_anomalies(model, tok, prompt, device)
            anom_data.append(a)

        n_ca = len(anom_data[0]['chiral_asymmetries'])
        n_ta = len(anom_data[0]['trace_anomalies'])
        all_results[size] = {
            'chiral_asymmetries': [round(float(np.mean([a['chiral_asymmetries'][i] for a in anom_data])), 4)
                                  for i in range(n_ca)],
            'chiral_anomaly_strength': round(float(np.mean([a['chiral_anomaly_strength'] for a in anom_data])), 6),
            'trace_anomalies': [round(float(np.mean([a['trace_anomalies'][i] for a in anom_data])), 6)
                               for i in range(n_ta)],
            'avg_trace_anomaly': round(float(np.mean([a['avg_trace_anomaly'] for a in anom_data])), 6),
            'Q_total': round(float(np.mean([a['Q_total'] for a in anom_data])), 4),
            'chiral_present': sum(1 for a in anom_data if a['chiral_anomaly_present']) >= 4,
        }
        present = 'YES' if all_results[size]['chiral_present'] else 'NO'
        print(f"  Chiral anomaly: {all_results[size]['chiral_anomaly_strength']:.6f}")
        print(f"  Trace anomaly: {all_results[size]['avg_trace_anomaly']:.6f}")
        print(f"  Q_total: {all_results[size]['Q_total']:.4f}")
        print(f"  Chiral present: {present}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['chiral_asymmetries'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('L-R Asymmetry')
    axes[0, 0].set_title('(a) Chiral Asymmetry', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['trace_anomalies'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Trace anomaly')
    axes[0, 1].set_title('(b) Trace Anomaly', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    axes[0, 2].bar(sizes, [all_results[s]['Q_total'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Q_total')
    axes[0, 2].set_title('(c) Topological Charge', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['chiral_anomaly_strength'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_title('(d) Chiral Anomaly Strength', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "QUANTUM ANOMALIES\n\n"
    for s in sizes:
        d = all_results[s]
        p = 'YES' if d['chiral_present'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  chiral = {d['chiral_anomaly_strength']:.5f}\n"
        txt += f"  trace = {d['avg_trace_anomaly']:.5f}\n"
        txt += f"  Q = {d['Q_total']:.3f}\n"
        txt += f"  present: {p}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 347: Quantum Anomalies", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase347_anomalies')
    plt.close()
    save_results('phase347_anomalies', {'experiment': 'Anomalies', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
