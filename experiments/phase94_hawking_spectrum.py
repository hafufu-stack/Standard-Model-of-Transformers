# -*- coding: utf-8 -*-
"""
Phase 94: Hawking Radiation Spectrum Analysis
Phase 87 detected 54 Hawking spikes. Analyze their spectral distribution
to test if it follows a Planckian (thermal) radiation law.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The history of mathematics spans thousands of years",
    "Quantum computing promises to revolutionize information",
    "The Amazon rainforest contains the greatest biodiversity",
    "The number one is followed by one which is followed by",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
    "This sentence is a sentence that is a sentence that is",
    "Once upon a time there was a story about a story about",
    "The word the is the most common word in the English",
    "One fish two fish red fish blue fish green fish yellow",
    "To be or not to be that is the question whether tis nobler",
]


def planck_law(E, A, kT):
    """Planck-like thermal distribution."""
    return A * E**2 / (np.exp(E / (kT + 1e-10)) - 1 + 1e-10)


def boltzmann_law(E, A, kT):
    """Boltzmann distribution for comparison."""
    return A * np.exp(-E / (kT + 1e-10))


def iterative_feed(model, tok, device, prompt, n_iters=150):
    """Feed output back iteratively, track detailed T profile."""
    t_trace = []
    current_text = prompt

    for i in range(n_iters):
        inp = tok(current_text, return_tensors='pt', truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(T):
            T = 0.0
        t_trace.append(T)

        next_id = torch.argmax(logits).item()
        current_text = current_text + tok.decode([next_id])

    return t_trace


def detect_spikes(t_trace, window=5):
    """Detect all T spikes (radiation events)."""
    spikes = []
    for i in range(window, len(t_trace)):
        local_mean = np.mean(t_trace[max(0, i-window):i])
        local_std = np.std(t_trace[max(0, i-window):i])
        if local_std > 0 and (t_trace[i] - local_mean) / (local_std + 1e-10) > 2.0:
            # Spike detected (2-sigma above local mean)
            spikes.append({
                'iteration': i,
                'T_spike': float(t_trace[i]),
                'T_local_mean': float(local_mean),
                'T_local_std': float(local_std),
                'amplitude': float(t_trace[i] - local_mean),
                'sigma': float((t_trace[i] - local_mean) / (local_std + 1e-10)),
            })
    return spikes


def main():
    print("=" * 70)
    print("Phase 94: Hawking Radiation Spectrum Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    all_spikes = []
    all_traces = []

    for prompt in PROMPTS:
        print(f"  {prompt[:40]}...")
        trace = iterative_feed(model, tok, device, prompt, n_iters=150)
        spikes = detect_spikes(trace)
        all_spikes.extend(spikes)
        all_traces.append({'prompt': prompt[:40], 'trace': [float(t) for t in trace],
                          'n_spikes': len(spikes)})
        print(f"    {len(spikes)} spikes detected")

    print(f"\n  Total spikes: {len(all_spikes)}")

    if len(all_spikes) < 5:
        print("  Not enough spikes for spectral analysis")
        save_results('phase94_hawking_spectrum', {
            'experiment': 'Hawking Spectrum', 'summary': {'n_spikes': len(all_spikes), 'verdict': 'INSUFFICIENT'}})
        return

    # === Spectral analysis ===
    amplitudes = np.array([s['amplitude'] for s in all_spikes])
    sigmas = np.array([s['sigma'] for s in all_spikes])

    # Fit Boltzmann to amplitude distribution
    hist, edges = np.histogram(amplitudes[amplitudes > 0], bins=20, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    bc = centers[mask]
    hv = hist[mask]

    boltz_r2 = 0
    planck_r2 = 0
    boltz_kT = 0
    planck_kT = 0

    if len(bc) >= 5:
        try:
            popt_b, _ = curve_fit(boltzmann_law, bc, hv, p0=[hv[0], np.mean(amplitudes)],
                                  maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
            res_b = hv - boltzmann_law(bc, *popt_b)
            ss_res = np.sum(res_b**2)
            ss_tot = np.sum((hv - np.mean(hv))**2)
            boltz_r2 = 1 - ss_res / (ss_tot + 1e-10)
            boltz_kT = popt_b[1]
        except Exception:
            pass

        try:
            popt_p, _ = curve_fit(planck_law, bc, hv, p0=[hv[0], np.mean(amplitudes)],
                                  maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
            res_p = hv - planck_law(bc, *popt_p)
            ss_res = np.sum(res_p**2)
            planck_r2 = 1 - ss_res / (ss_tot + 1e-10)
            planck_kT = popt_p[1]
        except Exception:
            pass

    # Inter-spike interval analysis
    spike_times = sorted([s['iteration'] for s in all_spikes])
    intervals = np.diff(spike_times) if len(spike_times) > 1 else np.array([])

    # Test for Poisson process (exponential intervals)
    poisson_ks = None
    if len(intervals) > 5:
        poisson_ks = sp_stats.kstest(intervals, 'expon', args=(0, np.mean(intervals)))

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Amplitude distribution + fits
    axes[0,0].hist(amplitudes, bins=20, density=True, color='#f39c12', alpha=0.7,
                   edgecolor='black', label='Data')
    if boltz_r2 > 0:
        x_fit = np.linspace(0.01, max(amplitudes), 100)
        axes[0,0].plot(x_fit, boltzmann_law(x_fit, *popt_b), 'r-', linewidth=2,
                       label=f'Boltzmann (R2={boltz_r2:.3f})')
    if planck_r2 > 0:
        axes[0,0].plot(x_fit, planck_law(x_fit, *popt_p), 'b--', linewidth=2,
                       label=f'Planck (R2={planck_r2:.3f})')
    axes[0,0].set_xlabel('Spike Amplitude')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('(a) Radiation Spectrum')
    axes[0,0].legend(fontsize=8)

    # (b) Spike sigma distribution
    axes[0,1].hist(sigmas, bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Spike Significance ($\\sigma$)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title(f'(b) Spike Significance (median={np.median(sigmas):.1f}$\\sigma$)')

    # (c) Inter-spike intervals
    if len(intervals) > 0:
        axes[0,2].hist(intervals, bins=15, color='#3498db', alpha=0.7, edgecolor='black',
                       density=True, label='Data')
        if poisson_ks:
            lam = 1.0 / np.mean(intervals)
            x_exp = np.linspace(0, max(intervals), 100)
            axes[0,2].plot(x_exp, lam * np.exp(-lam * x_exp), 'r-', linewidth=2,
                           label=f'Exponential (p={poisson_ks.pvalue:.3f})')
        axes[0,2].set_xlabel('Inter-spike Interval')
        axes[0,2].set_ylabel('Density')
        axes[0,2].set_title('(c) Poisson Test')
        axes[0,2].legend(fontsize=8)
    else:
        axes[0,2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0,2].transAxes)

    # (d) Example traces with spike markers
    styles = [{'c': '#c0392b', 'ls': '-'}, {'c': '#2980b9', 'ls': '--'},
              {'c': '#27ae60', 'ls': '-.'}, {'c': '#8e44ad', 'ls': ':'}]
    for i, tr in enumerate(all_traces[:4]):
        s = styles[i % 4]
        axes[1,0].plot(tr['trace'], color=s['c'], linestyle=s['ls'], alpha=0.7,
                       linewidth=1, label=f"{tr['prompt'][:15]}... ({tr['n_spikes']})")
    axes[1,0].set_xlabel('Iteration')
    axes[1,0].set_ylabel('T')
    axes[1,0].set_title('(d) T Traces')
    axes[1,0].legend(fontsize=6)

    # (e) Spike timing scatter
    for s in all_spikes:
        axes[1,1].scatter(s['iteration'], s['amplitude'], s=20,
                         c='#f39c12', edgecolors='black', alpha=0.5)
    axes[1,1].set_xlabel('Iteration')
    axes[1,1].set_ylabel('Spike Amplitude')
    axes[1,1].set_title(f'(e) Spike Timing (n={len(all_spikes)})')

    # (f) Summary
    is_thermal = boltz_r2 > 0.7 or planck_r2 > 0.7
    is_poisson = poisson_ks and poisson_ks.pvalue > 0.05 if poisson_ks else False
    best_fit = 'Planck' if planck_r2 > boltz_r2 else 'Boltzmann'
    best_r2 = max(planck_r2, boltz_r2)

    summary_text = (
        f"Hawking Radiation Analysis\n\n"
        f"Total spikes: {len(all_spikes)}\n"
        f"Boltzmann R2: {boltz_r2:.3f} (kT={boltz_kT:.3f})\n"
        f"Planck R2: {planck_r2:.3f} (kT={planck_kT:.3f})\n"
        f"Best fit: {best_fit}\n"
        f"Poisson: {'YES' if is_poisson else 'NO'}\n"
        f"\nVerdict: {'THERMAL' if is_thermal else 'NON-THERMAL'}\n"
        f"radiation"
    )
    axes[1,2].text(0.5, 0.5, summary_text, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 94: Hawking Radiation Spectrum ({best_fit} R2={best_r2:.3f}, '
                 f'n={len(all_spikes)})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase94_hawking_spectrum')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Spikes: {len(all_spikes)}")
    print(f"Boltzmann R2: {boltz_r2:.3f}, Planck R2: {planck_r2:.3f}")
    print(f"Best fit: {best_fit}")
    print(f"Poisson process: {is_poisson}")
    print(f"Verdict: {'THERMAL' if is_thermal else 'NON-THERMAL'} radiation")
    print(f"{'='*70}")

    save_results('phase94_hawking_spectrum', {
        'experiment': 'Hawking Radiation Spectrum',
        'traces': all_traces,
        'spikes': all_spikes,
        'spectrum': {
            'boltzmann_r2': float(boltz_r2),
            'boltzmann_kT': float(boltz_kT),
            'planck_r2': float(planck_r2),
            'planck_kT': float(planck_kT),
            'best_fit': best_fit,
        },
        'poisson': {
            'is_poisson': is_poisson,
            'ks_p': float(poisson_ks.pvalue) if poisson_ks else None,
        },
        'summary': {
            'n_spikes': len(all_spikes),
            'is_thermal': is_thermal,
            'best_fit': best_fit,
            'best_r2': float(best_r2),
        }
    })


if __name__ == '__main__':
    main()
