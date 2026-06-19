# -*- coding: utf-8 -*-
"""
Phase 287: Cosmological Constant x P1*T
=========================================
S-Qubit Q309 found vacuum_energy/stable_energy ratio = 0.875.
Standard Model found P1*T ~ 0.84.
Are these the same quantity? Both measure "baseline energy" ratios.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS_SHORT = [
    "The", "A", "In", "On", "At",
]

PROMPTS_LONG = [
    "The fundamental laws of physics state that energy is conserved in all closed systems",
    "Machine learning algorithms learn patterns from data by minimizing a loss function",
    "The theory of evolution explains how species change over time through natural selection",
    "Quantum mechanics describes the behavior of particles at the smallest scales of nature",
    "The structure of the universe is governed by gravity dark matter and dark energy",
]


def measure_vacuum_energy(model, tok, device):
    """Measure 'vacuum energy' = hidden state energy with minimal context (1 token)."""
    energies = []
    p1t_values = []
    for prompt in PROMPTS_SHORT:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Energy = norm of final hidden state
        h = out.hidden_states[-1][0, -1, :].float()
        energy = h.norm().item()
        energies.append(energy)

        # P1*T
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1t_values.append(p1 * t)

    return float(np.mean(energies)), float(np.mean(p1t_values))


def measure_stable_energy(model, tok, device):
    """Measure 'stable energy' = hidden state energy with full context."""
    energies = []
    p1t_values = []
    for prompt in PROMPTS_LONG:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        h = out.hidden_states[-1][0, -1, :].float()
        energy = h.norm().item()
        energies.append(energy)

        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1t_values.append(p1 * t)

    return float(np.mean(energies)), float(np.mean(p1t_values))


def measure_transition(model, tok, device):
    """Measure energy condensation as context grows."""
    base = "The fundamental laws of physics state that energy is conserved in all closed systems"
    words = base.split()
    transition_data = []

    for n_words in range(1, len(words)+1):
        prompt = " ".join(words[:n_words])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        h = out.hidden_states[-1][0, -1, :].float()
        energy = h.norm().item()

        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t = -(probs * torch.log(probs + 1e-10)).sum().item()

        transition_data.append({
            'n_words': n_words,
            'energy': round(energy, 2),
            'p1': round(p1, 4),
            't': round(t, 4),
            'p1t': round(p1 * t, 4),
        })

    return transition_data


def main():
    print("=" * 70)
    print("Phase 287: Cosmological Constant x P1*T")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        vac_energy, vac_p1t = measure_vacuum_energy(model, tok, device)
        stable_energy, stable_p1t = measure_stable_energy(model, tok, device)
        transition = measure_transition(model, tok, device)

        energy_ratio = vac_energy / (stable_energy + 1e-10)
        p1t_ratio = vac_p1t / (stable_p1t + 1e-10)

        # The cosmological constant problem:
        # vacuum energy should equal stable energy, but it doesn't
        # Similarly, P1*T at vacuum != P1*T at equilibrium
        all_results[size] = {
            'vacuum_energy': round(vac_energy, 2),
            'stable_energy': round(stable_energy, 2),
            'energy_ratio': round(energy_ratio, 4),
            'vacuum_p1t': round(vac_p1t, 4),
            'stable_p1t': round(stable_p1t, 4),
            'p1t_ratio': round(p1t_ratio, 4),
            'transition_data': transition,
            'p1t_constant_match': abs(stable_p1t - 0.84) < 0.3,
        }

        print(f"  Vacuum energy: {vac_energy:.1f}, P1T: {vac_p1t:.4f}")
        print(f"  Stable energy: {stable_energy:.1f}, P1T: {stable_p1t:.4f}")
        print(f"  Energy ratio: {energy_ratio:.4f}")
        print(f"  P1T ratio: {p1t_ratio:.4f}")
        print(f"  S-Qubit Q309 ratio was: 0.875")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Energy: vacuum vs stable
    x = np.arange(len(all_results))
    w = 0.35
    axes[0, 0].bar(x - w/2, [all_results[s]['vacuum_energy'] for s in all_results],
                  w, label='Vacuum', color='#3498db')
    axes[0, 0].bar(x + w/2, [all_results[s]['stable_energy'] for s in all_results],
                  w, label='Stable', color='#e74c3c')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(list(all_results.keys()))
    axes[0, 0].set_ylabel('Energy (L2 norm)')
    axes[0, 0].set_title('(a) Vacuum vs Stable Energy', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) P1*T: vacuum vs stable
    axes[0, 1].bar(x - w/2, [all_results[s]['vacuum_p1t'] for s in all_results],
                  w, label='Vacuum', color='#3498db')
    axes[0, 1].bar(x + w/2, [all_results[s]['stable_p1t'] for s in all_results],
                  w, label='Stable', color='#e74c3c')
    axes[0, 1].axhline(0.84, color='gold', ls='--', lw=2, label='P1T=0.84')
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(list(all_results.keys()))
    axes[0, 1].set_ylabel('P1 * T')
    axes[0, 1].set_title('(b) P1*T: Vacuum vs Stable', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Energy condensation transition
    for size, data in all_results.items():
        td = data['transition_data']
        axes[0, 2].plot([d['n_words'] for d in td], [d['energy'] for d in td],
                       '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[0, 2].set_xlabel('Number of Words')
    axes[0, 2].set_ylabel('Energy')
    axes[0, 2].set_title('(c) Energy Condensation', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) P1*T transition
    for size, data in all_results.items():
        td = data['transition_data']
        axes[1, 0].plot([d['n_words'] for d in td], [d['p1t'] for d in td],
                       '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[1, 0].axhline(0.84, color='gold', ls='--', lw=2, label='P1T=0.84')
    axes[1, 0].set_xlabel('Number of Words')
    axes[1, 0].set_ylabel('P1 * T')
    axes[1, 0].set_title('(d) P1*T Condensation', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Ratio comparison
    ratios_energy = [all_results[s]['energy_ratio'] for s in all_results]
    ratios_p1t = [all_results[s]['p1t_ratio'] for s in all_results]
    axes[1, 1].bar(x - w/2, ratios_energy, w, label='Energy ratio', color='#2ecc71')
    axes[1, 1].bar(x + w/2, ratios_p1t, w, label='P1T ratio', color='#9b59b6')
    axes[1, 1].axhline(0.875, color='orange', ls='--', label='Q309=0.875')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(list(all_results.keys()))
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('(e) Vacuum/Stable Ratios', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "COSMOLOGICAL CONSTANT x P1*T\n\n"
    txt += "S-Qubit Q309: ratio=0.875\n"
    txt += "P1*T constant: 0.84\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  E ratio: {data['energy_ratio']:.4f}\n"
        txt += f"  P1T ratio: {data['p1t_ratio']:.4f}\n"
        txt += f"  Stable P1T: {data['stable_p1t']:.4f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 287: Cosmological Constant x P1*T",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase287_cosmo_p1t')
    plt.close()

    save_results('phase287_cosmo_p1t', {
        'experiment': 'Cosmological Constant x P1T',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
