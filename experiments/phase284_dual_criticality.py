# -*- coding: utf-8 -*-
"""
Phase 284: Quantum Phase Transition x Critical Opalescence
============================================================
S-Qubit Q308 found quantum phase transition at eps_c=0.037 with beta=-0.45.
Standard Model P276 found critical opalescence at L0=19-20.
Are these the same critical phenomenon? Test on the same model.
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

PROMPT = "The fundamental laws of physics state that energy cannot be created or destroyed"


def measure_opalescence(model, tok, prompt, device):
    """Measure variance and susceptibility across layers (P276 method)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    norms = []
    for hs in out.hidden_states:
        h = hs[0, -1, :].float()
        norms.append(h.norm().item())

    # Variance per layer (sliding window)
    window = 3
    variances = []
    for i in range(len(norms) - window + 1):
        variances.append(float(np.var(norms[i:i+window])))

    # Susceptibility = d(variance)/d(layer)
    suscept = [abs(variances[i+1] - variances[i]) for i in range(len(variances)-1)]

    L0_var = int(np.argmax(variances))
    L0_chi = int(np.argmax(suscept)) if suscept else 0

    return {
        'norms': norms,
        'variances': variances,
        'susceptibilities': suscept,
        'L0_variance': L0_var,
        'L0_susceptibility': L0_chi,
    }


def measure_quantum_transition(model, tok, prompt, device, n_epsilons=25):
    """Measure output fidelity under increasing noise (Q308 method).
    Uses embedding-level noise injection to avoid hook compatibility issues.
    """
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    clean_logits = out_clean.logits[0, -1, :].float()
    clean_hidden = out_clean.hidden_states[-1][0, -1, :].float()

    epsilons = np.logspace(-3, 0, n_epsilons)
    transition_data = []

    for eps in epsilons:
        fidelities = []
        for _ in range(3):
            # Inject noise at embedding level via hook on embed_tokens
            def make_embed_hook(sigma):
                def hook_fn(module, input, output):
                    # output is (batch, seq, hidden) from embedding layer
                    noise = torch.randn_like(output) * sigma
                    return output + noise
                return hook_fn

            handle = model.model.embed_tokens.register_forward_hook(make_embed_hook(eps))

            with torch.no_grad():
                out_noisy = model(**inp, output_hidden_states=True)

            handle.remove()

            noisy_hidden = out_noisy.hidden_states[-1][0, -1, :].float()

            # Fidelity metrics
            fid_cos = torch.nn.functional.cosine_similarity(
                clean_hidden.unsqueeze(0), noisy_hidden.unsqueeze(0)).item()
            fidelities.append(fid_cos)

        transition_data.append({
            'epsilon': round(float(eps), 6),
            'fidelity': round(float(np.mean(fidelities)), 4),
            'fidelity_std': round(float(np.std(fidelities)), 4),
        })

    # Find critical point (steepest drop)
    fids = [d['fidelity'] for d in transition_data]
    dfids = [fids[i] - fids[i+1] for i in range(len(fids)-1)]
    crit_idx = int(np.argmax(dfids))
    eps_c = epsilons[crit_idx]

    # Critical exponent from log-log fit around transition
    half = len(fids)
    log_eps = np.log(epsilons[:half])
    log_fid = np.log(np.array(fids[:half]) + 1e-10)
    slope, _, _, _, _ = stats.linregress(log_eps, log_fid)

    return {
        'transition_data': transition_data,
        'eps_c': round(float(eps_c), 6),
        'critical_exponent': round(float(slope), 4),
        'max_susceptibility': round(float(max(dfids)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 284: Quantum Phase Transition x Critical Opalescence")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        n_layers = len(model.model.layers)

        # Measure critical opalescence
        opal = measure_opalescence(model, tok, PROMPT, device)
        print(f"  Opalescence: L0_var={opal['L0_variance']}, "
              f"L0_chi={opal['L0_susceptibility']}")

        # Measure quantum phase transition
        qpt = measure_quantum_transition(model, tok, PROMPT, device)
        print(f"  Quantum PT: eps_c={qpt['eps_c']:.4f}, "
              f"beta={qpt['critical_exponent']:.3f}")

        # Normalized critical layers
        opal_L0_norm = opal['L0_variance'] / n_layers
        qpt_eps_norm = qpt['eps_c']

        all_results[size] = {
            'n_layers': n_layers,
            'opalescence': opal,
            'quantum_transition': qpt,
            'opal_L0_normalized': round(opal_L0_norm, 4),
            'same_phenomenon': abs(opal_L0_norm - 0.7) < 0.2,  # both near 70%
        }

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Norm profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['opalescence']['norms'], '-', color=colors[size],
                       lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Hidden State Norm')
    axes[0, 0].set_title('(a) Norm Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Variance (opalescence)
    for size, data in all_results.items():
        axes[0, 1].plot(data['opalescence']['variances'], '-', color=colors[size],
                       lw=2, label=size)
        L0 = data['opalescence']['L0_variance']
        axes[0, 1].axvline(L0, color=colors[size], ls='--', alpha=0.5)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].set_title('(b) Critical Opalescence', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Quantum phase transition
    for size, data in all_results.items():
        td = data['quantum_transition']['transition_data']
        axes[0, 2].plot([d['epsilon'] for d in td], [d['fidelity'] for d in td],
                       '-o', color=colors[size], lw=2, markersize=4, label=size)
        axes[0, 2].axvline(data['quantum_transition']['eps_c'],
                          color=colors[size], ls='--', alpha=0.5)
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_xlabel('Noise epsilon')
    axes[0, 2].set_ylabel('Fidelity')
    axes[0, 2].set_title('(c) Quantum Phase Transition', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Susceptibility comparison
    for size, data in all_results.items():
        axes[1, 0].plot(data['opalescence']['susceptibilities'], '-',
                       color=colors[size], lw=2, label=f'{size} opal')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Susceptibility')
    axes[1, 0].set_title('(d) Susceptibility', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Critical point comparison
    opal_l0s = [data['opal_L0_normalized'] for data in all_results.values()]
    eps_cs = [data['quantum_transition']['eps_c'] for data in all_results.values()]
    axes[1, 1].scatter(opal_l0s, eps_cs, c=[colors[s] for s in all_results.keys()], s=100)
    for i, s in enumerate(all_results.keys()):
        axes[1, 1].annotate(s, (opal_l0s[i], eps_cs[i]),
                           textcoords="offset points", xytext=(10, 5))
    axes[1, 1].set_xlabel('Opalescence L0 (normalized)')
    axes[1, 1].set_ylabel('Quantum eps_c')
    axes[1, 1].set_title('(e) Critical Points', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "DUAL CRITICALITY TEST\n\n"
    txt += "Q308 Quantum Phase Transition\n"
    txt += "  x  P276 Critical Opalescence\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  Opal L0 = {data['opalescence']['L0_variance']}/{data['n_layers']}\n"
        txt += f"  QPT eps_c = {data['quantum_transition']['eps_c']:.4f}\n"
        txt += f"  QPT beta = {data['quantum_transition']['critical_exponent']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 284: Quantum Phase Transition x Critical Opalescence",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase284_dual_criticality')
    plt.close()

    save_results('phase284_dual_criticality', {
        'experiment': 'Quantum Phase Transition x Critical Opalescence',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
