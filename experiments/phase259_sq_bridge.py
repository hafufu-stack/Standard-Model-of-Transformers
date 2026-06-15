# -*- coding: utf-8 -*-
"""
Phase 259: Quantum Phase Transition & Uncertainty Principle (SQ Bridge)
=========================================================================
SQ-Q308: Phase transition at eps_c=0.037 (perturbation threshold)
SQ-Q371: Uncertainty principle Delta_pos * Delta_sem >= hbar_T/2

This phase:
1. Replicates phase transition detection with SM thermodynamic observables
   (monitoring T_sm, P1, PR during the transition)
2. Tests the uncertainty principle using SM framework
3. Measures how the 7 SM Laws behave across the phase transition
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

PROMPT = "The capital of Japan is"
N_TRIALS = 5


def phase_transition_sm(model, tok, device, model_name):
    """Measure SM thermodynamic observables across perturbation sweep."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Baseline
    inp = tok(PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
    clean_embed = out_clean.hidden_states[0][0, -1, :].float().clone()
    clean_token = int(np.argmax(clean_logits))

    # SM thermodynamic baseline
    probs_clean = torch.softmax(torch.tensor(clean_logits), dim=-1)
    T_sm_clean = -(probs_clean * torch.log(probs_clean + 1e-10)).sum().item()
    P1_clean = float(probs_clean.max().item())

    # Perturbation sweep
    epsilons = np.logspace(-4, 2, 40)
    transition_data = []

    for eps in epsilons:
        trial_T_sm, trial_P1, trial_fid, trial_changed = [], [], [], 0
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 1000 + int(eps * 1000))
            noise = torch.randn_like(clean_embed) * eps

            def make_hook(perturbed):
                def hook(module, args, output):
                    out = output.clone()
                    out[0, -1, :] = perturbed.to(out.device).to(out.dtype)
                    return out
                return hook

            handle = model.model.embed_tokens.register_forward_hook(
                make_hook(clean_embed + noise))
            with torch.no_grad():
                out_pert = model(**inp)
            handle.remove()

            pert_logits = out_pert.logits[0, -1, :].float().cpu().numpy()
            probs_pert = torch.softmax(torch.tensor(pert_logits), dim=-1)

            # SM observables
            T_sm = -(probs_pert * torch.log(probs_pert + 1e-10)).sum().item()
            P1 = float(probs_pert.max().item())
            fid = float(np.dot(
                clean_logits / (np.linalg.norm(clean_logits) + 1e-10),
                pert_logits / (np.linalg.norm(pert_logits) + 1e-10)))

            trial_T_sm.append(T_sm if not np.isnan(T_sm) else 0)
            trial_P1.append(P1)
            trial_fid.append(fid)
            if int(np.argmax(pert_logits)) != clean_token:
                trial_changed += 1

        transition_data.append({
            'epsilon': float(eps),
            'T_sm': float(np.mean(trial_T_sm)),
            'P1': float(np.mean(trial_P1)),
            'fidelity': float(np.mean(trial_fid)),
            'change_rate': trial_changed / N_TRIALS,
        })

    # Detect critical point (fidelity drops below 0.5)
    fids = [d['fidelity'] for d in transition_data]
    eps_vals = [d['epsilon'] for d in transition_data]
    critical_idx = None
    for i in range(len(fids) - 1):
        if fids[i] > 0.5 and fids[i+1] <= 0.5:
            critical_idx = i
            break
    eps_c = eps_vals[critical_idx] if critical_idx is not None else eps_vals[-1]

    # Susceptibility = |dT_sm/deps|
    T_vals = [d['T_sm'] for d in transition_data]
    suscept = []
    for i in range(1, len(T_vals)):
        dT = abs(T_vals[i] - T_vals[i-1])
        de = abs(eps_vals[i] - eps_vals[i-1])
        suscept.append(dT / (de + 1e-15))
    max_suscept = float(np.max(suscept)) if suscept else 0

    return {
        'model': model_name,
        'eps_c': round(float(eps_c), 6),
        'max_susceptibility': round(max_suscept, 4),
        'T_sm_clean': round(T_sm_clean, 4),
        'P1_clean': round(P1_clean, 4),
        'transition_data': transition_data,
    }


def uncertainty_principle(model, tok, device, model_name):
    """Test SQ's uncertainty principle: Delta_pos * Delta_sem >= hbar_T/2."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Various prompts with different entropy levels
    prompts = [
        "The", "The capital", "The capital of", "The capital of Japan",
        "The capital of Japan is", "The capital of Japan is a city called",
        "In the beginning there was nothing but darkness and",
        "Mathematics provides the language of science because",
        "Purple elephants flew over the mountain with",
        "2 + 2 = ", "E = mc", "H2O is the chemical formula for",
    ]

    results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        n_tok = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Position uncertainty: spread of attention/info across positions
        h_last = out.hidden_states[-1][0]  # (seq, hidden)
        h_all = h_last.float().cpu().numpy()
        # Position entropy: how spread is the info across positions?
        pos_norms = np.array([np.linalg.norm(h_all[i]) for i in range(n_tok)])
        pos_probs = pos_norms / (np.sum(pos_norms) + 1e-10)
        delta_pos = float(-np.sum(pos_probs[pos_probs > 1e-15] * np.log(pos_probs[pos_probs > 1e-15])))

        # Semantic uncertainty: SM temperature (entropy of output)
        with torch.no_grad():
            normed = norm_layer(out.hidden_states[-1][:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        delta_sem = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(delta_sem):
            delta_sem = 0

        product = delta_pos * delta_sem
        results.append({
            'prompt': prompt[:30],
            'n_tokens': n_tok,
            'delta_pos': round(delta_pos, 4),
            'delta_sem': round(delta_sem, 4),
            'product': round(product, 4),
        })

    # Is there a trade-off?
    dp = [r['delta_pos'] for r in results]
    ds = [r['delta_sem'] for r in results]
    r_corr, p_corr = stats.pearsonr(dp, ds)

    # Minimum product (= hbar_T/2 analogue)
    products = [r['product'] for r in results]
    hbar_T_half = float(np.min(products))

    return {
        'model': model_name,
        'results': results,
        'correlation': {'r': round(float(r_corr), 4), 'p': round(float(p_corr), 4)},
        'hbar_T_half': round(hbar_T_half, 4),
        'mean_product': round(float(np.mean(products)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 259: Phase Transition + Uncertainty Principle (SQ Bridge)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)

        pt = phase_transition_sm(model, tok, device, size)
        up = uncertainty_principle(model, tok, device, size)

        all_results[size] = {'phase_transition': pt, 'uncertainty': up}
        print(f"  Phase transition: eps_c={pt['eps_c']:.6f}, max chi={pt['max_susceptibility']:.4f}")
        print(f"  Uncertainty: r(dp,ds)={up['correlation']['r']:.3f}, hbar_T/2={up['hbar_T_half']:.3f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        pt = data['phase_transition']
        up = data['uncertainty']
        c = colors[size]
        eps = [d['epsilon'] for d in pt['transition_data']]

        # (a) Fidelity curve
        fids = [d['fidelity'] for d in pt['transition_data']]
        axes[0, 0].semilogx(eps, fids, 'o-', color=c, lw=2, markersize=3,
                           label=f"{size} (eps_c={pt['eps_c']:.4f})")
        axes[0, 0].axvline(pt['eps_c'], color=c, ls='--', alpha=0.5)

        # (b) T_sm across transition
        T_vals = [d['T_sm'] for d in pt['transition_data']]
        axes[0, 1].semilogx(eps, T_vals, 'o-', color=c, lw=2, markersize=3, label=size)

        # (c) P1 across transition
        P1_vals = [d['P1'] for d in pt['transition_data']]
        axes[0, 2].semilogx(eps, P1_vals, 'o-', color=c, lw=2, markersize=3, label=size)

        # (d) Uncertainty: Delta_pos vs Delta_sem
        dp = [r['delta_pos'] for r in up['results']]
        ds = [r['delta_sem'] for r in up['results']]
        axes[1, 0].scatter(dp, ds, c=c, s=40, alpha=0.7,
                          label=f"{size} (r={up['correlation']['r']:.3f})")

        # (e) Product histogram
        products = [r['product'] for r in up['results']]
        axes[1, 1].hist(products, bins=8, alpha=0.5, color=c, edgecolor='black',
                       label=f"{size} (hbar/2={up['hbar_T_half']:.2f})")

    axes[0, 0].set_xlabel('Perturbation (eps)'); axes[0, 0].set_ylabel('Fidelity')
    axes[0, 0].set_title('(a) Phase Transition: Fidelity')
    axes[0, 0].axhline(0.5, color='gray', ls=':', lw=1)
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_xlabel('Perturbation (eps)'); axes[0, 1].set_ylabel('T_sm')
    axes[0, 1].set_title('(b) SM Temperature Across Transition')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].set_xlabel('Perturbation (eps)'); axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) Max Probability Across Transition')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].set_xlabel('Delta_pos'); axes[1, 0].set_ylabel('Delta_sem (T_sm)')
    axes[1, 0].set_title('(d) Uncertainty Principle')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_xlabel('Delta_pos x Delta_sem'); axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('(e) Uncertainty Product Distribution')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "SQ BRIDGE RESULTS\n\n"
    for size, data in all_results.items():
        pt = data['phase_transition']
        up = data['uncertainty']
        summary += f"{size}:\n"
        summary += f"  eps_c = {pt['eps_c']:.4f}\n"
        summary += f"  chi_max = {pt['max_susceptibility']:.2f}\n"
        summary += f"  r(dp,ds) = {up['correlation']['r']:.3f}\n"
        summary += f"  hbar_T/2 = {up['hbar_T_half']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 259: Phase Transition + Uncertainty Principle (SQ Bridge)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase259_sq_bridge')
    plt.close()
    save_results('phase259_sq_bridge', {
        'experiment': 'Phase Transition + Uncertainty (SQ Bridge)',
        'results': all_results,
    })


if __name__ == '__main__':
    main()
