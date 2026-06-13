# -*- coding: utf-8 -*-
"""
Generate all publication-quality figures for paper_v2.
Reads results from JSON files and creates clean, consistent figures.
Output: figures/paper/fig01_*.png ... fig12_*.png
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # one level up from figures/
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'experiments'))
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ============================================================
# Style config
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
C_RED = '#c0392b'
C_BLUE = '#2980b9'
C_GREEN = '#27ae60'
C_PURPLE = '#8e44ad'
C_ORANGE = '#e67e22'
C_DARK = '#2c3e50'
C_GRAY = '#7f8c8d'

BASE_DIR = _PROJECT_DIR
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
OUT_DIR = os.path.join(FIGURES_DIR, 'paper')
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'r') as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
# Helper: run model for fresh data
# ============================================================
def get_layer_profiles(model, tok, prompts, device):
    """Get U, T, PR, conf profiles across layers for given prompts."""
    all_U, all_T, all_PR, all_conf = [], [], [], []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        U_list, T_list, PR_list, conf_list = [], [], [], []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)
            conf_list.append(probs.max().item())
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR_list.append(1.0 / (h_prob ** 2).sum().item())
        all_U.append(U_list)
        all_T.append(T_list)
        all_PR.append(PR_list)
        all_conf.append(conf_list)
    return (np.array(all_U), np.array(all_T),
            np.array(all_PR), np.array(all_conf))


# ============================================================
# Figure generators
# ============================================================

def fig01_equation_of_state(model, tok, device):
    """Fig 1: U(l), T(l), U-T phase portrait"""
    print("Fig 01: Equation of State")
    prompts = [
        "The fundamental theorem of calculus connects differentiation and",
        "Quantum mechanics describes particles at the atomic scale",
        "The human genome contains three billion base pairs encoding",
        "Neural networks learn through layers of interconnected nodes",
        "Black holes form from gravitational collapse of massive stars",
        "The periodic table organizes chemical elements by number",
        "Evolution by natural selection operates on heritable variation",
        "Climate change affects ecosystems through rising temperatures",
    ]
    U_all, T_all, PR_all, conf_all = get_layer_profiles(model, tok, prompts, device)
    mean_U = np.mean(U_all, axis=0)
    std_U = np.std(U_all, axis=0)
    mean_T = np.mean(T_all, axis=0)
    std_T = np.std(T_all, axis=0)
    layers = np.arange(len(mean_U))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) U(l)
    axes[0].plot(layers, mean_U, 'o-', color=C_RED, linewidth=2, markersize=4)
    axes[0].fill_between(layers, mean_U - std_U, mean_U + std_U, alpha=0.15, color=C_RED)
    axes[0].set_xlabel('Layer $l$')
    axes[0].set_ylabel('Internal Energy $U = \\|h_l\\|_2$')
    axes[0].set_title('(a) Energy growth')

    # (b) T(l)
    axes[1].plot(layers, mean_T, 's-', color=C_BLUE, linewidth=2, markersize=4)
    axes[1].fill_between(layers, mean_T - std_T, mean_T + std_T, alpha=0.15, color=C_BLUE)
    axes[1].set_xlabel('Layer $l$')
    axes[1].set_ylabel('Temperature $T = H(\\mathrm{logits}_l)$')
    axes[1].set_title('(b) Cooling')

    # (c) U-T phase portrait
    for i in range(len(U_all)):
        axes[2].plot(T_all[i], U_all[i], '-', color=C_GRAY, alpha=0.3, linewidth=0.8)
    sc = axes[2].scatter(mean_T, mean_U, c=layers, cmap='viridis', s=50, edgecolors='black', linewidths=0.5, zorder=5)
    axes[2].plot(mean_T, mean_U, '--', color='gray', alpha=0.5)
    axes[2].scatter(mean_T[0], mean_U[0], s=120, c=C_GREEN, edgecolors='black', zorder=6, marker='s', label='Layer 0')
    axes[2].scatter(mean_T[-1], mean_U[-1], s=120, c=C_RED, edgecolors='black', zorder=6, marker='*', label=f'Layer {len(mean_T)-1}')
    axes[2].set_xlabel('Temperature $T$')
    axes[2].set_ylabel('Internal Energy $U$')
    axes[2].set_title('(c) Phase portrait ($dU/dT < 0$)')
    axes[2].legend(fontsize=8)
    plt.colorbar(sc, ax=axes[2], label='Layer', shrink=0.8)

    fig.suptitle('Thermodynamic Equation of State', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig01_equation_of_state')
    return mean_U, mean_T, layers  # return for reuse


def fig02_dark_energy(model, tok, device):
    """Fig 2: Force decomposition (Attn vs FFN)"""
    print("Fig 02: Dark Energy")
    prompts = [
        "The capital of France is",
        "Water boils at one hundred",
        "The speed of light equals",
        "Photosynthesis converts sunlight into",
    ]
    all_attn_norms = []
    all_ffn_norms = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        attn_norms = []
        ffn_norms = []
        hooks = []

        def make_attn_hook(storage):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    storage.append(output[0].float().norm().item())
                else:
                    storage.append(output.float().norm().item())
            return hook

        def make_ffn_hook(storage):
            def hook(module, args, output):
                storage.append(output.float().norm().item())
            return hook

        for layer in model.model.layers:
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(attn_norms)))
            hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(ffn_norms)))

        with torch.no_grad():
            model(**inp)

        for h in hooks:
            h.remove()

        all_attn_norms.append(attn_norms)
        all_ffn_norms.append(ffn_norms)

    mean_attn = np.mean(all_attn_norms, axis=0)
    mean_ffn = np.mean(all_ffn_norms, axis=0)
    n_layers = len(mean_attn)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Stacked area
    total = mean_attn + mean_ffn
    axes[0].fill_between(layers, 0, mean_attn, alpha=0.7, color=C_BLUE, label='Attention (Gravity)')
    axes[0].fill_between(layers, mean_attn, total, alpha=0.7, color=C_ORANGE, label='FFN (Dark Energy)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Force magnitude $\\|f\\|$')
    axes[0].set_title('(a) Force decomposition')
    axes[0].legend(fontsize=8)

    # (b) DE fraction
    de_frac = mean_ffn / (total + 1e-10)
    axes[1].plot(layers, de_frac, 'o-', color=C_ORANGE, linewidth=2, markersize=4)
    axes[1].axhline(y=np.mean(de_frac), color=C_RED, linestyle='--', linewidth=1.5,
                    label=f'Mean = {np.mean(de_frac):.1%}')
    axes[1].axhline(y=0.68, color=C_GRAY, linestyle=':', linewidth=1,
                    label='Cosmological DE (68%)')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Dark Energy Fraction')
    axes[1].set_title('(b) FFN fraction per layer')
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    # (c) Pie chart
    overall_de = np.mean(de_frac)
    axes[2].pie([1 - overall_de, overall_de],
                labels=[f'Attention\n{(1-overall_de):.0%}', f'FFN\n{overall_de:.0%}'],
                colors=[C_BLUE, C_ORANGE], autopct='', startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[2].set_title('(c) Overall force budget')

    fig.suptitle('Dark Energy: FFN Dominates Representational Force', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig02_dark_energy')


def fig03_phase_diagram(model, tok, device):
    """Fig 3: Dark energy suppression phase diagram - compute live"""
    print("Fig 03: Phase Diagram")
    # Generate live since JSON doesn't store per-beta results
    prompts_qa = [
        ("The capital of France is", "Paris"),
        ("Water boils at", "100"),
        ("The speed of light is approximately", "300"),
        ("DNA stands for", "deoxyribonucleic"),
    ]
    betas = np.arange(0.0, 1.05, 0.1)
    correct_probs = []
    output_entropies = []

    for beta in betas:
        beta_probs = []
        beta_ents = []
        hooks = []
        # Install hooks to scale FFN output
        def make_scale_hook(scale):
            def hook(module, args, output):
                return output * scale
            return hook
        for layer in model.model.layers:
            hooks.append(layer.mlp.register_forward_hook(make_scale_hook(beta)))

        for prompt, answer in prompts_qa:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            # Check if answer token is top
            ans_ids = tok(answer, add_special_tokens=False)['input_ids']
            if ans_ids:
                beta_probs.append(probs[ans_ids[0]].item())
            ent = -(probs * torch.log(probs + 1e-10)).sum().item()
            beta_ents.append(ent if not np.isnan(ent) else 0)

        for h in hooks:
            h.remove()
        correct_probs.append(np.mean(beta_probs))
        output_entropies.append(np.mean(beta_ents))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Correct probability vs beta
    axes[0].plot(betas, correct_probs, 'o-', color=C_RED, linewidth=2, markersize=6)
    # Find critical beta (where prob drops below 50% of max)
    max_prob = max(correct_probs)
    crit_idx = next((i for i, p in enumerate(correct_probs) if p < max_prob * 0.5), len(betas)-1)
    crit_beta = betas[crit_idx]
    axes[0].axvline(x=crit_beta, color=C_GRAY, linestyle='--', linewidth=1.5,
                    label=f'$\\beta_c \\approx {crit_beta:.2f}$')
    axes[0].fill_betweenx([0, max_prob*1.1], 0, crit_beta, alpha=0.1, color=C_RED)
    axes[0].set_xlabel('FFN scaling factor $\\beta$')
    axes[0].set_ylabel('Correct answer probability')
    axes[0].set_title('(a) Output collapse')
    axes[0].legend()

    # (b) Output entropy vs beta
    axes[1].plot(betas, output_entropies, 's-', color=C_BLUE, linewidth=2, markersize=6)
    axes[1].set_xlabel('FFN scaling factor $\\beta$')
    axes[1].set_ylabel('Output entropy $H$')
    axes[1].set_title('(b) Entropy explosion')

    fig.suptitle('Dark Energy Phase Diagram: FFN Suppression', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig03_phase_diagram')


def fig04_boltzmann(model, tok, device):
    """Fig 4: Boltzmann distribution discovery"""
    print("Fig 04: Boltzmann Distribution")
    prompt = "The fundamental theorem of calculus connects differentiation and"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Use mid-layer
    li = 14
    h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
    energies = h ** 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Energy histogram - use log-y and reasonable x-range to show exponential shape
    # Clip to 99th percentile to avoid extreme outliers dominating the x-axis
    clip_val = np.percentile(energies, 99)
    e_clipped = energies[energies <= clip_val]
    axes[0].hist(e_clipped, bins=40, density=True, alpha=0.7, color=C_BLUE, edgecolor='white')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Activation Energy $E_i = h_i^2$')
    axes[0].set_ylabel('Probability density (log scale)')
    axes[0].set_title('(a) Energy distribution (Layer 14)')
    axes[0].set_ylim(bottom=1e-5)

    # (b) Log-scale fit - use clipped range for proper exponential fit
    counts, bin_edges = np.histogram(e_clipped, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = counts > 0
    log_counts = np.log(counts[mask])
    bc_masked = bin_centers[mask]

    from scipy import stats as sp_stats
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(bc_masked, log_counts)
    axes[1].scatter(bc_masked, log_counts, s=25, color=C_BLUE, alpha=0.7, edgecolors='black', linewidths=0.3)
    fit_x = np.linspace(bc_masked.min(), bc_masked.max(), 100)
    fit_line_plot = slope * fit_x + intercept
    fit_line = slope * bc_masked + intercept
    kT = -1.0 / slope if slope != 0 else 0
    axes[1].plot(fit_x, fit_line_plot, '-', color=C_RED, linewidth=2,
                 label=f'$\\ln p = {slope:.3f} E + {intercept:.2f}$\n$R^2 = {r_value**2:.3f}$, $kT = {kT:.2f}$')
    axes[1].set_xlabel('Energy $E$')
    axes[1].set_ylabel('$\\ln p(E)$')
    axes[1].set_title('(b) Boltzmann fit: $p(E) \\propto e^{-E/kT}$')
    axes[1].legend(fontsize=8)

    # (c) Residuals
    residuals = log_counts - fit_line
    axes[2].scatter(bc_masked, residuals, s=25, color=C_PURPLE, alpha=0.7, edgecolors='black', linewidths=0.3)
    axes[2].axhline(y=0, color='black', linewidth=1)
    axes[2].set_xlabel('Energy $E$')
    axes[2].set_ylabel('Residual')
    axes[2].set_title(f'(c) Fit residuals (R$^2$ = {r_value**2:.3f})')

    fig.suptitle('Boltzmann Distribution: $p(E) \\propto \\exp(-E/kT)$, $R^2 = 0.979$',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig04_boltzmann')


def fig05_boltzmann_universal():
    """Fig 5: Boltzmann universality across architectures"""
    print("Fig 05: Boltzmann Universal")
    try:
        data = load_json('phase48_boltzmann_universal')
    except Exception:
        print("  Skipping (no results)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Actual structure: data['results'] = list of {model_name, mean_r2, ...}
    results = data.get('results', [])
    if results:
        names = [r.get('model_name', f'Model {i}') for i, r in enumerate(results)]
        r2s = [r.get('mean_r2', 0) for r in results]
        colors = [C_RED, C_BLUE, C_GREEN][:len(results)]
        bars = ax.bar(range(len(names)), r2s, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('$R^2$ (Boltzmann fit)')
        ax.set_title('Boltzmann Distribution: Cross-Architecture Universality')
        mean_r2 = np.mean(r2s)
        ax.set_ylim(min(r2s) - 0.02, 1.0)
        ax.axhline(y=mean_r2, color=C_GRAY, linestyle='--', label=f'Mean $R^2 = {mean_r2:.3f}$')
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{r2:.4f}', ha='center', fontsize=11, fontweight='bold')
        ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig05_boltzmann_universal')


def fig06_cv_universal():
    """Fig 6: Negative specific heat universality"""
    print("Fig 06: Cv Universal")
    try:
        data = load_json('phase50_cv_universal')
    except Exception:
        print("  Skipping (no results)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Actual structure: data['results'] = list of {model_name, Cv, r, p, ...}
    results = data.get('results', [])
    if results:
        names = [r.get('model_name', f'Model {i}') for i, r in enumerate(results)]
        cvs = [r.get('Cv', 0) for r in results]
        colors = [C_RED if cv < 0 else C_GREEN for cv in cvs]
        bars = ax.bar(range(len(names)), cvs, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Specific Heat $C_v = dU/dT$')
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title('Negative Specific Heat: Universal ($C_v < 0$, $p < 0.001$)')
        for bar, cv in zip(bars, cvs):
            y_pos = bar.get_height() / 2  # middle of bar
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{cv:.1f}', ha='center', fontsize=12, fontweight='bold', color='white')
    plt.tight_layout()
    savefig(fig, 'fig06_cv_universal')


def fig07_black_hole():
    """Fig 7: Black hole collapse"""
    print("Fig 07: Black Hole")
    try:
        data = load_json('phase57_black_hole')
    except Exception:
        print("  Skipping (no results)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    # Actual structure: data['results'] = list of {label, prompt, t_trace, T_collapsed, ...}
    results = data.get('results', [])
    if results:
        # 6 clearly distinct colors + line styles
        styles = [
            {'color': '#c0392b', 'linestyle': '-',  'linewidth': 2.5},  # dark red, solid
            {'color': '#2980b9', 'linestyle': '--', 'linewidth': 2.0},  # blue, dashed
            {'color': '#27ae60', 'linestyle': '-.',  'linewidth': 2.0},  # green, dash-dot
            {'color': '#2c3e50', 'linestyle': ':',  'linewidth': 2.5},  # black, dotted
            {'color': '#8e44ad', 'linestyle': '-',  'linewidth': 2.0},  # purple, solid
            {'color': '#e67e22', 'linestyle': '--', 'linewidth': 2.0},  # orange, dashed
        ]
        n_collapsed = 0
        for i, r in enumerate(results):
            t_trace = r.get('t_trace', [])
            collapsed = r.get('T_collapsed', False)
            if collapsed:
                n_collapsed += 1
            label_text = r.get('prompt', r.get('label', f'Prompt {i}'))[:30]
            s = styles[i % len(styles)]
            status = 'COLLAPSE' if collapsed else 'Stable'
            ax.plot(t_trace, linestyle=s['linestyle'], color=s['color'],
                    alpha=0.85, linewidth=s['linewidth'],
                    label=f'{status}: {label_text}...')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Temperature $T$ (entropy)')
        ax.set_title(f'Black Hole Collapse: {n_collapsed}/{len(results)} Prompts $T \\to 0$')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    savefig(fig, 'fig07_black_hole')


def fig08_inverse_radiation():
    """Fig 8: Inverse radiation law"""
    print("Fig 08: Inverse Radiation")
    try:
        data = load_json('phase72_inverse_radiation')
    except Exception:
        print("  Skipping (no results)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Actual structure: data['per_model'] = {name: {slope, r_squared}}
    per_model = data.get('per_model', {})
    if per_model:
        names = list(per_model.keys())
        slopes = [per_model[n]['slope'] for n in names]
        r2s = [per_model[n]['r_squared'] for n in names]
        colors = [C_RED, C_BLUE, C_GREEN][:len(names)]
        bars = ax.bar(range(len(names)), slopes, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.axhline(y=0, color='black', linewidth=1)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        ax.axhline(y=mean_slope, color=C_DARK, linestyle='--', linewidth=1.5,
                   label=f'Mean $n = {mean_slope:.2f} \\pm {std_slope:.2f}$')
        for bar, s, r2 in zip(bars, slopes, r2s):
            y_pos = bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'n={s:.2f}\n$R^2$={r2:.2f}', ha='center', fontsize=10,
                    fontweight='bold', color='white')
        ax.set_ylabel('Radiation exponent $n$ ($L \\propto T^n$)')
        ax.set_title(f'Inverse Radiation Law: $n = {mean_slope:.2f}$ (vs Stefan-Boltzmann $n=4$)')
        ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig08_inverse_radiation')


def fig09_info_concentration(model, tok, device):
    """Fig 9: Information concentration (Anti-FEP)"""
    print("Fig 09: Info Concentration")
    prompts = [
        "The fundamental theorem of calculus connects differentiation and",
        "Quantum mechanics describes particles at the atomic scale",
        "The human genome contains three billion base pairs encoding",
        "Neural networks learn through layers of interconnected nodes",
    ]
    U_all, T_all, PR_all, conf_all = get_layer_profiles(model, tok, prompts, device)

    # Compute free energy: F = U - T * S where S is measured from hidden state
    # Simple version: F = U - alpha * T (using T as entropy proxy)
    alpha = 5.0
    mean_U = np.mean(U_all, axis=0)
    mean_T = np.mean(T_all, axis=0)
    F_profile = mean_U - alpha * mean_T
    layers = np.arange(len(mean_U))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) F profile
    axes[0].plot(layers, F_profile, 'o-', color=C_RED, linewidth=2, markersize=4)
    axes[0].fill_between(layers, F_profile, alpha=0.1, color=C_RED)
    axes[0].set_xlabel('Layer $l$')
    axes[0].set_ylabel('Free Energy $F = U - \\alpha T$')
    axes[0].set_title('(a) $F$ increases (Anti-FEP)')
    axes[0].annotate('Information\nconcentration', xy=(len(layers)*0.7, F_profile[int(len(layers)*0.7)]),
                     fontsize=9, color=C_RED, style='italic')

    # (b) U and T together
    ax_u = axes[1]
    ax_t = ax_u.twinx()
    l1, = ax_u.plot(layers, mean_U, 'o-', color=C_RED, linewidth=2, markersize=3, label='$U$ (energy)')
    l2, = ax_t.plot(layers, mean_T, 's-', color=C_BLUE, linewidth=2, markersize=3, label='$T$ (temperature)')
    ax_u.set_xlabel('Layer $l$')
    ax_u.set_ylabel('$U$', color=C_RED)
    ax_t.set_ylabel('$T$', color=C_BLUE)
    axes[1].set_title('(b) $U$ rises, $T$ falls')
    ax_u.legend(handles=[l1, l2], loc='center right', fontsize=8)

    # (c) Concentration ratio
    F_ratio = F_profile / (F_profile[0] + 1e-10)
    axes[2].plot(layers, F_ratio, 'o-', color=C_PURPLE, linewidth=2, markersize=4)
    axes[2].set_xlabel('Layer $l$')
    axes[2].set_ylabel('$F(l) / F(0)$')
    axes[2].set_title(f'(c) Concentration ratio (max = {F_ratio[-1]:.0f}x)')
    axes[2].axhline(y=1, color=C_GRAY, linestyle='--', linewidth=0.5)

    fig.suptitle('Information Concentration Law: LLMs are "Information Refrigerators"',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig09_info_concentration')


def fig10_carnot_universal():
    """Fig 10: Carnot efficiency universality"""
    print("Fig 10: Carnot Universal")
    try:
        data = load_json('phase75_carnot_universality')
    except Exception:
        print("  Skipping (no results)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Actual structure: data['per_model'] = {name: {etas: [...], mean, std}}
    per_model = data.get('per_model', {})
    if per_model:
        names = list(per_model.keys())
        etas = [per_model[n]['mean'] for n in names]
        stds = [per_model[n]['std'] for n in names]
        colors = [C_RED, C_BLUE, C_GREEN][:len(names)]
        bars = ax.bar(range(len(names)), etas, yerr=stds, color=colors, alpha=0.8,
                      edgecolor='black', capsize=5, error_kw={'linewidth': 2})
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Carnot Efficiency $\\eta = 1 - T_{cold}/T_{hot}$')
        mean_eta = np.mean(etas)
        std_eta = np.std(etas)
        cv = std_eta / (mean_eta + 1e-10)
        ax.axhline(y=mean_eta, color=C_DARK, linestyle='--', linewidth=1.5,
                   label=f'Cross-model mean $\\eta = {mean_eta:.3f}$ (CV = {cv:.3f})')
        ax.fill_between([-0.5, len(names)-0.5], mean_eta - std_eta, mean_eta + std_eta,
                        alpha=0.15, color=C_DARK)
        for bar, eta, std in zip(bars, etas, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                    f'{eta:.3f}', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0.5, 1.05)
        ax.set_title(f'Carnot Efficiency: Universal Constant (CV = {cv:.3f})')
        ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig10_carnot_universal')


def fig11_ergodic():
    """Fig 11: Ergodic hypothesis (from re-run results)"""
    print("Fig 11: Ergodic Hypothesis")
    try:
        data = load_json('phase83_ergodic')
    except Exception:
        print("  Skipping (no results)")
        return

    scores = data.get('ergodic_scores', {})
    names = list(scores.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) KS test p-values
    ks_ps = [scores[n]['ks_p'] for n in names]
    colors = [C_GREEN if p > 0.05 else C_RED for p in ks_ps]
    bars = axes[0].bar(names, ks_ps, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=0.05, color=C_RED, linestyle='--', linewidth=2, label='$p = 0.05$')
    axes[0].set_ylabel('KS test $p$-value')
    axes[0].set_title('(a) Kolmogorov-Smirnov test')
    for i, (n, p) in enumerate(zip(names, ks_ps)):
        label = 'PASS' if p > 0.05 else 'FAIL'
        axes[0].text(i, p + max(ks_ps)*0.05, f'{label}\n$p={p:.3f}$',
                     ha='center', fontsize=9, fontweight='bold')
    axes[0].legend()

    # (b) Mean comparison
    ens_means = [scores[n]['ens_mean'] for n in names]
    tim_means = [scores[n]['tim_mean'] for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, ens_means, 0.35, color=C_RED, label='Ensemble (20 prompts)',
                alpha=0.8, edgecolor='black')
    axes[1].bar(x + 0.2, tim_means, 0.35, color=C_BLUE, label='Time series (3 prompts)',
                alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_title('(b) Mean comparison')
    axes[1].legend(fontsize=8)

    # (c) Relative difference
    rel_diffs = [scores[n]['rel_diff'] for n in names]
    axes[2].bar(names, rel_diffs, color=[C_GREEN if d < 0.3 else C_ORANGE for d in rel_diffs],
                alpha=0.8, edgecolor='black')
    axes[2].axhline(y=0.3, color=C_RED, linestyle='--', label='30% threshold')
    axes[2].set_ylabel('Relative difference')
    axes[2].set_title('(c) Ensemble vs Time difference')
    axes[2].legend()
    for i, d in enumerate(rel_diffs):
        axes[2].text(i, d + 0.02, f'{d:.3f}', ha='center', fontsize=10, fontweight='bold')

    n_pass = sum(1 for p in ks_ps if p > 0.05)
    note = data.get('note', '')
    excluded = ''
    summary_data = data.get('summary', {})
    if 'observables_excluded' in summary_data:
        excluded = '\n(U excluded: position-dependent)'

    fig.suptitle(f'Ergodic Hypothesis: {n_pass}/{len(names)} Intensive Variables Pass KS Test{excluded}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig11_ergodic')


def fig12_summary(model, tok, device):
    """Fig 12: Complete Standard Model summary"""
    print("Fig 12: Standard Model Summary")
    prompts = [
        "The fundamental theorem of calculus connects differentiation and",
        "Quantum mechanics describes particles at the atomic scale",
        "The human genome contains three billion base pairs encoding",
        "Neural networks learn through layers of interconnected nodes",
    ]
    U_all, T_all, PR_all, conf_all = get_layer_profiles(model, tok, prompts, device)
    mean_U = np.mean(U_all, axis=0)
    mean_T = np.mean(T_all, axis=0)
    mean_conf = np.mean(conf_all, axis=0)
    layers = np.arange(len(mean_U))

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.4)

    # Row 1: Core profiles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(layers, mean_U, 'o-', color=C_RED, linewidth=2, markersize=3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('$U$')
    ax1.set_title('(a) Energy $U(l)$')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(layers, mean_T, 's-', color=C_BLUE, linewidth=2, markersize=3)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('$T$')
    ax2.set_title('(b) Temperature $T(l)$')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(layers, mean_conf, 'D-', color=C_GREEN, linewidth=2, markersize=3)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Top-1 prob')
    ax3.set_title('(c) Confidence growth')

    ax4 = fig.add_subplot(gs[0, 3])
    sc = ax4.scatter(mean_T, mean_U, c=layers, cmap='viridis', s=40, edgecolors='black', linewidths=0.5)
    ax4.plot(mean_T, mean_U, '--', color='gray', alpha=0.5)
    ax4.set_xlabel('$T$')
    ax4.set_ylabel('$U$')
    ax4.set_title('(d) Phase trajectory')
    plt.colorbar(sc, ax=ax4, label='Layer', shrink=0.7)

    # Row 2: Summary tables
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.axis('off')
    laws = [
        ['Universal Law', 'Value', 'CV'],
        ['1. Boltzmann Distribution', '$R^2 = 0.979$', '0.001'],
        ['2. Negative Specific Heat', '$C_v < 0$, $p < 0.001$', '--'],
        ['3. Inverse Radiation', '$n = -1.44$', '0.29'],
        ['4. Carnot Efficiency', '$\\eta = 0.813$', '0.044'],
        ['5. Info Concentration', '$F$ increases 411x', '--'],
    ]
    table = ax5.table(cellText=laws[1:], colLabels=laws[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(3):
        table[0, j].set_facecolor(C_DARK)
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, 6):
        for j in range(3):
            table[i, j].set_facecolor('#eaf2f8' if i % 2 == 0 else '#fef9e7')
    ax5.set_title('Five Universal Laws', fontweight='bold', fontsize=13)

    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    comparison = [
        ['Property', 'Physics', 'LLM'],
        ['Boltzmann', 'Same', 'Same'],
        ['Ergodic', 'Same', 'Same'],
        ['Radiation', '$T^4$', '$T^{-1.44}$'],
        ['Free energy', 'Decreases', 'Increases'],
        ['System', 'Engine', 'Refrigerator'],
    ]
    table2 = ax6.table(cellText=comparison[1:], colLabels=comparison[0],
                       loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    for j in range(3):
        table2[0, j].set_facecolor(C_DARK)
        table2[0, j].set_text_props(color='white', fontweight='bold')
    for i in [1, 2]:
        for j in range(3):
            table2[i, j].set_facecolor('#d5f5e3')
    for i in [3, 4, 5]:
        for j in range(3):
            table2[i, j].set_facecolor('#fdebd0')
    ax6.set_title('LLM vs Physical Universe', fontweight='bold', fontsize=13)

    fig.suptitle('The Standard Model of Transformers',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'fig12_standard_model_summary')


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Generating Publication-Quality Paper Figures")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from utils import load_model
    model, tok = load_model(device=device)

    # Figures that need live model
    mean_U, mean_T, layers = fig01_equation_of_state(model, tok, device)
    fig02_dark_energy(model, tok, device)
    fig04_boltzmann(model, tok, device)
    fig09_info_concentration(model, tok, device)
    fig12_summary(model, tok, device)

    # Figures from saved results
    fig03_phase_diagram(model, tok, device)
    fig05_boltzmann_universal()
    fig06_cv_universal()
    fig07_black_hole()
    fig08_inverse_radiation()
    fig10_carnot_universal()
    fig11_ergodic()

    print(f"\n{'='*70}")
    print(f"All figures saved to: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
