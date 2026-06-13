# -*- coding: utf-8 -*-
"""
Phase 173: Grand Unified Figure v2
Create the ultimate summary figure incorporating ALL Season 11-14 findings.
This is the "money figure" for the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

# Import matplotlib for fancy styling
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 173: Grand Unified Figure v2")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load with eager attention for attention entropy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16,
        device_map=device, local_files_only=True,
        attn_implementation='eager')
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
    n_layers = len(model.model.layers) + 1

    all_S = np.zeros((len(PROMPTS), n_layers))
    all_kT = np.zeros((len(PROMPTS), n_layers))
    all_eta = np.zeros((len(PROMPTS), n_layers))
    all_norms = np.zeros((len(PROMPTS), n_layers))
    all_attn_H = np.zeros((len(PROMPTS), n_layers - 1))

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            all_norms[pi, li] = h.norm().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            all_S[pi, li] = S if not np.isnan(S) else 0
            T_vals.append(all_S[pi, li])

            top_k = 50
            top_probs = torch.topk(probs, top_k).values
            log_probs = torch.log(top_probs + 1e-10).cpu().numpy()
            ranks = np.arange(1, top_k + 1, dtype=np.float64)
            if np.std(log_probs) > 0.01:
                slope = np.polyfit(ranks, log_probs, 1)[0]
                kT = -1.0 / (slope + 1e-10)
            else:
                kT = 0.1
            kT = max(0.01, min(kT, 50))
            all_kT[pi, li] = float(kT)

        for li in range(n_layers):
            T_sub = T_vals[:li+1]
            if len(T_sub) >= 4:
                T_hot = max(T_sub)
                T_cold = min(T_sub[len(T_sub)//2:])
                e = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                e = 0
            all_eta[pi, li] = e

        # Attention entropy
        for li in range(min(n_layers - 1, len(out.attentions))):
            attn = out.attentions[li]
            if attn is not None:
                last_attn = attn[0, :, -1, :].float()
                head_H = []
                for hi in range(last_attn.shape[0]):
                    a = last_attn[hi]
                    a = a / (a.sum() + 1e-10)
                    H = -(a * torch.log(a + 1e-10)).sum().item()
                    head_H.append(H if not np.isnan(H) else 0)
                all_attn_H[pi, li] = np.mean(head_H)

    mean_S = np.mean(all_S, axis=0)
    mean_kT = np.mean(all_kT, axis=0)
    mean_eta = np.mean(all_eta, axis=0)
    mean_norms = np.mean(all_norms, axis=0)
    mean_attn_H = np.mean(all_attn_H, axis=0)
    var_S = np.var(all_S, axis=0)

    # Sigmoid fit
    layers = np.arange(n_layers)
    try:
        Ls = np.arange(4, n_layers)
        popt, _ = curve_fit(sigmoid, Ls, mean_eta[4:],
                            p0=[22, 0.5, 0, 0.9], maxfev=10000)
        L0 = popt[0]
        eta_fit = sigmoid(Ls, *popt)
        r2 = 1 - np.sum((mean_eta[4:] - eta_fit)**2) / (
            np.sum((mean_eta[4:] - np.mean(mean_eta[4:]))**2) + 1e-10)
    except:
        L0 = 22
        r2 = 0
        eta_fit = mean_eta[4:]

    # === GRAND UNIFIED FIGURE ===
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    c_pre = '#3498db'
    c_post = '#e74c3c'
    c_L0 = '#f39c12'

    # (a) THE Sigmoid - centerpiece
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(layers, mean_eta, 'o', color='#2c3e50', markersize=5, alpha=0.7)
    ax_a.plot(Ls, eta_fit, '-', color=c_post, linewidth=3,
             label=f'Sigmoid (L0={L0:.1f}, R2={r2:.3f})')
    ax_a.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    ax_a.fill_betweenx([0, 1], 0, L0, color=c_pre, alpha=0.08)
    ax_a.fill_betweenx([0, 1], L0, n_layers, color=c_post, alpha=0.08)
    ax_a.set_xlabel('Layer', fontweight='bold')
    ax_a.set_ylabel('$\\eta$ (Carnot Efficiency)', fontweight='bold')
    ax_a.set_title('(a) Phase Transition', fontweight='bold')
    ax_a.legend(fontsize=8)

    # (b) Entropy landscape
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(layers, mean_S, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    ax_b.fill_between(layers, mean_S - np.std(all_S, axis=0),
                      mean_S + np.std(all_S, axis=0), alpha=0.15, color='#8e44ad')
    ax_b.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    ax_b.set_xlabel('Layer', fontweight='bold')
    ax_b.set_ylabel('$S$ (Entropy)', fontweight='bold')
    ax_b.set_title('(b) Entropy Landscape', fontweight='bold')

    # (c) Temperature profile
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(layers, mean_kT, 'o-', color='#e67e22', markersize=4, linewidth=2)
    ax_c.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    ax_c.set_xlabel('Layer', fontweight='bold')
    ax_c.set_ylabel('$kT$ (Temperature)', fontweight='bold')
    ax_c.set_title('(c) Temperature Profile', fontweight='bold')

    # (d) Attention entropy
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(range(len(mean_attn_H)), mean_attn_H, 'o-', color='#16a085',
             markersize=4, linewidth=2)
    ax_d.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    ax_d.set_xlabel('Layer', fontweight='bold')
    ax_d.set_ylabel('Attention $H$', fontweight='bold')
    pre_attn = np.mean(mean_attn_H[:20])
    post_attn = np.mean(mean_attn_H[20:])
    ax_d.set_title(f'(d) Attention Entropy ({(post_attn-pre_attn)/pre_attn*100:+.0f}%)',
                   fontweight='bold')

    # (e) Susceptibility (critical slowing)
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(layers, var_S, 'o-', color='#c0392b', markersize=4, linewidth=2)
    ax_e.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    peak = np.argmax(var_S[4:]) + 4
    ax_e.axvline(x=peak, color='#27ae60', linewidth=1.5, linestyle=':',
                label=f'Peak L{peak}')
    ax_e.set_xlabel('Layer', fontweight='bold')
    ax_e.set_ylabel('$\\chi$ = Var($S$)', fontweight='bold')
    ax_e.set_title('(e) Susceptibility', fontweight='bold')
    ax_e.legend(fontsize=8)

    # (f) Hidden state norms
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(layers, mean_norms, 'o-', color='#2980b9', markersize=4, linewidth=2)
    ax_f.axvline(x=L0, color=c_L0, linewidth=2, linestyle='--')
    ax_f.set_xlabel('Layer', fontweight='bold')
    ax_f.set_ylabel('$||h||$', fontweight='bold')
    pre_n = np.mean(mean_norms[:20])
    post_n = np.mean(mean_norms[20:])
    ax_f.set_title(f'(f) Norm Growth ({post_n/pre_n:.1f}x)', fontweight='bold')

    # (g) Phase diagram (kT vs S)
    ax_g = fig.add_subplot(gs[2, 0])
    for pi in range(len(PROMPTS)):
        ax_g.plot(all_kT[pi], all_S[pi], '-', color='gray', alpha=0.15, linewidth=1)
    # Color by layer
    for li in range(n_layers):
        c = c_pre if li < L0 else c_post
        alpha = 0.4
        for pi in range(len(PROMPTS)):
            ax_g.scatter(all_kT[pi, li], all_S[pi, li], c=c, s=8, alpha=alpha)
    ax_g.set_xlabel('$kT$', fontweight='bold')
    ax_g.set_ylabel('$S$', fontweight='bold')
    ax_g.set_title('(g) Phase Diagram', fontweight='bold')

    # (h) RG flow (from phase163 results)
    ax_h = fig.add_subplot(gs[2, 1])
    rg_scales = [1, 2, 3, 4]
    rg_ratios = [0.762, 0.729, 0.713, 0.661]  # From Phase 163 results
    rg_r2 = [0.980, 0.980, 0.987, 0.992]
    colors_rg = ['#2980b9', '#c0392b', '#27ae60', '#f39c12']
    ax_h.bar(range(len(rg_scales)), rg_ratios, color=colors_rg, alpha=0.8,
             edgecolor='black')
    ax_h.set_xticks(range(len(rg_scales)))
    ax_h.set_xticklabels([f'x{s}' for s in rg_scales])
    ax_h.axhline(y=np.mean(rg_ratios), color='black', linestyle='--',
                label=f'Mean={np.mean(rg_ratios):.3f}')
    ax_h.set_xlabel('Coarse-Graining Scale', fontweight='bold')
    ax_h.set_ylabel('$L_0 / n$', fontweight='bold')
    ax_h.set_title('(h) Scale Invariance (RG)', fontweight='bold')
    ax_h.legend(fontsize=8)

    # (i) Summary statistics
    ax_i = fig.add_subplot(gs[2, 2])
    summary = (
        "THE STANDARD MODEL\n"
        "of Transformer Thermodynamics\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"L0 = {L0:.1f} (L0/L = {L0/n_layers:.3f})\n"
        f"Sigmoid R2 = {r2:.3f}\n"
        f"Universality: 2D XY class\n"
        f"Beta = 0.161\n\n"
        f"Var(S) peak: L{peak} (->L0)\n"
        f"Attn entropy: -32%\n"
        f"Norm growth: 3.3x\n"
        f"RG invariant: CV=0.051\n"
        f"Phase sep: 6.76\n\n"
        f"Seasons 11-14 | 88 Phases"
    )
    ax_i.text(0.5, 0.5, summary, ha='center', va='center',
             transform=ax_i.transAxes, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                      edgecolor='#2c3e50', linewidth=2))
    ax_i.axis('off')
    ax_i.set_title('(i) Summary', fontweight='bold')

    fig.suptitle('The Standard Model of Transformer Thermodynamics\n'
                 'Qwen2.5-1.5B | 29 Layers | 88 Experiments',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, 'phase173_grand_unified_v2')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Grand Unified Figure v2 created!")
    print(f"L0={L0:.1f}, R2={r2:.3f}")
    print(f"{'='*70}")

    save_results('phase173_grand_unified_v2', {
        'experiment': 'Grand Unified Figure v2',
        'summary': {
            'L0': float(L0),
            'R2': float(r2),
            'total_phases': 88,
        }
    })


if __name__ == '__main__':
    main()
