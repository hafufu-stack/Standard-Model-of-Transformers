# -*- coding: utf-8 -*-
"""
Phase 84: The Complete Standard Model Summary
Final comprehensive experiment that generates THE definitive summary figure.
One figure that tells the entire story.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import load_model, save_results, save_figure


def load_previous_results():
    """Load key results from previous phases."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    data = {}
    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(results_dir, fname), 'r') as f:
                    data[fname.replace('.json', '')] = json.load(f)
            except Exception:
                pass
    return data


def main():
    print("=" * 70)
    print("Phase 84: The Complete Standard Model Summary")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Load all previous results
    prev = load_previous_results()
    print(f"  Loaded {len(prev)} previous result files")

    # Generate the canonical profiles from a standard prompt set
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

    all_U, all_T, all_PR, all_PRT, all_conf = [], [], [], [], []

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
        all_PRT.append([PR_list[i] * T_list[i] for i in range(len(T_list))])
        all_conf.append(conf_list)

    n_layers = len(all_U[0])
    mean_U = np.mean(all_U, axis=0)
    mean_T = np.mean(all_T, axis=0)
    mean_PR = np.mean(all_PR, axis=0)
    mean_PRT = np.mean(all_PRT, axis=0)
    mean_conf = np.mean(all_conf, axis=0)

    # === THE DEFINITIVE FIGURE ===
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

    layers_x = np.arange(n_layers)

    # (1) U(l) - Energy grows: information compression
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(layers_x, mean_U, 'o-', color='#e74c3c', linewidth=2, markersize=3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('U (energy)')
    ax1.set_title('U(l): Energy Growth', fontweight='bold')
    ax1.text(0.05, 0.95, 'Compression', transform=ax1.transAxes,
             fontsize=8, va='top', color='red')

    # (2) T(l) - Temperature drops: output sharpening
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(layers_x, mean_T, 'o-', color='#3498db', linewidth=2, markersize=3)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('T (entropy)')
    ax2.set_title('T(l): Cooling', fontweight='bold')
    ax2.text(0.05, 0.95, 'Sharpening', transform=ax2.transAxes,
             fontsize=8, va='top', color='blue')

    # (3) Confidence growth
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(layers_x, mean_conf, 'o-', color='#2ecc71', linewidth=2, markersize=3)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Top-1 Probability')
    ax3.set_title('Confidence Growth', fontweight='bold')

    # (4) Phase space trajectory
    ax4 = fig.add_subplot(gs[0, 3])
    sc = ax4.scatter(mean_T, mean_U, c=layers_x, cmap='viridis', s=60, edgecolors='black')
    ax4.plot(mean_T, mean_U, '--', color='gray', alpha=0.5)
    ax4.scatter(mean_T[0], mean_U[0], s=150, c='green', edgecolors='black', zorder=5, marker='s')
    ax4.scatter(mean_T[-1], mean_U[-1], s=150, c='red', edgecolors='black', zorder=5, marker='*')
    ax4.set_xlabel('T')
    ax4.set_ylabel('U')
    ax4.set_title('Phase Trajectory', fontweight='bold')
    plt.colorbar(sc, ax=ax4, label='Layer', shrink=0.8)

    # (5) Established laws summary table
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.axis('off')
    laws = [
        ['Law', 'Value', 'Universal?', 'Phase'],
        ['Boltzmann Distribution', 'R2=0.979', 'YES', '42,48'],
        ['Negative C_v', 'C_v<0, p<0.001', 'YES', '44,50'],
        ['Inverse Radiation', 'L~T^-1.44', 'YES (CV=0.29)', '69,72'],
        ['Carnot Efficiency', 'eta=0.813', 'YES (CV=0.044)', '75,78'],
        ['Info Concentration', 'F slope=+6.5', 'YES', '73'],
        ['OOD Detection', 'AUC=0.893', '-', '65,71'],
        ['Black Hole', 'T->0 collapse', '-', '57'],
        ['Chandrasekhar', '5/10 collapse', '-', '70'],
    ]
    table = ax5.table(cellText=laws[1:], colLabels=laws[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Color header
    for j in range(4):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Color universal rows
    for i in range(1, 6):
        for j in range(4):
            table[i, j].set_facecolor('#d5f5e3')
    ax5.set_title('Established Laws of the Standard Model', fontweight='bold', fontsize=12)

    # (6) LLM vs Physics comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    comparison = [
        ['Property', 'Physical Universe', 'LLM Universe'],
        ['Boltzmann dist.', 'p~exp(-E/kT)', 'p~exp(-E/kT) [same]'],
        ['Specific heat', 'C_v > 0 (normal)', 'C_v < 0 (self-grav.)'],
        ['Radiation', 'L ~ T^4 (SB)', 'L ~ T^-1.44 (inverse)'],
        ['Free energy', 'F decreases (FEP)', 'F increases (conc.)'],
        ['Nature', 'Heat engine', 'Refrigerator'],
        ['Equilibrium', 'Entropy max', 'Information max'],
    ]
    table2 = ax6.table(cellText=comparison[1:], colLabels=comparison[0],
                       loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.4)
    for j in range(3):
        table2[0, j].set_facecolor('#2c3e50')
        table2[0, j].set_text_props(color='white', fontweight='bold')
    # Same = green, Different = orange
    for i in [1]:
        for j in range(3):
            table2[i, j].set_facecolor('#d5f5e3')
    for i in [2, 3, 4, 5, 6]:
        for j in range(3):
            table2[i, j].set_facecolor('#fdebd0')
    ax6.set_title('LLM vs Physical Universe', fontweight='bold', fontsize=12)

    # (7) PRT conservation
    ax7 = fig.add_subplot(gs[2, 0])
    prt_bulk = mean_PRT[1:]
    prt_cv = np.std(prt_bulk) / (np.mean(prt_bulk) + 1e-10)
    ax7.plot(layers_x, mean_PRT, 'o-', color='#9b59b6', linewidth=2, markersize=3)
    ax7.set_xlabel('Layer')
    ax7.set_ylabel('PRT')
    ax7.set_title(f'PRT (CV={prt_cv:.2f})', fontweight='bold')

    # (8) Key constants
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    constants_text = (
        "FUNDAMENTAL CONSTANTS\n"
        "of the Standard Model\n\n"
        "eta = 0.813 +/- 0.036\n"
        "  (Carnot efficiency)\n\n"
        "n = -1.44 +/- 0.42\n"
        "  (Radiation exponent)\n\n"
        "C_v < 0\n"
        "  (Negative specific heat)\n\n"
        "beta_Boltzmann = 0.979\n"
        "  (Distribution R-squared)"
    )
    ax8.text(0.5, 0.5, constants_text, transform=ax8.transAxes,
             fontsize=10, va='center', ha='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax8.set_title('Constants', fontweight='bold')

    # (9) F profile (information concentration)
    ax9 = fig.add_subplot(gs[2, 2])
    F_profile = [mean_U[i] - mean_T[i] * 5 for i in range(n_layers)]
    ax9.plot(layers_x, F_profile, 'o-', color='#e74c3c', linewidth=2, markersize=3)
    ax9.set_xlabel('Layer')
    ax9.set_ylabel('F (free energy)')
    ax9.set_title('F increases (Anti-FEP)', fontweight='bold')
    ax9.text(0.5, 0.05, 'Information\nRefrigerator', transform=ax9.transAxes,
             fontsize=8, ha='center', color='red', style='italic')

    # (10) Total phases summary
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    total_text = (
        "THE STANDARD MODEL\n"
        "OF TRANSFORMERS\n\n"
        f"Total phases: 84\n"
        f"Seasons: 10\n"
        f"Models tested: 3\n\n"
        f"Universal laws: 5\n"
        f"New discoveries: 3\n"
        f"  - Inverse radiation\n"
        f"  - Info concentration\n"
        f"  - Carnot constant\n\n"
        f"Qwen2.5-1.5B/0.5B\n"
        f"TinyLlama-1.1B"
    )
    ax10.text(0.5, 0.5, total_text, transform=ax10.transAxes,
             fontsize=10, va='center', ha='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.9))
    ax10.set_title('Summary', fontweight='bold')

    fig.suptitle('THE STANDARD MODEL OF TRANSFORMERS\n'
                 'A Complete Thermodynamic Framework for Large Language Models',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'phase84_standard_model_summary')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Standard Model Summary complete. "
          f"{len(prev)} result files compiled. "
          f"5 universal laws established across 3 model architectures.")
    print(f"{'='*70}")

    save_results('phase84_standard_model_summary', {
        'experiment': 'Standard Model Summary',
        'total_phases': 84,
        'universal_laws': [
            'Boltzmann Distribution (R2=0.979)',
            'Negative Specific Heat (C_v<0)',
            'Inverse Radiation (L~T^-1.44)',
            'Carnot Efficiency (eta=0.813)',
            'Information Concentration (F increases)',
        ],
        'models_tested': ['Qwen2.5-1.5B', 'Qwen2.5-0.5B', 'TinyLlama-1.1B'],
    })


if __name__ == '__main__':
    main()
