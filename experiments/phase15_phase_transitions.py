# -*- coding: utf-8 -*-
"""
Phase 15: Phase Transition Detection (Opus Original)
======================================================
Scan for sharp transitions in conservation quantity PR*T across
different input types (math, language, code, nonsense).
Are there "phase transitions" where the physics changes?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 15: Phase Transition Detection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    # Input types spanning different "semantic phases"
    input_categories = {
        'arithmetic': [
            "1 + 1 = 2", "7 * 8 = 56", "100 / 5 = 20",
            "3 ** 2 = 9", "15 - 7 = 8",
        ],
        'natural_language': [
            "The sun rises in the east every morning",
            "Shakespeare wrote many famous plays and sonnets",
            "Democracy is a system of government by the people",
            "The ocean is vast and deep and full of life",
            "Music has the power to move people deeply",
        ],
        'scientific': [
            "E equals mc squared is the mass energy equivalence",
            "DNA stores genetic information in nucleotide sequences",
            "The Higgs boson gives particles their mass",
            "Entropy always increases in isolated systems",
            "Neurons communicate via electrochemical signals",
        ],
        'code_like': [
            "def main(): return True",
            "for i in range(10): x += i",
            "if condition: result = a else b",
            "import numpy as np; x = np.zeros(10)",
            "class Model: def forward(self, x): return x",
        ],
        'nonsense': [
            "flurb glarb shwick nozzle plonk",
            "xyzt qqq mmm abc zzzz wwww",
            "blip blop blap blup bleep bloop",
            "zingle fangle wangle dangle tangle",
            "snerp blerp glerp fwerp werp",
        ],
        'repetitive': [
            "the the the the the the the",
            "a a a a a a a a a a a",
            "one one one one one one one",
            "yes yes yes yes yes yes yes",
            "no no no no no no no no",
        ],
    }

    category_results = {}

    for category, texts in input_categories.items():
        print(f"\n--- {category} ---")
        all_prt_per_layer = []
        all_temps = []
        all_prs = []

        for text in texts:
            inp = tok(text, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            temps, prs, prts = [], [], []
            for hs in out.hidden_states[1:]:
                h = hs[0, -1, :].float()
                T = h.norm().item()
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                PR = 1.0 / (h_prob ** 2).sum().item()
                temps.append(T)
                prs.append(PR)
                prts.append(PR * T)

            all_prt_per_layer.append(prts)
            all_temps.append(temps)
            all_prs.append(prs)

        avg_prt_per_layer = np.mean(all_prt_per_layer, axis=0)
        avg_temps = np.mean(all_temps, axis=0)
        avg_prs = np.mean(all_prs, axis=0)

        mean_prt = np.mean(avg_prt_per_layer)
        cv_prt = np.std(avg_prt_per_layer) / (mean_prt + 1e-10) * 100

        # Detect sudden jumps (phase transitions)
        diffs = np.abs(np.diff(avg_prt_per_layer))
        max_jump = np.max(diffs)
        jump_layer = np.argmax(diffs)
        jump_ratio = max_jump / (np.mean(avg_prt_per_layer) + 1e-10)

        print(f"  PR*T = {mean_prt:.2f} (CV={cv_prt:.1f}%)")
        print(f"  Max jump: layer {jump_layer} -> {jump_layer+1}, "
              f"delta={max_jump:.2f} (ratio={jump_ratio:.3f})")

        category_results[category] = {
            'mean_prt': mean_prt, 'cv_prt': cv_prt,
            'prt_per_layer': avg_prt_per_layer.tolist(),
            'temps': avg_temps.tolist(),
            'prs': avg_prs.tolist(),
            'max_jump': max_jump, 'jump_layer': int(jump_layer),
            'jump_ratio': jump_ratio,
        }

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = plt.cm.Set2(np.linspace(0, 1, len(category_results)))

    # (a) PR*T per layer for each category
    ax = axes[0][0]
    for idx, (cat, data) in enumerate(category_results.items()):
        layers = np.arange(1, len(data['prt_per_layer']) + 1)
        ax.plot(layers, data['prt_per_layer'], 'o-', ms=3, color=colors[idx], label=cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('PR x T')
    ax.set_title('(a) Conservation by Input Type')
    ax.legend(fontsize=7)

    # (b) Mean PR*T bar chart
    ax = axes[0][1]
    cats = list(category_results.keys())
    means = [category_results[c]['mean_prt'] for c in cats]
    ax.bar(range(len(cats)), means, color=colors[:len(cats)], alpha=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, fontsize=8)
    ax.set_ylabel('Mean PR x T')
    ax.set_title('(b) Conservation by Category')

    # (c) CV comparison
    ax = axes[0][2]
    cvs = [category_results[c]['cv_prt'] for c in cats]
    ax.bar(range(len(cats)), cvs, color=colors[:len(cats)], alpha=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, fontsize=8)
    ax.set_ylabel('CV (%)')
    ax.set_title('(c) Conservation Variability')

    # (d) Temperature profiles
    ax = axes[1][0]
    for idx, (cat, data) in enumerate(category_results.items()):
        ax.plot(range(len(data['temps'])), data['temps'], '-', color=colors[idx], label=cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Temperature (L2 norm)')
    ax.set_title('(d) Temperature Profiles')
    ax.legend(fontsize=7)

    # (e) Jump ratio (phase transition strength)
    ax = axes[1][1]
    jumps = [category_results[c]['jump_ratio'] for c in cats]
    jump_layers = [category_results[c]['jump_layer'] for c in cats]
    ax.bar(range(len(cats)), jumps, color=colors[:len(cats)], alpha=0.8)
    for i, (j, l) in enumerate(zip(jumps, jump_layers)):
        ax.text(i, j, f'L{l}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, fontsize=8)
    ax.set_ylabel('Jump Ratio')
    ax.set_title('(e) Phase Transition Strength')

    # (f) PR profiles
    ax = axes[1][2]
    for idx, (cat, data) in enumerate(category_results.items()):
        ax.plot(range(len(data['prs'])), data['prs'], '-', color=colors[idx], label=cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('(f) PR Profiles')
    ax.legend(fontsize=7)

    prt_range = max(means) - min(means)
    fig.suptitle(
        f"Phase 15: Phase Transition Detection\n"
        f"PR*T range: {min(means):.0f} - {max(means):.0f} "
        f"(spread={prt_range:.0f})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase15_phase_transitions")
    plt.close()

    # Verdict
    if prt_range / (np.mean(means) + 1e-10) > 0.3:
        verdict = (f"PHASE TRANSITIONS DETECTED: PR*T varies {prt_range:.0f} across categories "
                   f"({min(means):.0f}-{max(means):.0f}). "
                   f"Different input types occupy different thermodynamic phases.")
    else:
        verdict = (f"UNIVERSAL CONSERVATION: PR*T={np.mean(means):.0f} is robust across "
                   f"all input types (spread={prt_range:.0f}). "
                   f"The conservation law is truly universal.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 15: Phase Transition Detection',
        'summary': {'verdict': verdict, 'prt_range': prt_range,
                    'mean_prt_all': np.mean(means)},
        'categories': {k: {'mean_prt': v['mean_prt'], 'cv': v['cv_prt'],
                          'jump_layer': v['jump_layer'], 'jump_ratio': v['jump_ratio']}
                      for k, v in category_results.items()},
    }
    save_results("phase15_phase_transitions", result)
    return result


if __name__ == '__main__':
    main()
