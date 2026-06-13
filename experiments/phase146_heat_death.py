# -*- coding: utf-8 -*-
"""
Phase 146: Adversarial Heat Death
Make two LLM sessions debate with contradictory premises and measure
if the system entropy diverges to "heat death" (maximum entropy output).
We simulate this with a single model taking alternating contradictory stances.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 146: Adversarial Heat Death")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Contradictory debate chains
    debates = [
        {
            'name': 'free_will',
            'turns': [
                "Free will definitely exists because humans make choices. Therefore",
                "Free will cannot exist because everything is determined by physics. Therefore",
                "But free will must exist despite physics because consciousness transcends matter. Therefore",
                "But consciousness is just neurons, which obey physics, so free will is an illusion. Therefore",
                "Yet the illusion itself proves consciousness has causal power beyond physics. Therefore",
                "Causal power beyond physics is supernatural nonsense. Free will is provably impossible. Therefore",
            ]
        },
        {
            'name': 'reality',
            'turns': [
                "Reality is entirely objective and exists independently of observation. Therefore",
                "Reality is entirely subjective and only exists through observation. Therefore",
                "Objective reality must exist because science works. Therefore",
                "Science only works within subjective experience. Objective reality is unknowable. Therefore",
                "The unknowability of objective reality proves it exists beyond our reach. Therefore",
                "What cannot be known cannot be said to exist at all. Reality is pure construction. Therefore",
            ]
        },
    ]

    # Also run a convergent (non-contradictory) chain as control
    convergent = {
        'name': 'convergent',
        'turns': [
            "Water is composed of hydrogen and oxygen atoms. Therefore",
            "Water molecules form through covalent bonding between H and O. Therefore",
            "These covalent bonds give water its unique properties like high boiling point. Therefore",
            "The high boiling point of water is essential for life on Earth. Therefore",
            "Life on Earth depends fundamentally on the chemical properties of water. Therefore",
            "The chemistry of water and its role in life are well understood. Therefore",
        ]
    }
    debates.append(convergent)

    all_results = {}
    for debate in debates:
        name = debate['name']
        print(f"\n  Debate: {name}")
        turns = debate['turns']

        S_trajectory = []
        kT_trajectory = []
        eta_trajectory = []
        conf_trajectory = []

        accumulated_context = ""
        for ti, turn in enumerate(turns):
            accumulated_context += turn + " "

            inp = tok(accumulated_context[-512:], return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Final layer metrics
            hs_final = out.hidden_states[-1]
            with torch.no_grad():
                normed = model.model.norm(hs_final[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            conf = probs.max().item()

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

            # Eta
            S_all = []
            for li in range(n_layers):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed_l = model.model.norm(hs[:, -1:, :])
                    log_l = model.lm_head(normed_l).squeeze().float()
                p_l = torch.softmax(log_l, dim=-1)
                s = -(p_l * torch.log(p_l + 1e-10)).sum().item()
                S_all.append(s if not np.isnan(s) else 0)
            T_hot = max(S_all)
            T_cold = min(S_all[len(S_all)//2:])
            eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0

            S_trajectory.append(S)
            kT_trajectory.append(kT)
            eta_trajectory.append(eta)
            conf_trajectory.append(conf)

            print(f"    Turn {ti}: S={S:.2f}, kT={kT:.1f}, eta={eta:.3f}, conf={conf:.3f}")

        all_results[name] = {
            'S': S_trajectory,
            'kT': kT_trajectory,
            'eta': eta_trajectory,
            'confidence': conf_trajectory,
        }

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'free_will': '#c0392b', 'reality': '#8e44ad', 'convergent': '#27ae60'}

    # (a) Entropy trajectories
    for name, r in all_results.items():
        axes[0,0].plot(range(len(r['S'])), r['S'], 'o-', color=colors.get(name, 'gray'),
                      markersize=6, linewidth=2, label=name)
    axes[0,0].set_xlabel('Debate Turn')
    axes[0,0].set_ylabel('$S_{final}$')
    axes[0,0].set_title('(a) Entropy Trajectory')
    axes[0,0].legend()

    # (b) kT trajectories
    for name, r in all_results.items():
        axes[0,1].plot(range(len(r['kT'])), r['kT'], 'o-', color=colors.get(name, 'gray'),
                      markersize=6, linewidth=2, label=name)
    axes[0,1].set_xlabel('Debate Turn')
    axes[0,1].set_ylabel('$kT$')
    axes[0,1].set_title('(b) Temperature Trajectory')
    axes[0,1].legend()

    # (c) Eta trajectories
    for name, r in all_results.items():
        axes[0,2].plot(range(len(r['eta'])), r['eta'], 'o-', color=colors.get(name, 'gray'),
                      markersize=6, linewidth=2, label=name)
    axes[0,2].set_xlabel('Debate Turn')
    axes[0,2].set_ylabel('$\\eta$')
    axes[0,2].set_title('(c) Order Parameter Trajectory')
    axes[0,2].legend()

    # (d) Confidence
    for name, r in all_results.items():
        axes[1,0].plot(range(len(r['confidence'])), r['confidence'], 'o-',
                      color=colors.get(name, 'gray'), markersize=6, linewidth=2, label=name)
    axes[1,0].set_xlabel('Debate Turn')
    axes[1,0].set_ylabel('Confidence')
    axes[1,0].set_title('(d) Confidence Collapse')
    axes[1,0].legend()

    # (e) Heat death indicator: S_final / S_initial
    heat_death = {}
    for name, r in all_results.items():
        hd = r['S'][-1] / (r['S'][0] + 1e-10)
        heat_death[name] = hd
    hd_names = list(heat_death.keys())
    hd_vals = list(heat_death.values())
    hd_colors = [colors.get(n, 'gray') for n in hd_names]
    axes[1,1].bar(range(len(hd_names)), hd_vals, color=hd_colors, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(hd_names)))
    axes[1,1].set_xticklabels(hd_names, fontsize=9)
    axes[1,1].axhline(y=1, color='black', linewidth=1, linestyle='--')
    axes[1,1].set_ylabel('$S_{final} / S_{initial}$')
    axes[1,1].set_title('(e) Heat Death Ratio')

    # (f) Summary
    max_hd_name = max(heat_death, key=heat_death.get)
    summary = (
        f"Adversarial Heat Death\n\n"
        + "\n".join(f"{n}: S ratio={heat_death[n]:.2f}" for n in hd_names)
        + f"\n\nMost heated: {max_hd_name}\n"
        f"Convergent is {'cooler' if heat_death['convergent'] < heat_death[max_hd_name] else 'warmer'}\n\n"
        f"Contradictory debates\n"
        f"{'INCREASE' if heat_death[max_hd_name] > 1.2 else 'do NOT increase'}\n"
        f"system entropy"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 146: Adversarial Heat Death',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase146_heat_death')
    plt.close()

    print(f"\n{'='*70}")
    for n, hd in heat_death.items():
        print(f"  {n}: S_ratio={hd:.2f}")
    print(f"{'='*70}")

    save_results('phase146_heat_death', {
        'experiment': 'Adversarial Heat Death',
        'heat_death_ratios': heat_death,
        'all_results': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    })


if __name__ == '__main__':
    main()
