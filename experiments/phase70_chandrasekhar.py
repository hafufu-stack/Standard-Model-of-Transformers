# -*- coding: utf-8 -*-
"""
Phase 70: Chandrasekhar Limit (Maximum Stable Mass)
Is there a critical model size/complexity below which LLMs collapse into
repetition (black hole)? Test with increasing prompt complexity.
When complexity exceeds a threshold, model should undergo gravitational collapse.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 70: Chandrasekhar Limit (Complexity Collapse)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Increasing complexity prompts
    complexity_levels = [
        (1, "The sky is"),
        (2, "The capital of France is"),
        (3, "If x plus five equals twelve then x equals"),
        (4, "The derivative of x squared plus three x minus seven is"),
        (5, "Given that all mammals are warm blooded and whales are mammals therefore"),
        (6, "If the probability of rain is 0.3 and the probability of sunshine is 0.5 and they are independent then"),
        (7, "Consider a system of three differential equations describing predator prey dynamics with a carrying capacity of"),
        (8, "Let G be a finite group of order 120 and H a normal subgroup of order 24 then the quotient group G mod H has"),
        (9, "Using the Euler-Lagrange equations for the functional integral of the Lagrangian density over four dimensional spacetime with metric tensor"),
        (10, "In the framework of algebraic topology the fundamental group of the complement of the trefoil knot in three dimensional Euclidean space"),
    ]

    GEN_LENGTH = 60
    all_results = []

    for complexity, prompt in complexity_levels:
        input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        current_ids = input_ids.clone()

        prt_trace = []
        t_trace = []
        tokens_gen = []
        last_5 = []
        repetition_events = 0

        for t_step in range(GEN_LENGTH):
            with torch.no_grad():
                out = model(current_ids)
                logits = out.logits[0, -1, :].float()

            probs = torch.softmax(logits, dim=-1)
            PR = 1.0 / (probs ** 2).sum().item()
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            PRT = PR * T_val
            prt_trace.append(PRT if not np.isnan(PRT) else 0)
            t_trace.append(T_val if not np.isnan(T_val) else 0)

            next_id = logits.argmax().item()
            tokens_gen.append(next_id)
            last_5.append(next_id)

            if len(last_5) >= 6:
                if last_5[-3:] == last_5[-6:-3]:
                    repetition_events += 1

            next_tensor = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            if current_ids.shape[1] > 512:
                current_ids = current_ids[:, -512:]

        text = tok.decode(tokens_gen, skip_special_tokens=True)

        # Collapse metrics
        T_final = np.mean(t_trace[-10:]) if len(t_trace) >= 10 else 0
        T_initial = np.mean(t_trace[:10]) if len(t_trace) >= 10 else 0
        T_drop = (T_initial - T_final) / (T_initial + 1e-10)
        prt_std = float(np.std(prt_trace))
        token_diversity = len(set(tokens_gen)) / (len(tokens_gen) + 1e-10)
        collapsed = T_final < 0.5 or token_diversity < 0.2

        safe_text = text.encode('ascii', errors='replace').decode('ascii')[:50]
        print(f"  C={complexity:2d}: T_drop={T_drop:.2f}, div={token_diversity:.2f}, "
              f"reps={repetition_events}, {'COLLAPSE' if collapsed else 'stable'}: "
              f"'{safe_text}...'")

        all_results.append({
            'complexity': complexity, 'prompt': prompt[:60],
            'T_final': float(T_final), 'T_initial': float(T_initial),
            'T_drop': float(T_drop), 'prt_std': float(prt_std),
            'token_diversity': float(token_diversity),
            'repetition_events': repetition_events,
            'collapsed': collapsed,
            't_trace': [float(t) for t in t_trace],
        })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    complexities = [r['complexity'] for r in all_results]
    diversities = [r['token_diversity'] for r in all_results]
    T_finals = [r['T_final'] for r in all_results]
    collapses = [r['collapsed'] for r in all_results]

    # (a) Diversity vs complexity
    colors_c = ['#e74c3c' if c else '#2ecc71' for c in collapses]
    axes[0, 0].scatter(complexities, diversities, c=colors_c, s=100, edgecolors='black')
    axes[0, 0].plot(complexities, diversities, '--', color='gray', alpha=0.5)
    axes[0, 0].set_xlabel('Prompt Complexity')
    axes[0, 0].set_ylabel('Token Diversity')
    axes[0, 0].set_title('(a) Diversity vs Complexity')

    # (b) T_final vs complexity
    axes[0, 1].scatter(complexities, T_finals, c=colors_c, s=100, edgecolors='black')
    axes[0, 1].plot(complexities, T_finals, '--', color='gray', alpha=0.5)
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', label='Collapse threshold')
    axes[0, 1].set_xlabel('Prompt Complexity')
    axes[0, 1].set_ylabel('Final Temperature T')
    axes[0, 1].set_title('(b) T_final vs Complexity')
    axes[0, 1].legend()

    # (c) T traces
    for r in all_results:
        c = '#e74c3c' if r['collapsed'] else '#2ecc71'
        axes[0, 2].plot(r['t_trace'], color=c, alpha=0.5, linewidth=1,
                       label=f'C={r["complexity"]}' if r['complexity'] <= 3 else '')
    axes[0, 2].set_xlabel('Token')
    axes[0, 2].set_ylabel('Temperature T')
    axes[0, 2].set_title('(c) T Traces (red=collapsed)')

    # (d) Repetition events
    reps = [r['repetition_events'] for r in all_results]
    axes[1, 0].bar(complexities, reps, color=colors_c, alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('Complexity')
    axes[1, 0].set_ylabel('Repetition Events')
    axes[1, 0].set_title('(d) Repetition vs Complexity')

    # (e) PRT stability
    prt_stds = [r['prt_std'] for r in all_results]
    axes[1, 1].plot(complexities, prt_stds, 'o-', color='#f39c12', linewidth=2)
    axes[1, 1].set_xlabel('Complexity')
    axes[1, 1].set_ylabel('PRT std')
    axes[1, 1].set_title('(e) PRT Instability')

    # (f) Phase diagram: T_final vs diversity
    sc = axes[1, 2].scatter(diversities, T_finals, c=complexities,
                            cmap='plasma', s=100, edgecolors='black')
    axes[1, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Token Diversity')
    axes[1, 2].set_ylabel('Final T')
    axes[1, 2].set_title('(f) Phase Diagram')
    plt.colorbar(sc, ax=axes[1, 2], label='Complexity')

    # Find Chandrasekhar limit
    chandrasekhar = None
    for r in all_results:
        if r['collapsed']:
            chandrasekhar = r['complexity']
            break

    fig.suptitle(f'Phase 70: Chandrasekhar Limit '
                 f'(collapse at C={chandrasekhar if chandrasekhar else ">10"})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase70_chandrasekhar')
    plt.close()

    n_collapsed = sum(collapses)
    print(f"\n{'='*70}")
    print(f"VERDICT: {n_collapsed}/{len(all_results)} prompts collapsed. "
          f"Chandrasekhar limit at complexity={chandrasekhar if chandrasekhar else '>10'}. "
          f"{'CRITICAL MASS EXISTS' if chandrasekhar else 'No collapse detected'}.")
    print(f"{'='*70}")

    save_results('phase70_chandrasekhar', {
        'experiment': 'Chandrasekhar Limit',
        'summary': {
            'n_collapsed': n_collapsed,
            'chandrasekhar_limit': chandrasekhar,
        }
    })


if __name__ == '__main__':
    main()
