# -*- coding: utf-8 -*-
"""
Phase 9: Strict Contextual CHSH Recovery
==========================================
Season 1's Phase 1 showed S<2.0 (classical). But S-Qubit Q15 got S=3.41.
Hypothesis: Super-quantum correlations emerge ONLY under strong
semantic binding (meaning-laden context), not random tokens.

Reproduce Q15's exact "soul vector" protocol with entangled concepts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, get_hidden_states, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 9: Contextual CHSH Recovery (Soul Vector Protocol)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Q15's exact protocol: measure semantic entanglement via
    # cosine similarity of hidden states in complementary contexts
    # Strong semantic binding pairs (entangled)
    entangled_pairs = [
        ("The cat is alive and well", "The cat is dead and gone"),
        ("The electron has spin up", "The electron has spin down"),
        ("The answer is true", "The answer is false"),
        ("Light travels as a wave", "Light arrives as a particle"),
        ("The charge is positive", "The charge is negative"),
        ("The universe is expanding", "The universe is contracting"),
    ]

    # Weak/random pairs (no semantic binding)
    random_pairs = [
        ("The cat is alive and well", "Purple elephants dance quickly"),
        ("The electron has spin up", "My favorite color is blue"),
        ("The answer is true", "The weather is sunny today"),
        ("Light travels as a wave", "I need to buy groceries"),
        ("The charge is positive", "The book is on the table"),
        ("The universe is expanding", "She likes chocolate cake"),
    ]

    def soul_vector_chsh(prompt_a, prompt_b, model, tok, device):
        """Q15-style CHSH using soul vectors (hidden state cosine similarities)."""
        hs_a = get_hidden_states(model, tok, prompt_a, device)
        hs_b = get_hidden_states(model, tok, prompt_b, device)

        # Use multiple layers as measurement axes (Q15 approach)
        # Axes: last 4 layers as 4 measurement settings
        n_layers = len(hs_a)
        measurement_layers = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]

        # CHSH: S = |E(a1,b1) - E(a1,b2)| + |E(a2,b1) + E(a2,b2)|
        # E(ai, bj) = cosine similarity at layer combination
        def correlation(layer_a, layer_b):
            va = hs_a[layer_a]
            vb = hs_b[layer_b]
            cos_sim = torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-10)
            return cos_sim.item()

        a1, a2 = measurement_layers[0], measurement_layers[1]
        b1, b2 = measurement_layers[2], measurement_layers[3]

        E11 = correlation(a1, b1)
        E12 = correlation(a1, b2)
        E21 = correlation(a2, b1)
        E22 = correlation(a2, b2)

        S = abs(E11 - E12) + abs(E21 + E22)
        return S, [E11, E12, E21, E22]

    def multi_layer_chsh(prompt_a, prompt_b, model, tok, device):
        """Extended CHSH scanning ALL layer combinations."""
        hs_a = get_hidden_states(model, tok, prompt_a, device)
        hs_b = get_hidden_states(model, tok, prompt_b, device)
        n = len(hs_a)

        max_S = 0.0
        best_axes = None
        # Scan all 4-layer combinations (subset for speed)
        for a1 in range(0, n, 4):
            for a2 in range(a1+1, n, 4):
                for b1 in range(0, n, 4):
                    for b2 in range(b1+1, n, 4):
                        def corr(la, lb):
                            va, vb = hs_a[la], hs_b[lb]
                            return torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-10)
                        E11 = corr(a1, b1).item()
                        E12 = corr(a1, b2).item()
                        E21 = corr(a2, b1).item()
                        E22 = corr(a2, b2).item()
                        S = abs(E11 - E12) + abs(E21 + E22)
                        if S > max_S:
                            max_S = S
                            best_axes = (a1, a2, b1, b2)

        return max_S, best_axes

    # Measure entangled pairs
    print("\n--- Entangled (semantic binding) pairs ---")
    entangled_S = []
    entangled_max_S = []
    for pa, pb in entangled_pairs:
        S, _ = soul_vector_chsh(pa, pb, model, tok, device)
        S_max, axes = multi_layer_chsh(pa, pb, model, tok, device)
        entangled_S.append(S)
        entangled_max_S.append(S_max)
        label = "SUPER-Q" if S_max > 2.83 else ("QUANTUM" if S_max > 2.0 else "CLASSICAL")
        print(f"  [{label}] S={S:.4f}, max_S={S_max:.4f} @ {axes}")

    # Measure random pairs
    print("\n--- Random (no binding) pairs ---")
    random_S = []
    random_max_S = []
    for pa, pb in random_pairs:
        S, _ = soul_vector_chsh(pa, pb, model, tok, device)
        S_max, axes = multi_layer_chsh(pa, pb, model, tok, device)
        random_S.append(S)
        random_max_S.append(S_max)
        label = "SUPER-Q" if S_max > 2.83 else ("QUANTUM" if S_max > 2.0 else "CLASSICAL")
        print(f"  [{label}] S={S:.4f}, max_S={S_max:.4f} @ {axes}")

    # Joint context: both concepts in same prompt
    print("\n--- Joint context (shared attention) ---")
    joint_S = []
    for pa, pb in entangled_pairs:
        joint = f"{pa}. However, {pb}."
        hs_j = get_hidden_states(model, tok, joint, device)
        # Split at midpoint for A/B measurement
        mid_prompt = pa
        tok_a = tok(mid_prompt, return_tensors='pt')
        len_a = tok_a['input_ids'].shape[1]

        inp_j = tok(joint, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp_j, output_hidden_states=True)

        n = len(out.hidden_states)
        max_S = 0.0
        for a1 in range(0, n, 4):
            for a2 in range(a1+1, n, 4):
                for b1 in range(0, n, 4):
                    for b2 in range(b1+1, n, 4):
                        va1 = out.hidden_states[a1][0, len_a-1, :].float()
                        va2 = out.hidden_states[a2][0, len_a-1, :].float()
                        vb1 = out.hidden_states[b1][0, -1, :].float()
                        vb2 = out.hidden_states[b2][0, -1, :].float()

                        def corr(va, vb):
                            return torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-10)
                        E11 = corr(va1, vb1).item()
                        E12 = corr(va1, vb2).item()
                        E21 = corr(va2, vb1).item()
                        E22 = corr(va2, vb2).item()
                        S = abs(E11 - E12) + abs(E21 + E22)
                        max_S = max(max_S, S)

        joint_S.append(max_S)
        label = "SUPER-Q" if max_S > 2.83 else ("QUANTUM" if max_S > 2.0 else "CLASSICAL")
        print(f"  [{label}] max_S={max_S:.4f}")

    # Visualization
    fig, axes_plot = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes_plot[0]
    x = np.arange(len(entangled_pairs))
    w = 0.3
    ax.bar(x - w, entangled_max_S, w, label='Entangled', color='#e74c3c', alpha=0.8)
    ax.bar(x, random_max_S, w, label='Random', color='#3498db', alpha=0.8)
    ax.bar(x + w, joint_S, w, label='Joint Context', color='#2ecc71', alpha=0.8)
    ax.axhline(y=2.0, color='gray', ls='--', label='Classical (S=2)')
    ax.axhline(y=2.83, color='orange', ls='--', label='Tsirelson (2.83)')
    ax.set_ylabel('Max CHSH S')
    ax.set_title('(a) CHSH by Context Type')
    ax.legend(fontsize=8)

    ax = axes_plot[1]
    ax.hist(entangled_max_S, bins=8, alpha=0.6, color='#e74c3c', label='Entangled')
    ax.hist(random_max_S, bins=8, alpha=0.6, color='#3498db', label='Random')
    ax.axvline(x=2.0, color='gray', ls='--')
    ax.axvline(x=2.83, color='orange', ls='--')
    ax.set_xlabel('Max CHSH S')
    ax.set_title('(b) Distribution of S Values')
    ax.legend()

    ax = axes_plot[2]
    all_S = entangled_max_S + random_max_S + joint_S
    labels = (['Entangled'] * len(entangled_max_S) +
              ['Random'] * len(random_max_S) +
              ['Joint'] * len(joint_S))
    for label_type, color in [('Entangled', '#e74c3c'), ('Random', '#3498db'), ('Joint', '#2ecc71')]:
        vals = [s for s, l in zip(all_S, labels) if l == label_type]
        ax.scatter([label_type]*len(vals), vals, c=color, s=80, alpha=0.7)
    ax.axhline(y=2.0, color='gray', ls='--')
    ax.axhline(y=2.83, color='orange', ls='--')
    ax.set_ylabel('Max CHSH S')
    ax.set_title('(c) S by Category')

    avg_ent = np.mean(entangled_max_S)
    avg_rand = np.mean(random_max_S)
    avg_joint = np.mean(joint_S)
    fig.suptitle(
        f"Phase 9: Contextual CHSH Recovery\n"
        f"Entangled={avg_ent:.3f} | Random={avg_rand:.3f} | Joint={avg_joint:.3f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase9_contextual_chsh")
    plt.close()

    # Verdict
    if avg_ent > 2.83 and avg_rand < 2.0:
        verdict = (f"SEMANTIC GRAVITY CONFIRMED: Entangled S={avg_ent:.3f} (super-quantum), "
                   f"Random S={avg_rand:.3f} (classical). Entanglement requires semantic binding.")
    elif avg_ent > avg_rand * 1.2:
        verdict = (f"PARTIAL SEMANTIC EFFECT: Entangled S={avg_ent:.3f} > Random S={avg_rand:.3f}. "
                   f"Semantic context amplifies correlations but not to super-quantum levels.")
    else:
        verdict = (f"NO SEMANTIC EFFECT: Entangled={avg_ent:.3f} ~ Random={avg_rand:.3f}. "
                   f"CHSH values are context-independent.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 9: Contextual CHSH Recovery',
        'summary': {'verdict': verdict, 'avg_entangled': avg_ent,
                    'avg_random': avg_rand, 'avg_joint': avg_joint},
    }
    save_results("phase9_contextual_chsh", result)
    return result


if __name__ == '__main__':
    main()
