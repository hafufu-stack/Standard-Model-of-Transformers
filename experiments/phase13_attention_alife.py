# -*- coding: utf-8 -*-
"""
Phase 13: Attention-Graph ALife
================================
Phase 5 failed because KV Cache has no spatial locality (1.02).
Deep Think insight: the true "space" is not positional but
the Attention Weight matrix itself. Tokens that attend strongly
to each other are "neighbors" in the semantic manifold.

Use Attention Weights as the adjacency matrix for NCA.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 13: Attention-Graph ALife (Non-Euclidean CA)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    # Generate attention maps for structured prompts
    prompts = [
        "The cat sat on the mat and the dog ate the food",
        "One two three four five six seven eight nine ten",
        "The quick brown fox jumps over the lazy dog today",
        "Positive negative up down left right hot cold bright dark",
    ]

    all_results = {}

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {prompt_idx+1}: '{prompt[:40]}...' ---")
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_attentions=True)

        # Use attention from a middle layer, head 0
        mid_layer = n_layers // 2
        attn = out.attentions[mid_layer][0, 0, :, :].float().cpu().numpy()  # (seq, seq)

        # The attention matrix IS the adjacency (neighborhood) matrix
        # Run NCA: cell state = hidden state magnitude at each position
        hidden_states = out.hidden_states if hasattr(out, 'hidden_states') else None

        # Initialize cell states from token embeddings magnitude
        init_states = []
        for pos in range(seq_len):
            token_id = inp['input_ids'][0, pos].item()
            # Simple rule: odd token_id = 1, even = 0 (binary automaton)
            init_states.append(1 if token_id % 2 == 1 else 0)

        init_states = np.array(init_states, dtype=float)

        # Evolve NCA using attention as connectivity
        n_generations = 20
        threshold = 0.5
        evolution = [init_states.copy()]

        current = init_states.copy()
        for gen in range(n_generations):
            new_state = np.zeros_like(current)
            for i in range(seq_len):
                # Weighted neighbor sum using attention weights
                neighbor_sum = np.dot(attn[i, :], current)
                # Game of Life-like rule on weighted sum
                if current[i] == 1:
                    new_state[i] = 1 if 0.3 < neighbor_sum < 0.8 else 0
                else:
                    new_state[i] = 1 if 0.4 < neighbor_sum < 0.6 else 0
            current = new_state
            evolution.append(current.copy())

        evolution = np.array(evolution)

        # Metrics
        alive_history = [e.sum() for e in evolution]
        entropy_history = []
        for e in evolution:
            p = e.mean()
            if 0 < p < 1:
                ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
            else:
                ent = 0.0
            entropy_history.append(ent)

        # Pattern diversity (unique states)
        unique_patterns = len(set(tuple(e.astype(int)) for e in evolution))
        survived = unique_patterns > 2 and alive_history[-1] > 0

        # Attention locality: are strongly attending tokens actually "close"?
        locality_score = 0.0
        count = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    locality_score += attn[i, j] * (1.0 / (abs(i - j) + 1))
                    count += 1
        locality_score /= (count + 1e-10)

        status = "ALIVE" if survived else "DEAD"
        print(f"  Generations: {len(evolution)}, Unique patterns: {unique_patterns}, "
              f"Status: {status}")
        print(f"  Alive cells: {alive_history[0]:.0f} -> {alive_history[-1]:.0f}")
        print(f"  Attention locality score: {locality_score:.4f}")

        all_results[f"prompt_{prompt_idx}"] = {
            'prompt': prompt, 'evolution': evolution.tolist(),
            'alive_history': alive_history, 'entropy_history': entropy_history,
            'unique_patterns': unique_patterns, 'survived': survived,
            'attn_matrix': attn.tolist(), 'locality_score': locality_score,
        }

    # Multi-head comparison: which head creates best ALife?
    print("\n--- Multi-head ALife scan (best prompt) ---")
    best_prompt = prompts[0]
    inp = tok(best_prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    head_scores = []
    for layer in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        for head in range(min(n_heads, 4)):
            attn_h = out.attentions[layer][0, head, :, :].float().cpu().numpy()
            init = np.array([1 if i % 2 == 0 else 0 for i in range(seq_len)], dtype=float)
            current = init.copy()
            states = [current.copy()]
            for _ in range(15):
                new = np.zeros_like(current)
                for i in range(seq_len):
                    ns = np.dot(attn_h[i], current)
                    if current[i] == 1:
                        new[i] = 1 if 0.3 < ns < 0.8 else 0
                    else:
                        new[i] = 1 if 0.4 < ns < 0.6 else 0
                current = new
                states.append(current.copy())
            unique = len(set(tuple(s.astype(int)) for s in states))
            head_scores.append({
                'layer': layer, 'head': head, 'unique': unique,
                'final_alive': int(current.sum()),
            })

    best_head = max(head_scores, key=lambda x: x['unique'])
    print(f"  Best head: L{best_head['layer']}H{best_head['head']} "
          f"({best_head['unique']} unique, {best_head['final_alive']} alive)")

    # Visualization
    n_prompts = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (key, data) in enumerate(all_results.items()):
        if idx >= 4:
            break
        row, col = idx // 3, idx % 3
        if idx == 3:
            row, col = 1, 0
        ax = axes[row][col]
        evo = np.array(data['evolution'])
        ax.imshow(evo, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Generation')
        status = "ALIVE" if data['survived'] else "DEAD"
        ax.set_title(f'{key} [{status}]\n{data["prompt"][:30]}...', fontsize=9)

    # (e) Alive cell history
    ax = axes[1][1]
    for key, data in all_results.items():
        ax.plot(data['alive_history'], 'o-', ms=3, label=key[:10])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Alive Cells')
    ax.set_title('(e) Population Dynamics')
    ax.legend(fontsize=8)

    # (f) Head diversity
    ax = axes[1][2]
    layers_h = [h['layer'] for h in head_scores]
    uniques = [h['unique'] for h in head_scores]
    ax.scatter(layers_h, uniques, s=60, c='#e74c3c', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Unique Patterns')
    ax.set_title('(f) ALife by Attention Head')

    n_survived = sum(1 for d in all_results.values() if d['survived'])
    fig.suptitle(
        f"Phase 13: Attention-Graph ALife\n"
        f"{n_survived}/{len(all_results)} survived | "
        f"Best head: L{best_head['layer']}H{best_head['head']}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase13_attention_alife")
    plt.close()

    if n_survived >= len(all_results) // 2:
        verdict = (f"SEMANTIC ALIFE CONFIRMED: {n_survived}/{len(all_results)} survived in "
                   f"attention-graph space. Attention IS the topology of life.")
    else:
        verdict = (f"PARTIAL ALIFE: {n_survived}/{len(all_results)} survived. "
                   f"Best head: L{best_head['layer']}H{best_head['head']} "
                   f"({best_head['unique']} unique patterns).")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 13: Attention-Graph ALife',
        'summary': {'verdict': verdict, 'survived': n_survived, 'total': len(all_results),
                    'best_head': best_head},
    }
    save_results("phase13_attention_alife", result)
    return result


if __name__ == '__main__':
    main()
