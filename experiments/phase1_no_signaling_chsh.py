# -*- coding: utf-8 -*-
"""
Phase 1: Strict No-Signaling CHSH Test
=============================================
The most critical experiment. S-Qubit measures S=3.41 for CHSH,
but this exceeds the Tsirelson bound (2*sqrt(2)=2.83). Deep Think
argues this is because Attention is a global communication channel
that violates the no-signaling condition.

This experiment proves the causal link by applying a "light-cone mask"
that prevents two tokens from communicating via Attention, then
re-measuring CHSH. If S drops to <=2.0, Attention IS the channel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

def main():
    print("=" * 70)
    print("Phase 1: Strict No-Signaling CHSH Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Model: {model.config._name_or_path}, {n_layers} layers, d={hidden_size}")

    # ================================================================
    # CHSH measurement protocol (from S-Qubit Q15)
    # ================================================================
    # Concept pairs for "entangled" measurements
    concept_pairs = [
        ("The cat is alive", "The cat is dead"),
        ("spin up electron", "spin down electron"),
        ("true proposition", "false proposition"),
        ("hot temperature", "cold temperature"),
        ("light wave", "dark shadow"),
        ("positive charge", "negative charge"),
    ]

    # Measurement axes (prompt suffixes that define measurement direction)
    axes_A = [" is", " means"]
    axes_B = [" implies", " suggests"]

    def extract_hidden_pair(prompt_a, prompt_b, mask_attention=False):
        """Extract hidden states for two prompts.
        If mask_attention=True, we prevent cross-token attention
        by using separate forward passes (strict no-signaling).
        """
        if mask_attention:
            # STRICT NO-SIGNALING: Each token processed independently
            # No shared KV cache, no cross-attention possible
            inp_a = tok(prompt_a, return_tensors='pt').to(device)
            inp_b = tok(prompt_b, return_tensors='pt').to(device)
            with torch.no_grad():
                out_a = model(**inp_a, output_hidden_states=True)
                out_b = model(**inp_b, output_hidden_states=True)
            h_a = out_a.hidden_states[-1][0, -1, :].cpu().float()
            h_b = out_b.hidden_states[-1][0, -1, :].cpu().float()
        else:
            # STANDARD: Both concepts in same context (Attention links them)
            combined = f"{prompt_a}. {prompt_b}"
            inp = tok(combined, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            # Token positions: find boundary
            tok_a = tok(prompt_a, return_tensors='pt')
            len_a = tok_a['input_ids'].shape[1]
            h_a = out.hidden_states[-1][0, len_a - 1, :].cpu().float()
            h_b = out.hidden_states[-1][0, -1, :].cpu().float()
        return h_a, h_b

    def compute_correlation(h_a, h_b, axis_a, axis_b):
        """Compute CHSH-style correlation E(a,b) using cosine similarity
        projected onto measurement axes."""
        # Project onto axis directions
        inp_ax_a = tok(axis_a, return_tensors='pt').to(device)
        inp_ax_b = tok(axis_b, return_tensors='pt').to(device)
        with torch.no_grad():
            ax_a_out = model(**inp_ax_a, output_hidden_states=True)
            ax_b_out = model(**inp_ax_b, output_hidden_states=True)
        ax_a_vec = ax_a_out.hidden_states[-1][0, -1, :].cpu().float()
        ax_b_vec = ax_b_out.hidden_states[-1][0, -1, :].cpu().float()

        # Project hidden states onto axes
        proj_a = torch.dot(h_a, ax_a_vec) / (h_a.norm() * ax_a_vec.norm() + 1e-10)
        proj_b = torch.dot(h_b, ax_b_vec) / (h_b.norm() * ax_b_vec.norm() + 1e-10)

        # Correlation = product of projections (CHSH-style)
        return (proj_a * proj_b).item()

    def compute_chsh(concept_a, concept_b, mask_attention=False):
        """Compute CHSH parameter S = |E(a1,b1) - E(a1,b2)| + |E(a2,b1) + E(a2,b2)|"""
        a1, a2 = axes_A
        b1, b2 = axes_B

        h_a, h_b = extract_hidden_pair(
            concept_a + a1, concept_b + b1, mask_attention=mask_attention
        )
        E11 = compute_correlation(h_a, h_b, a1, b1)

        h_a, h_b = extract_hidden_pair(
            concept_a + a1, concept_b + b2, mask_attention=mask_attention
        )
        E12 = compute_correlation(h_a, h_b, a1, b2)

        h_a, h_b = extract_hidden_pair(
            concept_a + a2, concept_b + b1, mask_attention=mask_attention
        )
        E21 = compute_correlation(h_a, h_b, a2, b1)

        h_a, h_b = extract_hidden_pair(
            concept_a + a2, concept_b + b2, mask_attention=mask_attention
        )
        E22 = compute_correlation(h_a, h_b, a2, b2)

        S = abs(E11 - E12) + abs(E21 + E22)
        return S, [E11, E12, E21, E22]

    # ================================================================
    # Experiment: Compare masked vs unmasked CHSH
    # ================================================================
    results_unmasked = []
    results_masked = []

    for i, (ca, cb) in enumerate(concept_pairs):
        print(f"\n--- Pair {i+1}/{len(concept_pairs)}: '{ca}' vs '{cb}' ---")

        # Standard CHSH (Attention links concepts)
        S_std, E_std = compute_chsh(ca, cb, mask_attention=False)
        results_unmasked.append({'pair': f"{ca} / {cb}", 'S': S_std, 'E': E_std})
        print(f"  Standard (Attention ON):  S = {S_std:.4f}")

        # No-signaling CHSH (Attention blocked)
        S_mask, E_mask = compute_chsh(ca, cb, mask_attention=True)
        results_masked.append({'pair': f"{ca} / {cb}", 'S': S_mask, 'E': E_mask})
        print(f"  No-Signal (Attention OFF): S = {S_mask:.4f}")

        drop = S_std - S_mask
        print(f"  Drop: {drop:+.4f} ({'COLLAPSED' if S_mask <= 2.0 else 'STILL SUPER-QUANTUM'})")

    # ================================================================
    # Additional: Attention Mask Hook (partial blocking)
    # Block attention between specific token positions within same context
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 1b: Gradual Light-Cone Restriction")
    print("=" * 70)

    ca, cb = concept_pairs[0]
    mask_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    gradual_results = []

    for frac in mask_fractions:
        if frac == 0.0:
            S_val, _ = compute_chsh(ca, cb, mask_attention=False)
        elif frac == 1.0:
            S_val, _ = compute_chsh(ca, cb, mask_attention=True)
        else:
            # Partial masking: block attention in frac of layers
            n_block = int(n_layers * frac)
            block_layers = list(range(n_layers - n_block, n_layers))
            handles = []

            def make_mask_hook(layer_idx):
                def hook(module, args, kwargs):
                    # Modify attention_mask to block cross-token attention
                    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                        mask = kwargs['attention_mask'].clone()
                        seq_len = mask.shape[-1]
                        mid = seq_len // 2
                        # Block first half from attending to second half and vice versa
                        mask[:, :, :mid, mid:] = torch.finfo(mask.dtype).min
                        mask[:, :, mid:, :mid] = torch.finfo(mask.dtype).min
                        kwargs['attention_mask'] = mask
                    return args, kwargs
                return hook

            for li in block_layers:
                h = model.model.layers[li].self_attn.register_forward_pre_hook(
                    make_mask_hook(li), with_kwargs=True
                )
                handles.append(h)

            S_val, _ = compute_chsh(ca, cb, mask_attention=False)

            for h in handles:
                h.remove()

        gradual_results.append({'mask_fraction': frac, 'S': S_val})
        label = "CLASSICAL" if S_val <= 2.0 else ("QUANTUM" if S_val <= 2.83 else "SUPER-Q")
        print(f"  Mask fraction {frac:.0%}: S = {S_val:.4f} [{label}]")

    # ================================================================
    # Analysis & Visualization
    # ================================================================
    S_unmasked = [r['S'] for r in results_unmasked]
    S_masked = [r['S'] for r in results_masked]
    avg_unmasked = np.mean(S_unmasked)
    avg_masked = np.mean(S_masked)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Bar comparison
    ax = axes[0]
    x = np.arange(len(concept_pairs))
    w = 0.35
    ax.bar(x - w/2, S_unmasked, w, label='Attention ON', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, S_masked, w, label='No-Signaling', color='#3498db', alpha=0.8)
    ax.axhline(y=2.0, color='gray', ls='--', lw=1.5, label='Classical limit (S=2)')
    ax.axhline(y=2.83, color='orange', ls='--', lw=1.5, label='Tsirelson bound (2.83)')
    ax.set_xlabel('Concept Pair')
    ax.set_ylabel('CHSH Parameter S')
    ax.set_title('(a) CHSH: Attention ON vs No-Signaling')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(concept_pairs))], fontsize=9)
    ax.legend(fontsize=8)

    # (b) S-value drop
    ax = axes[1]
    drops = [u - m for u, m in zip(S_unmasked, S_masked)]
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in drops]
    ax.bar(x, drops, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Concept Pair')
    ax.set_ylabel('S Drop (ON - OFF)')
    ax.set_title('(b) CHSH Collapse When Attention Blocked')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(concept_pairs))], fontsize=9)

    # (c) Gradual light-cone
    ax = axes[2]
    fracs = [r['mask_fraction'] for r in gradual_results]
    s_vals = [r['S'] for r in gradual_results]
    ax.plot(fracs, s_vals, 'o-', color='#9b59b6', lw=2, ms=8)
    ax.axhline(y=2.0, color='gray', ls='--', lw=1.5, label='Classical (S=2)')
    ax.axhline(y=2.83, color='orange', ls='--', lw=1.5, label='Tsirelson (2.83)')
    ax.set_xlabel('Fraction of Layers Masked')
    ax.set_ylabel('CHSH Parameter S')
    ax.set_title('(c) Gradual Light-Cone Restriction')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)

    fig.suptitle(
        f"Phase 1: No-Signaling CHSH Test\n"
        f"Attention ON: avg S={avg_unmasked:.3f} | No-Signal: avg S={avg_masked:.3f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase1_no_signaling_chsh")
    plt.close()

    # ================================================================
    # Verdict
    # ================================================================
    collapsed = sum(1 for s in S_masked if s <= 2.0)
    total = len(S_masked)

    if avg_masked <= 2.0:
        verdict = (f"CAUSAL PROOF: S collapses from {avg_unmasked:.3f} -> {avg_masked:.3f} "
                   f"({collapsed}/{total} below classical limit). "
                   f"Attention IS the entanglement channel.")
    elif avg_masked <= 2.83:
        verdict = (f"PARTIAL COLLAPSE: S drops from {avg_unmasked:.3f} -> {avg_masked:.3f}. "
                   f"Attention contributes but geometry also plays a role.")
    else:
        verdict = (f"NO COLLAPSE: S remains {avg_masked:.3f} even without Attention. "
                   f"Super-quantum correlations are geometric, not communication-based.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 1: Strict No-Signaling CHSH Test',
        'summary': {
            'verdict': verdict,
            'S_attention_on': avg_unmasked,
            'S_no_signaling': avg_masked,
            'collapsed_count': f"{collapsed}/{total}",
            'drop': avg_unmasked - avg_masked,
        },
        'unmasked': results_unmasked,
        'masked': results_masked,
        'gradual': gradual_results,
    }
    save_results("phase1_no_signaling_chsh", result)
    return result


if __name__ == '__main__':
    main()
