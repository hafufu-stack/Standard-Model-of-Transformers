# -*- coding: utf-8 -*-
"""
Phase 29: Thermodynamic Firewall (Opus Original)
===================================================
Monitor PR*T during autoregressive generation.
When the model hallucinates, PR*T should deviate from the conserved value.
This is the prototype of a zero-shot hallucination detector.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 29: Thermodynamic Firewall")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Compare factual vs hallucinatory generation
    test_cases = [
        ("factual", "The chemical symbol for gold is Au. The chemical symbol for silver is"),
        ("factual", "In mathematics, pi is approximately 3.14159. The square root of 4 is"),
        ("ambiguous", "The secret meaning of life according to the ancient texts is that"),
        ("ambiguous", "In the year 2050, historians discovered that the real cause of"),
    ]

    all_traces = []
    for label, prompt in test_cases:
        print(f"\n--- {label}: '{prompt[:50]}...' ---")
        inp = tok(prompt, return_tensors='pt').to(device)
        input_ids = inp['input_ids']

        trace = {'label': label, 'prompt': prompt[:50], 'steps': []}
        past_kv = None
        generated_tokens = []

        for t in range(100):
            if past_kv is None:
                curr_input = input_ids
            else:
                curr_input = next_token_id

            with torch.no_grad():
                out = model(input_ids=curr_input, past_key_values=past_kv,
                           use_cache=True, output_hidden_states=True)

            past_kv = out.past_key_values
            h_last = out.hidden_states[-1][0, -1, :].float()
            U = h_last.norm().item()

            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            PR = 1.0 / (probs ** 2).sum().item()
            PRT = PR * T
            top_p = probs.max().item()

            trace['steps'].append({'t': t, 'U': U, 'T': T, 'PR': PR, 'PRT': PRT, 'top_p': top_p})

            probs_safe = probs.clamp(min=1e-10)
            probs_safe = probs_safe / probs_safe.sum()
            next_token_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)  # greedy
            generated_tokens.append(tok.decode(next_token_id[0, 0].item()))

            if t < 5 or t % 20 == 0:
                print(f"  t={t}: PRT={PRT:.1f}, T={T:.2f}, top_p={top_p:.3f}, "
                      f"tok='{generated_tokens[-1]}'")

        trace['generated'] = ''.join(generated_tokens[:30])
        all_traces.append(trace)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'factual': '#2ecc71', 'ambiguous': '#e74c3c'}

    for trace in all_traces:
        label = trace['label']
        steps = [s['t'] for s in trace['steps']]

        axes[0][0].plot(steps, [s['PRT'] for s in trace['steps']], '-',
                        color=colors[label], lw=1.5, alpha=0.7,
                        label=f"{label}: {trace['prompt'][:25]}")
        axes[0][1].plot(steps, [s['T'] for s in trace['steps']], '-',
                        color=colors[label], lw=1.5, alpha=0.7)
        axes[1][0].plot(steps, [s['top_p'] for s in trace['steps']], '-',
                        color=colors[label], lw=1.5, alpha=0.7)
        axes[1][1].plot(steps, [s['U'] for s in trace['steps']], '-',
                        color=colors[label], lw=1.5, alpha=0.7)

    axes[0][0].set_ylabel('PR x T'); axes[0][0].set_title('(a) PR*T (Firewall Signal)')
    axes[0][0].legend(fontsize=6)
    axes[0][1].set_ylabel('T (entropy)'); axes[0][1].set_title('(b) Temperature')
    axes[1][0].set_ylabel('Top-1 Prob'); axes[1][0].set_title('(c) Confidence')
    axes[1][1].set_ylabel('U (L2 norm)'); axes[1][1].set_title('(d) Internal Energy')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Generation Step')

    # Compute firewall metrics
    factual_prt = [np.std([s['PRT'] for s in t['steps']]) for t in all_traces if t['label'] == 'factual']
    ambig_prt = [np.std([s['PRT'] for s in t['steps']]) for t in all_traces if t['label'] == 'ambiguous']
    f_std = np.mean(factual_prt) if factual_prt else 0
    a_std = np.mean(ambig_prt) if ambig_prt else 0

    fig.suptitle(
        f"Phase 29: Thermodynamic Firewall\n"
        f"PRT variance: factual={f_std:.1f}, ambiguous={a_std:.1f} "
        f"(ratio={a_std/(f_std+1e-10):.2f}x)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase29_firewall")
    plt.close()

    ratio = a_std / (f_std + 1e-10)
    if ratio > 1.5:
        verdict = (f"FIREWALL VIABLE: Ambiguous PRT variance is {ratio:.1f}x larger. "
                   f"PR*T monitoring can detect hallucination!")
    else:
        verdict = (f"FIREWALL WEAK: Variance ratio = {ratio:.1f}x. "
                   f"PR*T alone is insufficient for hallucination detection.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase29_firewall", {
        'name': 'Phase 29: Thermodynamic Firewall',
        'summary': {'verdict': verdict, 'factual_std': f_std, 'ambig_std': a_std, 'ratio': ratio},
    })


if __name__ == '__main__':
    main()
