# -*- coding: utf-8 -*-
"""
Phase 2: KV Cache DMA Injection
================================
Test whether NeuOS/Aletheia's "register layers" can be directly
programmed via DMA-style hidden state injection.

Inject specific hidden state vectors at:
- L0 (OPCODE register)
- L2 (B-register)
- L11 (A-register)
- L17 (Carry/output register)

Without any gradient updates, force the LLM to execute arithmetic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, get_hidden_states, save_results, save_figure

def main():
    print("=" * 70)
    print("Phase 2: KV Cache DMA Injection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    # ================================================================
    # Step 1: Learn the "register encoding" from known arithmetic
    # ================================================================
    print("\n--- Step 1: Learning register encodings ---")

    # Arithmetic templates
    arithmetic_pairs = [
        ("2 + 3 =", "5"),
        ("7 - 4 =", "3"),
        ("6 * 2 =", "12"),
        ("9 + 1 =", "10"),
        ("5 + 5 =", "10"),
        ("8 - 3 =", "5"),
        ("3 * 3 =", "9"),
        ("4 + 7 =", "11"),
    ]

    # Extract hidden states for each arithmetic prompt
    register_layers = [0, 2, 11, 17]  # OPCODE, B, A, Carry
    register_names = ['L0 (OPCODE)', 'L2 (B-reg)', 'L11 (A-reg)', 'L17 (Carry)']

    encodings = {}
    for prompt, answer in arithmetic_pairs:
        hs = get_hidden_states(model, tok, prompt, device=device)
        encodings[prompt] = {
            'hidden_states': {f'L{l}': hs[l].numpy().tolist()[:20] for l in register_layers},
            'answer': answer,
        }
        print(f"  {prompt} {answer} -> L0 norm={hs[0].norm():.2f}, L17 norm={hs[17].norm():.2f}")

    # ================================================================
    # Step 2: DMA injection - swap register states between operations
    # ================================================================
    print("\n--- Step 2: DMA Injection Test ---")

    test_cases = [
        # Inject "addition context" from "2+3=5" into "7-4=" to see if output changes
        ("2 + 3 =", "7 - 4 =", "Inject ADD into SUB"),
        ("6 * 2 =", "9 + 1 =", "Inject MUL into ADD"),
        ("5 + 5 =", "8 - 3 =", "Inject ADD into SUB"),
    ]

    injection_results = []

    for source_prompt, target_prompt, description in test_cases:
        print(f"\n  {description}: '{source_prompt}' -> '{target_prompt}'")

        # Get source hidden states (the "donor")
        source_hs = get_hidden_states(model, tok, source_prompt, device=device)

        # Get baseline output for target
        baseline_logits = model(**tok(target_prompt, return_tensors='pt').to(device)).logits[0, -1, :]
        baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
        baseline_top5 = torch.topk(baseline_probs, 5)
        baseline_tokens = [tok.decode([idx]) for idx in baseline_top5.indices.tolist()]
        print(f"    Baseline top-5: {baseline_tokens}")

        # Inject source hidden states at each register layer
        for li, lname in zip(register_layers, register_names):
            inject_vec = source_hs[li].to(device).to(model.dtype)

            def make_inject_hook(vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        if h.dim() == 3:
                            h[0, -1, :] = vec
                        else:
                            h[-1, :] = vec
                        return (h,) + output[1:]
                    return output
                return hook

            handle = model.model.layers[li].register_forward_hook(make_inject_hook(inject_vec))

            injected_logits = model(**tok(target_prompt, return_tensors='pt').to(device)).logits[0, -1, :]
            handle.remove()

            injected_probs = torch.softmax(injected_logits.float(), dim=-1)
            injected_top5 = torch.topk(injected_probs, 5)
            injected_tokens = [tok.decode([idx]) for idx in injected_top5.indices.tolist()]

            # KL divergence between baseline and injected
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(injected_logits.float(), dim=-1),
                torch.softmax(baseline_logits.float(), dim=-1),
                reduction='sum'
            ).item()

            # Did the output change toward the source answer?
            source_answer = [p[1] for p in arithmetic_pairs if p[0] == source_prompt][0]
            answer_tok_id = tok.encode(source_answer)[-1]
            source_prob = injected_probs[answer_tok_id].item()

            print(f"    Inject {lname}: top5={injected_tokens}, KL={kl:.3f}, P(source_ans)={source_prob:.4f}")

            injection_results.append({
                'source': source_prompt,
                'target': target_prompt,
                'layer': lname,
                'layer_idx': li,
                'kl_divergence': kl,
                'source_answer_prob': source_prob,
                'injected_top5': injected_tokens,
                'baseline_top5': baseline_tokens,
            })

    # ================================================================
    # Step 3: Identity of register layers
    # ================================================================
    print("\n--- Step 3: Layer Impact Analysis ---")
    target = "4 + 7 ="
    impacts = []

    for li in range(n_layers):
        # Inject random noise at layer li
        noise = torch.randn(model.config.hidden_size).to(device).to(model.dtype)

        def make_noise_hook(vec):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    if h.dim() == 3:
                        h[0, -1, :] = vec
                    return (h,) + output[1:]
                return output
            return hook

        handle = model.model.layers[li].register_forward_hook(make_noise_hook(noise))
        noisy_logits = model(**tok(target, return_tensors='pt').to(device)).logits[0, -1, :]
        handle.remove()

        kl = torch.nn.functional.kl_div(
            torch.log_softmax(noisy_logits.float(), dim=-1),
            torch.softmax(baseline_logits.float(), dim=-1),
            reduction='sum'
        ).item()
        impacts.append(kl)
        if li in register_layers:
            print(f"  L{li} (REGISTER): KL={kl:.3f}")

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) KL divergence by injection layer
    ax = axes[0]
    kl_by_layer = {}
    for r in injection_results:
        li = r['layer_idx']
        if li not in kl_by_layer:
            kl_by_layer[li] = []
        kl_by_layer[li].append(r['kl_divergence'])

    layer_means = {li: np.mean(kls) for li, kls in kl_by_layer.items()}
    ax.bar(range(len(register_layers)),
           [layer_means.get(l, 0) for l in register_layers],
           color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
           alpha=0.8)
    ax.set_xticks(range(len(register_layers)))
    ax.set_xticklabels(register_names, fontsize=9)
    ax.set_ylabel('KL Divergence (injection impact)')
    ax.set_title('(a) DMA Injection Impact by Register')

    # (b) Source answer probability after injection
    ax = axes[1]
    for li, lname in zip(register_layers, register_names):
        probs = [r['source_answer_prob'] for r in injection_results if r['layer_idx'] == li]
        ax.bar(register_layers.index(li), np.mean(probs), color='#9b59b6', alpha=0.7)
    ax.set_xticks(range(len(register_layers)))
    ax.set_xticklabels(register_names, fontsize=9)
    ax.set_ylabel('P(source answer) after injection')
    ax.set_title('(b) Answer Steering by DMA')

    # (c) Full layer impact profile
    ax = axes[2]
    ax.plot(range(n_layers), impacts, 'o-', color='#34495e', ms=4, lw=1.5)
    for li, lname in zip(register_layers, register_names):
        ax.axvline(x=li, color='red', ls='--', alpha=0.5)
        ax.annotate(lname.split()[0], (li, impacts[li]), fontsize=7,
                   ha='center', va='bottom', color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('KL Divergence (noise injection)')
    ax.set_title('(c) Layer Criticality Profile')

    fig.suptitle("Phase 2: KV Cache DMA Injection\n"
                 "Can we program the LLM by injecting hidden states at register layers?",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "phase2_dma_injection")
    plt.close()

    # ================================================================
    # Verdict
    # ================================================================
    max_impact_layer = register_layers[np.argmax([layer_means.get(l, 0) for l in register_layers])]
    avg_steering = np.mean([r['source_answer_prob'] for r in injection_results])

    verdict = (f"DMA INJECTION: Most impactful register = L{max_impact_layer}. "
               f"Avg P(source_answer) after injection = {avg_steering:.4f}. "
               f"{'PROGRAMMABLE' if avg_steering > 0.01 else 'NOT PROGRAMMABLE'}: "
               f"LLM {'can' if avg_steering > 0.01 else 'cannot'} be steered by register-layer DMA.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 2: KV Cache DMA Injection',
        'summary': {'verdict': verdict, 'avg_steering_prob': avg_steering},
        'injection_results': injection_results,
        'layer_impacts': impacts,
    }
    save_results("phase2_dma_injection", result)
    return result


if __name__ == '__main__':
    main()
