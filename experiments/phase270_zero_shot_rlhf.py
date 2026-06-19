# -*- coding: utf-8 -*-
"""
Phase 270: Zero-Shot RLHF via Artificial Cooling
===================================================
RLHF functions as a cooling mechanism (Phase 265 proved).
Can we replicate RLHF alignment by artificially cooling the Base model?

Two approaches tested:
  (A) Hidden state hook: scale down final-layer hidden state norm
  (B) Logits temperature: apply post-hoc temperature to reduce entropy

Target: Match Instruct model's thermodynamic profile (T ~ 2.96)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_any_model, get_model_internals, save_results, save_figure

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "How do I sort a list in Python?",
    "What are the benefits of exercise?",
]

MAX_NEW_TOKENS = 60


def measure_thermodynamic_profile(model, tok, prompt, device, internals):
    """Measure P1, T at the final layer for a prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs_last = out.hidden_states[-1]
    normed = internals['norm'](hs_last[:, -1:, :])
    logits = internals['lm_head'](normed).squeeze().float()
    probs = torch.softmax(logits, dim=-1)
    p1 = probs.max().item()
    t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
    u = hs_last[0, -1, :].float().norm().item()
    return {'P1': p1, 'T': t_val, 'U': u, 'P1T': p1 * t_val}


def generate_with_cooling(model, tok, prompt, device, internals,
                          cooling_method='logits', target_t=2.96):
    """Generate text with artificial cooling applied."""
    inp = tok(prompt, return_tensors='pt').to(device)
    generated = inp['input_ids'].clone()
    input_len = generated.shape[1]
    p1_trace, t_trace = [], []

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(generated, output_hidden_states=True)

        raw_logits = out.logits[0, -1, :].float()
        raw_probs = torch.softmax(raw_logits, dim=-1)
        raw_t = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()

        if cooling_method == 'logits':
            # Method B: Scale logits to reduce entropy to target
            if raw_t > target_t and raw_t > 0.1:
                scale = target_t / raw_t
                # Lower temperature = sharper distribution
                scaled_logits = raw_logits / max(scale, 0.1)
            else:
                scaled_logits = raw_logits
            probs = torch.softmax(scaled_logits, dim=-1)

        elif cooling_method == 'hidden':
            # Method A: Scale hidden state norm, then recompute logits
            hs = out.hidden_states[-1][:, -1:, :].clone()
            current_norm = hs.float().norm().item()
            # Reduce norm to cool the system
            if raw_t > target_t and current_norm > 1.0:
                cool_factor = target_t / max(raw_t, 0.01)
                hs = hs * cool_factor
            normed = internals['norm'](hs)
            cooled_logits = internals['lm_head'](normed).squeeze().float()
            probs = torch.softmax(cooled_logits, dim=-1)

        else:
            probs = raw_probs

        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        t_trace.append(t_val)

        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break

    text = tok.decode(generated[0, input_len:], skip_special_tokens=True)
    return text, p1_trace, t_trace


def main():
    print("=" * 70)
    print("Phase 270: Zero-Shot RLHF via Artificial Cooling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load all models
    models_to_test = [
        ('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'),
        ('Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
    ]

    profiles = {}
    generations = {}

    for name, model_id in models_to_test:
        print(f"\n--- Loading {name} ---")
        model, tok = load_any_model(model_id, device=device)
        internals = get_model_internals(model)

        # Measure baseline thermodynamic profile
        profile_data = []
        for prompt in TEST_PROMPTS:
            p = measure_thermodynamic_profile(model, tok, prompt, device, internals)
            profile_data.append(p)
        profiles[name] = profile_data
        mean_t = np.mean([p['T'] for p in profile_data])
        mean_p1t = np.mean([p['P1T'] for p in profile_data])
        print(f"  Mean T = {mean_t:.3f}, Mean P1T = {mean_p1t:.3f}")

        # Generate sample text
        gen_data = []
        for prompt in TEST_PROMPTS[:3]:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(inp['input_ids'], max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=False)
            text = tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            gen_data.append({'prompt': prompt, 'text': text[:200]})
        generations[name] = gen_data

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Now test artificial cooling on Base model
    instruct_t = np.mean([p['T'] for p in profiles['Qwen2.5-0.5B-Instruct']])
    print(f"\n--- Artificial Cooling (target T = {instruct_t:.3f}) ---")

    model, tok = load_any_model('Qwen/Qwen2.5-0.5B', device=device)
    internals = get_model_internals(model)

    cooling_results = {}
    for method in ['logits', 'hidden', 'none']:
        print(f"\n  Method: {method}")
        method_data = []
        for prompt in TEST_PROMPTS[:3]:
            text, p1s, ts = generate_with_cooling(
                model, tok, prompt, device, internals,
                cooling_method=method, target_t=instruct_t
            )
            method_data.append({
                'prompt': prompt,
                'text': text[:200],
                'mean_p1': round(float(np.mean(p1s)), 4),
                'mean_t': round(float(np.mean(ts)), 4),
                'mean_p1t': round(float(np.mean([p*t for p, t in zip(p1s, ts)])), 4),
            })
            print(f"    T={method_data[-1]['mean_t']:.3f}, P1T={method_data[-1]['mean_p1t']:.3f}")
        cooling_results[method] = method_data

    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_model = {'Qwen2.5-0.5B': '#3498db', 'Qwen2.5-0.5B-Instruct': '#e74c3c'}
    colors_method = {'logits': '#2ecc71', 'hidden': '#9b59b6', 'none': '#95a5a6'}

    # (a) Temperature comparison: Base vs Instruct
    for name in profiles:
        ts = [p['T'] for p in profiles[name]]
        axes[0, 0].bar(name, np.mean(ts), color=colors_model.get(name, '#999'),
                      yerr=np.std(ts), capsize=5)
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Base vs Instruct Temperature', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # (b) P1*T comparison
    for name in profiles:
        p1ts = [p['P1T'] for p in profiles[name]]
        axes[0, 1].bar(name, np.mean(p1ts), color=colors_model.get(name, '#999'),
                      yerr=np.std(p1ts), capsize=5)
    axes[0, 1].axhline(0.84, color='red', ls='--', label='P1T=0.84')
    axes[0, 1].set_ylabel('P1 * T')
    axes[0, 1].set_title('(b) P1*T Conservation Check', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].tick_params(axis='x', rotation=15)

    # (c) Cooling method comparison: T achieved
    method_names = list(cooling_results.keys())
    mean_ts = [np.mean([d['mean_t'] for d in cooling_results[m]]) for m in method_names]
    bars = axes[0, 2].bar(method_names, mean_ts,
                          color=[colors_method[m] for m in method_names])
    axes[0, 2].axhline(instruct_t, color='red', ls='--', label=f'Instruct T={instruct_t:.2f}')
    axes[0, 2].set_ylabel('Achieved Temperature')
    axes[0, 2].set_title('(c) Cooling Method Effectiveness', fontweight='bold')
    axes[0, 2].legend()

    # (d) P1*T after cooling
    mean_p1ts = [np.mean([d['mean_p1t'] for d in cooling_results[m]]) for m in method_names]
    bars = axes[1, 0].bar(method_names, mean_p1ts,
                          color=[colors_method[m] for m in method_names])
    axes[1, 0].axhline(0.84, color='red', ls='--', label='P1T=0.84')
    axes[1, 0].set_ylabel('P1 * T')
    axes[1, 0].set_title('(d) P1*T After Cooling', fontweight='bold')
    axes[1, 0].legend()

    # (e) Sample generations comparison
    gen_text = "SAMPLE GENERATIONS\n\n"
    gen_text += "Base (uncooled):\n"
    if 'none' in cooling_results:
        gen_text += f"  {cooling_results['none'][0]['text'][:80]}...\n\n"
    gen_text += "Base (logits-cooled):\n"
    if 'logits' in cooling_results:
        gen_text += f"  {cooling_results['logits'][0]['text'][:80]}...\n\n"
    gen_text += "Instruct (natural):\n"
    if 'Qwen2.5-0.5B-Instruct' in generations:
        gen_text += f"  {generations['Qwen2.5-0.5B-Instruct'][0]['text'][:80]}...\n"
    axes[1, 1].text(0.05, 0.95, gen_text, ha='left', va='top',
                   transform=axes[1, 1].transAxes, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace', wrap=True)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('(e) Generation Samples')

    # (f) Summary
    summary_txt = "ZERO-SHOT RLHF VIA COOLING\n\n"
    summary_txt += f"Instruct target T: {instruct_t:.3f}\n\n"
    for method in method_names:
        mt = np.mean([d['mean_t'] for d in cooling_results[method]])
        mp = np.mean([d['mean_p1t'] for d in cooling_results[method]])
        summary_txt += f"{method}: T={mt:.3f}, P1T={mp:.3f}\n"
    axes[1, 2].text(0.5, 0.5, summary_txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 270: Zero-Shot RLHF via Artificial Cooling",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase270_zero_shot_rlhf')
    plt.close()

    save_results('phase270_zero_shot_rlhf', {
        'experiment': 'Zero-Shot RLHF via Artificial Cooling',
        'instruct_target_t': round(instruct_t, 4),
        'profiles': {k: [dict(p) for p in v] for k, v in profiles.items()},
        'cooling_results': cooling_results,
    })


if __name__ == '__main__':
    main()
