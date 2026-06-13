# -*- coding: utf-8 -*-
"""
Phase 155b: Thermodynamic Decoding
Use kT to automatically adjust sampling temperature at each layer.
Theory: if the model is "hot" (high kT), it's exploring;
if "cold" (low kT), it's confident. Use this to adaptively decode.
Compare: greedy, standard sampling, and thermodynamic decoding.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def safe_str(s):
    """Sanitize string for cp932 console output."""
    return s.encode('ascii', errors='replace').decode('ascii')


def measure_kT_profile(model, tok, input_ids, device, n_layers):
    """Get kT at each layer for the current input."""
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    kT_vals = []
    for li in range(n_layers):
        hs = out.hidden_states[li]
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)

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
        kT_vals.append(float(kT))

    return kT_vals, out.logits[0, -1, :].float()


def generate_greedy(model, tok, prompt, n_tokens, device, n_layers):
    """Standard greedy generation."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    tokens = []
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids)
        next_id = torch.argmax(out.logits[0, -1, :])
        tokens.append(tok.decode([next_id]))
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return "".join(tokens)


def generate_fixed_temp(model, tok, prompt, n_tokens, device, n_layers, temp=0.8):
    """Fixed temperature sampling."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    tokens = []
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, -1, :].float() / temp
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        tokens.append(tok.decode(next_id[0]))
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    return "".join(tokens)


def generate_thermodynamic(model, tok, prompt, n_tokens, device, n_layers):
    """Thermodynamic decoding: use final kT to set sampling temperature."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    tokens = []
    kT_history = []
    temp_history = []

    for _ in range(n_tokens):
        kT_profile, final_logits = measure_kT_profile(model, tok, input_ids, device, n_layers)
        final_kT = kT_profile[-1]

        # Adaptive temperature: normalize kT to reasonable sampling range
        # High kT (model uncertain) -> lower temperature (force decision)
        # Low kT (model confident) -> slightly higher temperature (allow creativity)
        adaptive_temp = max(0.3, min(1.5, 10.0 / (final_kT + 1e-10)))

        logits = final_logits / adaptive_temp
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        tokens.append(tok.decode(next_id[0]))
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
        kT_history.append(final_kT)
        temp_history.append(adaptive_temp)

    return "".join(tokens), kT_history, temp_history


def main():
    print("=" * 70)
    print("Phase 155b: Thermodynamic Decoding")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    n_tokens = 30

    prompts = [
        "The most beautiful equation in mathematics is",
        "Artificial intelligence will transform society by",
        "The secret to understanding quantum mechanics is",
    ]

    all_results = {}
    for pi, prompt in enumerate(prompts):
        print(f"\n  Prompt {pi}: {prompt}")

        # Greedy
        greedy_text = generate_greedy(model, tok, prompt, n_tokens, device, n_layers)
        print(f"    Greedy:  {safe_str(greedy_text[:60])}...")

        # Fixed temp
        fixed_text = generate_fixed_temp(model, tok, prompt, n_tokens, device, n_layers, 0.8)
        print(f"    Fixed:   {safe_str(fixed_text[:60])}...")

        # Thermodynamic
        thermo_text, kT_hist, temp_hist = generate_thermodynamic(
            model, tok, prompt, n_tokens, device, n_layers)
        print(f"    Thermo:  {safe_str(thermo_text[:60])}...")

        all_results[f'prompt_{pi}'] = {
            'prompt': prompt,
            'greedy': greedy_text,
            'fixed': fixed_text,
            'thermo': thermo_text,
            'kT_history': kT_hist,
            'temp_history': temp_hist,
        }

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) kT history for all prompts
    for pi, (key, r) in enumerate(all_results.items()):
        axes[0,0].plot(range(len(r['kT_history'])), r['kT_history'],
                      'o-', markersize=4, label=f'P{pi}')
    axes[0,0].set_xlabel('Token Step')
    axes[0,0].set_ylabel('$kT$')
    axes[0,0].set_title('(a) kT During Generation')
    axes[0,0].legend()

    # (b) Adaptive temperature
    for pi, (key, r) in enumerate(all_results.items()):
        axes[0,1].plot(range(len(r['temp_history'])), r['temp_history'],
                      'o-', markersize=4, label=f'P{pi}')
    axes[0,1].axhline(y=0.8, color='gray', linestyle='--', label='Fixed temp')
    axes[0,1].set_xlabel('Token Step')
    axes[0,1].set_ylabel('Adaptive Temperature')
    axes[0,1].set_title('(b) Adaptive Sampling Temperature')
    axes[0,1].legend(fontsize=7)

    # (c) kT vs adaptive temp scatter
    all_kTs = []
    all_temps = []
    for r in all_results.values():
        all_kTs.extend(r['kT_history'])
        all_temps.extend(r['temp_history'])
    axes[0,2].scatter(all_kTs, all_temps, c='#8e44ad', s=40, alpha=0.6, edgecolors='black')
    axes[0,2].set_xlabel('$kT$ (model temperature)')
    axes[0,2].set_ylabel('Sampling temperature')
    axes[0,2].set_title('(c) Feedback Loop')

    # (d-f) Generated text comparisons
    for pi, (key, r) in enumerate(all_results.items()):
        if pi < 3:
            ax = axes[1, pi]
            text = (
                f"Prompt: {r['prompt'][:30]}...\n\n"
                f"GREEDY:\n{r['greedy'][:80]}...\n\n"
                f"FIXED (0.8):\n{r['fixed'][:80]}...\n\n"
                f"THERMO:\n{r['thermo'][:80]}..."
            )
            ax.text(0.05, 0.95, text, ha='left', va='top',
                    transform=ax.transAxes, fontsize=7,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.axis('off')
            ax.set_title(f'({"def"[pi]}) Prompt {pi} Outputs')

    fig.suptitle('Phase 155b: Thermodynamic Decoding',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase155b_decoding')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Thermodynamic decoding complete")
    print(f"Mean kT: {np.mean(all_kTs):.1f}")
    print(f"Mean adaptive temp: {np.mean(all_temps):.3f}")
    print(f"{'='*70}")

    save_results('phase155b_decoding', {
        'experiment': 'Thermodynamic Decoding',
        'summary': {
            'mean_kT': float(np.mean(all_kTs)),
            'mean_adaptive_temp': float(np.mean(all_temps)),
        }
    })


if __name__ == '__main__':
    main()
