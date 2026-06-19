# -*- coding: utf-8 -*-
"""
Phase 281: Speed of Light vs Speed of Sound
==============================================
Phase 271 measured c_s ~ 3.5 pos/layer (speed of sound).
Now measure the causal "speed of light" = attention reach per layer.
Compare c_s/c_light = Mach number. If M < 1, subsonic; M > 1, supersonic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "The fundamental theorem of calculus connects",
    "Machine learning models are trained by",
]


def measure_speed_of_sound(model, tok, prompt, device, sigma=0.1):
    """Measure how fast a perturbation propagates through layers."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    clean_states = [h[0].float().cpu() for h in out_clean.hidden_states]

    # Inject noise at position 0 via embedding hook
    def make_embed_hook(sigma_val):
        def hook_fn(module, input, output):
            # output from embed_tokens: (batch, seq, hidden)
            h = output
            if h.dim() == 3:
                noise = torch.randn(1, 1, h.shape[-1], device=h.device, dtype=h.dtype) * sigma_val
                h_new = h.clone()
                h_new[:, 0:1, :] = h[:, 0:1, :] + noise
            elif h.dim() == 2:
                noise = torch.randn(1, h.shape[-1], device=h.device, dtype=h.dtype) * sigma_val
                h_new = h.clone()
                h_new[0:1, :] = h[0:1, :] + noise
            else:
                return output
            return h_new
        return hook_fn

    handle = model.model.embed_tokens.register_forward_hook(make_embed_hook(sigma))
    with torch.no_grad():
        out_noisy = model(**inp, output_hidden_states=True)
    handle.remove()

    noisy_states = [h[0].float().cpu() for h in out_noisy.hidden_states]

    # Measure perturbation spread
    speeds = []
    for li in range(1, len(clean_states)):
        diff = (noisy_states[li] - clean_states[li]).norm(dim=-1)  # (seq,)
        if diff.sum() < 1e-8:
            continue
        # Find furthest position with significant perturbation
        threshold = diff.max() * 0.1
        affected = (diff > threshold).nonzero(as_tuple=True)[0]
        if len(affected) > 0:
            max_reach = affected.max().item()
            speeds.append(max_reach / li)

    return float(np.mean(speeds)) if speeds else 0.0


def measure_attention_light_cone(model, tok, prompt, device):
    """Measure effective attention reach per layer (causal light cone)."""
    # For causal models, the light cone is trivially seq_len at each layer
    # But the *effective* cone is how far attention actually reaches
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    # We measure via gradient flow: which positions influence the final token
    # Approximate with hidden state correlation falloff
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    reaches = []
    for li, hs in enumerate(out.hidden_states[1:], 1):
        h = hs[0].float()  # (seq, hidden)
        if h.shape[0] < 3:
            continue
        # Correlation of last token with all others
        last = h[-1]
        sims = torch.nn.functional.cosine_similarity(h[:-1], last.unsqueeze(0), dim=-1)
        # Effective reach = number of tokens with sim > 0.5
        effective_reach = (sims > 0.5).sum().item()
        reaches.append(effective_reach)

    # Speed of light = rate of reach expansion
    if len(reaches) >= 2:
        x = np.arange(len(reaches))
        slope, _, _, _, _ = __import__('scipy').stats.linregress(x, reaches)
        return float(slope), reaches
    return 0.0, reaches


def main():
    print("=" * 70)
    print("Phase 281: Speed of Light vs Speed of Sound")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        # Speed of sound
        c_s_values = []
        for prompt in PROMPTS:
            cs = measure_speed_of_sound(model, tok, prompt, device)
            c_s_values.append(cs)
        c_s = float(np.mean(c_s_values))

        # Speed of light (attention reach)
        c_l_values = []
        reach_profiles = []
        for prompt in PROMPTS:
            cl, reaches = measure_attention_light_cone(model, tok, prompt, device)
            c_l_values.append(cl)
            reach_profiles.append(reaches)
        c_l = float(np.mean(c_l_values))

        mach = c_s / max(c_l, 1e-10)

        all_results[size] = {
            'c_sound': round(c_s, 4),
            'c_light': round(c_l, 4),
            'mach_number': round(mach, 4),
            'subsonic': mach < 1.0,
            'reach_profile': [round(float(np.mean([p[i] for p in reach_profiles if i < len(p)])), 2)
                             for i in range(max(len(p) for p in reach_profiles))],
        }
        print(f"  c_s = {c_s:.3f} pos/layer")
        print(f"  c_light = {c_l:.3f} pos/layer")
        print(f"  Mach = {mach:.3f} ({'subsonic' if mach < 1 else 'SUPERSONIC'})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    sizes = list(all_results.keys())

    # (a) Speed comparison
    x = np.arange(len(sizes))
    w = 0.35
    axes[0, 0].bar(x - w/2, [all_results[s]['c_sound'] for s in sizes], w,
                  label='Sound', color='#3498db')
    axes[0, 0].bar(x + w/2, [all_results[s]['c_light'] for s in sizes], w,
                  label='Light', color='#e74c3c')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(sizes)
    axes[0, 0].set_ylabel('Speed (pos/layer)')
    axes[0, 0].set_title('(a) Speed of Sound vs Light', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Mach number
    machs = [all_results[s]['mach_number'] for s in sizes]
    bars = axes[0, 1].bar(sizes, machs, color=[colors[s] for s in sizes])
    axes[0, 1].axhline(1.0, color='red', ls='--', label='Mach = 1')
    axes[0, 1].set_ylabel('Mach Number')
    axes[0, 1].set_title('(b) Mach Number', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Attention reach profile
    for s in sizes:
        rp = all_results[s]['reach_profile']
        axes[0, 2].plot(rp, '-', color=colors[s], lw=2, label=s)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Effective Reach (# tokens)')
    axes[0, 2].set_title('(c) Attention Light Cone', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d-f) Summary
    for idx in range(3):
        axes[1, idx].axis('off')

    txt = "SPEED OF LIGHT vs SPEED OF SOUND\n\n"
    for s in sizes:
        r = all_results[s]
        txt += f"{s}:\n"
        txt += f"  c_sound  = {r['c_sound']:.3f} pos/layer\n"
        txt += f"  c_light  = {r['c_light']:.3f} pos/layer\n"
        txt += f"  Mach     = {r['mach_number']:.3f}\n"
        txt += f"  Regime   = {'Subsonic' if r['subsonic'] else 'SUPERSONIC'}\n\n"
    txt += "Mach < 1 -> subsonic (smooth)\n"
    txt += "Mach > 1 -> supersonic (shocks)"
    axes[1, 1].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')

    fig.suptitle("Phase 281: Speed of Light vs Speed of Sound",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase281_mach_number')
    plt.close()

    save_results('phase281_mach_number', {
        'experiment': 'Speed of Light vs Speed of Sound',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
