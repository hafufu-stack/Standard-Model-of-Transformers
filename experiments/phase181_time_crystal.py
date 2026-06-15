# -*- coding: utf-8 -*-
"""
Phase 181: Time Crystal Dynamics
Combine iterative loop (Phase 57 black hole) with Brownian ratchet.
Test if waste heat recycling prevents T->0 collapse and creates
perpetual oscillation (Time Crystal state).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def iterate_with_ratchet(model, tok, seed_text, device, n_iterations=30, ratchet_on=False, gain=0.01):
    """Feed output back as input iteratively, optionally with ratchet."""
    current_text = seed_text
    history = {'T': [], 'U': [], 'S': [], 'conf': [], 'text': []}

    for step in range(n_iterations):
        inp = tok(current_text, return_tensors='pt', truncation=True, max_length=128).to(device)
        n_layers_total = len(model.model.layers)

        # Ratchet hooks
        waste_store = {}
        handles = []
        if ratchet_on:
            def make_hook(li):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_f = h.float()
                    waste_store[li] = h_f.norm().item() * 0.01  # waste fraction
                    if li > 0 and (li - 1) in waste_store:
                        noise = torch.randn_like(h_f) * gain * waste_store[li - 1]
                        cos = torch.nn.functional.cosine_similarity(noise, h_f, dim=-1, eps=1e-8)
                        gate = (cos > 0).float().unsqueeze(-1)
                        h_mod = h_f + noise * gate
                        h_mod = torch.nan_to_num(h_mod, nan=0.0)
                        result = h_mod.to(h.dtype)
                        if isinstance(output, tuple):
                            return (result,) + output[1:]
                        return result
                return hook
            for i in range(n_layers_total):
                handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in handles:
            h.remove()

        # Measure at final hidden state
        hs_last = out.hidden_states[-1]
        h = hs_last[0, -1, :].float()
        U = h.norm().item()
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        conf = probs.max().item()

        history['U'].append(U if not np.isnan(U) else 0)
        history['T'].append(T if not np.isnan(T) else 0)
        history['S'].append(S if not np.isnan(S) else 0)
        history['conf'].append(conf)

        # Generate next token and feed back
        next_token = probs.argmax().item()
        next_text = tok.decode([next_token])
        current_text = current_text + next_text
        history['text'].append(next_text)

        # Keep context manageable
        if len(current_text) > 500:
            current_text = current_text[-300:]

    return history


def main():
    print("=" * 70)
    print("Phase 181: Time Crystal Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    seed = "The nature of consciousness is"
    n_iter = 40

    # Without ratchet (should collapse to T~0)
    print("\n--- Without Ratchet (Black Hole) ---")
    hist_no = iterate_with_ratchet(model, tok, seed, device, n_iterations=n_iter, ratchet_on=False)

    # With ratchet (should oscillate)
    print("--- With Ratchet (Time Crystal?) ---")
    gains_to_test = [0.005, 0.01, 0.05]
    hist_ratchets = {}
    for g in gains_to_test:
        print(f"  gain={g}")
        hist_ratchets[g] = iterate_with_ratchet(model, tok, seed, device,
                                                 n_iterations=n_iter, ratchet_on=True, gain=g)

    # === Analysis ===
    # Check for oscillation (autocorrelation)
    def detect_oscillation(series):
        if len(series) < 10:
            return 0.0
        s = np.array(series)
        s = s - np.mean(s)
        if np.std(s) < 1e-10:
            return 0.0
        autocorr = np.correlate(s, s, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        # Find first significant peak after lag 0
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                return float(autocorr[i])
        return 0.0

    osc_no = detect_oscillation(hist_no['T'])
    osc_ratchet = {g: detect_oscillation(hist_ratchets[g]['T']) for g in gains_to_test}

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    steps = np.arange(n_iter)

    # (a) Temperature evolution
    axes[0, 0].plot(steps, hist_no['T'], 'o-', color='#2c3e50', markersize=3, linewidth=2, label='No Ratchet')
    for g in gains_to_test:
        axes[0, 0].plot(steps, hist_ratchets[g]['T'], 'o-', markersize=3, linewidth=1.5, label=f'gain={g}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Temperature $T$')
    axes[0, 0].set_title('(a) Temperature: Black Hole vs Time Crystal')
    axes[0, 0].legend(fontsize=7)

    # (b) Energy evolution
    axes[0, 1].plot(steps, hist_no['U'], 'o-', color='#2c3e50', markersize=3, linewidth=2, label='No Ratchet')
    for g in gains_to_test:
        axes[0, 1].plot(steps, hist_ratchets[g]['U'], 'o-', markersize=3, linewidth=1.5, label=f'gain={g}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Energy $U$')
    axes[0, 1].set_title('(b) Energy Evolution')
    axes[0, 1].legend(fontsize=7)

    # (c) Entropy evolution
    axes[0, 2].plot(steps, hist_no['S'], 'o-', color='#2c3e50', markersize=3, linewidth=2, label='No Ratchet')
    for g in gains_to_test:
        axes[0, 2].plot(steps, hist_ratchets[g]['S'], 'o-', markersize=3, linewidth=1.5, label=f'gain={g}')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Entropy $S$')
    axes[0, 2].set_title('(c) Entropy Evolution')
    axes[0, 2].legend(fontsize=7)

    # (d) Phase portrait T vs U
    axes[1, 0].plot(hist_no['U'], hist_no['T'], 'o-', color='#2c3e50', markersize=3, linewidth=1.5, label='No Ratchet')
    best_g = max(gains_to_test, key=lambda g: osc_ratchet[g])
    axes[1, 0].plot(hist_ratchets[best_g]['U'], hist_ratchets[best_g]['T'],
                    'o-', color='#e74c3c', markersize=3, linewidth=1.5, label=f'Ratchet (gain={best_g})')
    axes[1, 0].scatter(hist_no['U'][0], hist_no['T'][0], s=100, c='green', marker='^', zorder=5)
    axes[1, 0].set_xlabel('Energy $U$')
    axes[1, 0].set_ylabel('Temperature $T$')
    axes[1, 0].set_title('(d) Phase Portrait (T vs U)')
    axes[1, 0].legend(fontsize=8)

    # (e) Confidence evolution
    axes[1, 1].plot(steps, hist_no['conf'], 'o-', color='#2c3e50', markersize=3, linewidth=2, label='No Ratchet')
    for g in gains_to_test:
        axes[1, 1].plot(steps, hist_ratchets[g]['conf'], 'o-', markersize=3, linewidth=1.5, label=f'gain={g}')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('(e) Confidence (Convergence)')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    T_final_no = hist_no['T'][-1]
    T_final_best = hist_ratchets[best_g]['T'][-1]
    summary = (
        f"Time Crystal Dynamics\n\n"
        f"NO RATCHET (Black Hole):\n"
        f"  T(final) = {T_final_no:.2f}\n"
        f"  Oscillation = {osc_no:.3f}\n\n"
        f"RATCHET (gain={best_g}):\n"
        f"  T(final) = {T_final_best:.2f}\n"
        f"  Oscillation = {osc_ratchet[best_g]:.3f}\n\n"
        f"Time Crystal: {'YES' if osc_ratchet[best_g] > 0.2 else 'NO'}\n"
        f"(osc > 0.2 = periodic)"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 181: Time Crystal Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase181_time_crystal')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Black Hole T(final)={T_final_no:.2f}, osc={osc_no:.3f}")
    print(f"Best Ratchet (g={best_g}) T(final)={T_final_best:.2f}, osc={osc_ratchet[best_g]:.3f}")
    print(f"Time Crystal: {'YES' if osc_ratchet[best_g] > 0.2 else 'NO'}")
    print(f"{'=' * 70}")

    save_results('phase181_time_crystal', {
        'experiment': 'Time Crystal Dynamics',
        'no_ratchet': {'T': hist_no['T'], 'U': hist_no['U'], 'S': hist_no['S'],
                       'osc': float(osc_no)},
        'ratchet': {str(g): {'T': hist_ratchets[g]['T'], 'U': hist_ratchets[g]['U'],
                             'S': hist_ratchets[g]['S'], 'osc': float(osc_ratchet[g])}
                    for g in gains_to_test},
    })


if __name__ == '__main__':
    main()
