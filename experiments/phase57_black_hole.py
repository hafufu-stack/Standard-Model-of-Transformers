# -*- coding: utf-8 -*-
"""
Phase 57: Black Hole Singularity
Observe gravitational collapse (repetition loops) thermodynamically.
U should diverge and T should collapse to zero = black hole formation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 57: Black Hole Singularity (Repetition Collapse)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Prompts designed to induce repetition collapse
    collapse_prompts = [
        "The number one is followed by one which is followed by one which is followed by",
        "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo Buffalo buffalo",
        "This sentence is a sentence that is a sentence that is a sentence that is",
    ]

    # Normal prompts for comparison
    normal_prompts = [
        "The history of mathematics spans thousands of years across many civilizations and",
        "Quantum computing promises to revolutionize fields like cryptography and drug discovery by",
        "The Amazon rainforest contains the greatest biodiversity on Earth with millions of",
    ]

    GEN_LENGTH = 100
    all_results = []

    for prompts, label in [(normal_prompts, 'normal'), (collapse_prompts, 'collapse')]:
        for prompt in prompts:
            input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
            current_ids = input_ids.clone()

            prt_trace = []
            pr_trace = []
            t_trace = []
            u_trace = []
            tokens_gen = []
            repetition_count = 0
            last_tokens = []

            for t_step in range(GEN_LENGTH):
                with torch.no_grad():
                    out = model(current_ids, output_hidden_states=True)
                    logits_raw = out.logits[0, -1, :].float()

                # Measure thermodynamics from last hidden state
                last_hs = out.hidden_states[-1][0, -1, :].float()
                U = last_hs.norm().item()

                probs = torch.softmax(logits_raw, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T_val

                prt_trace.append(PRT if not np.isnan(PRT) else 0)
                pr_trace.append(PR)
                t_trace.append(T_val)
                u_trace.append(U)

                next_id = logits_raw.argmax().item()
                tokens_gen.append(next_id)

                # Track repetition
                last_tokens.append(next_id)
                if len(last_tokens) > 5:
                    # Check for 3-gram repetition
                    recent = last_tokens[-3:]
                    earlier = last_tokens[-6:-3]
                    if recent == earlier:
                        repetition_count += 1

                next_tensor = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)
                if current_ids.shape[1] > 512:
                    current_ids = current_ids[:, -512:]

            text = tok.decode(tokens_gen, skip_special_tokens=True)
            safe_text = text.encode('ascii', errors='replace').decode('ascii')[:50]

            # Detect collapse: did T approach zero?
            T_final_avg = np.mean(t_trace[-10:]) if len(t_trace) >= 10 else 0
            U_final_avg = np.mean(u_trace[-10:]) if len(u_trace) >= 10 else 0
            T_collapsed = T_final_avg < 0.5
            U_diverged = U_final_avg > np.mean(u_trace[:10]) * 1.5 if len(u_trace) > 10 else False

            print(f"  [{label}] T_final={T_final_avg:.2f}, U_final={U_final_avg:.1f}, "
                  f"reps={repetition_count}, '{safe_text}...'")

            all_results.append({
                'label': label, 'prompt': prompt[:60],
                'prt_trace': [float(v) for v in prt_trace],
                'pr_trace': [float(v) for v in pr_trace],
                't_trace': [float(v) for v in t_trace],
                'u_trace': [float(v) for v in u_trace],
                'repetition_count': repetition_count,
                'T_collapsed': T_collapsed,
                'U_diverged': U_diverged,
                'text': text[:200],
            })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) T traces
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'normal' else '#e74c3c'
        axes[0, 0].plot(r['t_trace'], color=c, alpha=0.5, linewidth=1)
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([0],[0],color='#2ecc71',label='Normal'),
                                Line2D([0],[0],color='#e74c3c',label='Collapse-prone')])
    axes[0, 0].set_xlabel('Token')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Temperature (T -> 0 = singularity)')

    # (b) U traces
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'normal' else '#e74c3c'
        axes[0, 1].plot(r['u_trace'], color=c, alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Token')
    axes[0, 1].set_ylabel('Internal Energy U')
    axes[0, 1].set_title('(b) Energy (U diverges = gravitational collapse)')

    # (c) PR traces (sharpening)
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'normal' else '#e74c3c'
        axes[0, 2].plot(r['pr_trace'], color=c, alpha=0.5, linewidth=1)
    axes[0, 2].set_xlabel('Token')
    axes[0, 2].set_ylabel('Participation Ratio')
    axes[0, 2].set_title('(c) PR (PR -> 1 = total collapse)')

    # (d) PRT traces
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'normal' else '#e74c3c'
        axes[1, 0].plot(r['prt_trace'], color=c, alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Token')
    axes[1, 0].set_ylabel('PRT')
    axes[1, 0].set_title('(d) PRT (conservation breaks at singularity?)')

    # (e) Repetition count comparison
    normal_reps = [r['repetition_count'] for r in all_results if r['label'] == 'normal']
    collapse_reps = [r['repetition_count'] for r in all_results if r['label'] == 'collapse']
    axes[1, 1].boxplot([normal_reps, collapse_reps], labels=['Normal', 'Collapse-prone'])
    axes[1, 1].set_ylabel('Repetition Count')
    axes[1, 1].set_title('(e) Repetition Events')

    # (f) T_final vs U_final scatter
    for r in all_results:
        c = '#2ecc71' if r['label'] == 'normal' else '#e74c3c'
        T_f = np.mean(r['t_trace'][-10:])
        U_f = np.mean(r['u_trace'][-10:])
        axes[1, 2].scatter(T_f, U_f, color=c, s=80, edgecolors='black', alpha=0.7)
    axes[1, 2].set_xlabel('T_final (low = singularity)')
    axes[1, 2].set_ylabel('U_final (high = collapse)')
    axes[1, 2].set_title('(f) Final State (BH = low-T, high-U)')

    fig.suptitle('Phase 57: Black Hole Singularity (Repetition = Gravitational Collapse)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase57_black_hole')
    plt.close()

    n_collapsed = sum(1 for r in all_results if r['label'] == 'collapse' and r['T_collapsed'])
    n_collapse_total = sum(1 for r in all_results if r['label'] == 'collapse')

    print(f"\n{'='*70}")
    print(f"VERDICT: {n_collapsed}/{n_collapse_total} collapse prompts reached T->0 singularity. "
          f"Repetition collapse {'IS' if n_collapsed > 0 else 'is NOT'} a thermodynamic black hole.")
    print(f"{'='*70}")

    save_results('phase57_black_hole', {
        'experiment': 'Black Hole Singularity',
        'results': [{k: v for k, v in r.items() if k != 'text'} for r in all_results],
        'summary': {
            'n_collapsed': n_collapsed,
            'n_total_collapse': n_collapse_total,
        }
    })


if __name__ == '__main__':
    main()
