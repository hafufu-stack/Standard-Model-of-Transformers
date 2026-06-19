# -*- coding: utf-8 -*-
"""
Phase 354: D-brane Dynamics -- Open/Closed String Duality
=====================================================
D-branes are surfaces on which open strings end. Open strings
on D-branes correspond to gauge theories, while closed strings
in the bulk correspond to gravity (gauge/gravity duality).
Test whether attention heads (open strings) and residual stream
(closed strings) satisfy this duality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_dbrane(model, tok, prompt, device):
    """Test D-brane / open-closed string duality."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, output_attentions=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]

    # "Open string" = attention patterns (localized, on the brane)
    # "Closed string" = residual stream (propagating in bulk)
    open_energies = []
    closed_energies = []

    for li in range(n_layers):
        attn = out.attentions[li][0]  # (n_heads, seq, seq)
        # Open string energy: entropy of attention pattern
        attn_last = attn[:, -1, :]  # attention from last token
        open_E = 0.0
        for head in range(attn_last.shape[0]):
            p = attn_last[head].float().cpu()
            p = p / (p.sum() + 1e-30)
            s = float(-torch.sum(p * torch.log(p + 1e-30)).item())
            open_E += s
        open_E /= attn_last.shape[0]
        open_energies.append(round(float(open_E), 4))

        # Closed string energy: change in residual stream norm
        delta_h = hiddens[li + 1] - hiddens[li]
        closed_E = float(torch.norm(delta_h).item())
        closed_energies.append(round(float(closed_E), 4))

    # Open-closed duality: correlation between open and closed energies
    if len(open_energies) > 3:
        r_oc, p_oc = stats.pearsonr(open_energies, closed_energies)
    else:
        r_oc, p_oc = 0, 1

    # Brane tension: T_p ~ 1/g_s * (2*pi*alpha')^{-(p+1)/2}
    # Proxy: ratio of total open energy to total closed energy
    total_open = float(np.sum(open_energies))
    total_closed = float(np.sum(closed_energies))
    tension = total_open / (total_closed + 1e-10)

    # Chan-Paton factors: number of D-brane stacks
    # Proxy: number of distinct attention head clusters
    if n_layers > 0:
        attn_0 = out.attentions[0][0][:, -1, :].float().cpu()
        n_heads = attn_0.shape[0]
        # Cluster by correlation
        corr_matrix = np.corrcoef(attn_0.numpy())
        # Count clusters (eigenvalues above threshold)
        eigs = np.linalg.eigvalsh(corr_matrix)
        n_stacks = int(np.sum(eigs > 0.5 * np.max(eigs)))
    else:
        n_stacks = 0

    # Dirichlet boundary condition: attention is "pinned" at certain positions
    # Neumann boundary condition: residual stream flows freely
    dirichlet_score = float(np.std(open_energies))  # Low std = pinned
    neumann_score = float(np.std(closed_energies))  # High std = free

    return {
        'open_energies': open_energies,
        'closed_energies': closed_energies,
        'r_oc': round(float(r_oc), 4),
        'p_oc': round(float(p_oc), 6),
        'tension': round(float(tension), 4),
        'n_stacks': n_stacks,
        'dirichlet': round(float(dirichlet_score), 4),
        'neumann': round(float(neumann_score), 4),
        'duality_holds': abs(r_oc) > 0.3,
    }


def main():
    print("=" * 70)
    print("Phase 354: D-brane Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        if size == '0.5B':
            mid = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                              "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")
        else:
            mid = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                              "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float32, device_map=device,
            local_files_only=True, attn_implementation='eager',
        )
        model.eval()

        db_data = []
        for prompt in PROMPTS:
            d = measure_dbrane(model, tok, prompt, device)
            db_data.append(d)

        n = len(db_data[0]['open_energies'])
        all_results[size] = {
            'open_energies': [round(float(np.mean([d['open_energies'][i] for d in db_data])), 4)
                             for i in range(n)],
            'closed_energies': [round(float(np.mean([d['closed_energies'][i] for d in db_data])), 4)
                               for i in range(n)],
            'r_oc': round(float(np.mean([d['r_oc'] for d in db_data])), 4),
            'tension': round(float(np.mean([d['tension'] for d in db_data])), 4),
            'n_stacks': round(float(np.mean([d['n_stacks'] for d in db_data])), 1),
            'dirichlet': round(float(np.mean([d['dirichlet'] for d in db_data])), 4),
            'neumann': round(float(np.mean([d['neumann'] for d in db_data])), 4),
            'duality_holds': sum(1 for d in db_data if d['duality_holds']) >= 4,
        }
        holds = 'YES' if all_results[size]['duality_holds'] else 'NO'
        print(f"  Open-closed R: {all_results[size]['r_oc']:.4f}")
        print(f"  Tension: {all_results[size]['tension']:.4f}")
        print(f"  D-brane stacks: {all_results[size]['n_stacks']:.1f}")
        print(f"  Duality: {holds}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['open_energies'], '-', color=colors[size], lw=2, label=f'{size} open')
        axes[0, 0].plot(data['closed_energies'], '--', color=colors[size], lw=2, label=f'{size} closed')
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('(a) Open vs Closed String Energy', fontweight='bold')
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].scatter(data['open_energies'], data['closed_energies'],
                          color=colors[size], s=40, alpha=0.7, label=size)
    axes[0, 1].set_xlabel('Open E'); axes[0, 1].set_ylabel('Closed E')
    axes[0, 1].set_title('(b) Open-Closed Correlation', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    axes[0, 2].bar(sizes, [all_results[s]['r_oc'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_title('(c) Duality Correlation', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    w = 0.25
    axes[1, 0].bar(x - w/2, [all_results[s]['dirichlet'] for s in sizes], w,
                  label='Dirichlet', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['neumann'] for s in sizes], w,
                  label='Neumann', color='#e74c3c')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) Boundary Conditions', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "D-BRANE DYNAMICS\n\n"
    for s in sizes:
        d = all_results[s]
        h = 'YES' if d['duality_holds'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  R_oc = {d['r_oc']:.3f}\n"
        txt += f"  T = {d['tension']:.3f}\n"
        txt += f"  stacks = {d['n_stacks']:.0f}\n"
        txt += f"  duality: {h}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 354: D-brane Dynamics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase354_dbrane')
    plt.close()
    save_results('phase354_dbrane', {'experiment': 'D-brane', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
