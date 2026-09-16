import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimal_configurations')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')


def to_dB(r):
    """Convert squeezing parameter r (natural units) to dB."""
    return -10 * np.log10(np.exp(-2 * r))


all_Fids = np.load(os.path.join(data_dir, 'all_fidelities.npy'))

print(all_Fids)

labels = ['$p=3$', '$p=4$', '$p=5$', '$p=6$']          # x-axis ticks
titles = ['$N=1$', '$N=2$', '$N=3$', '$N=4$']

x = np.arange(len(labels))              # tick positions
width = 0.18                            # bar width


colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


fig, ax = plt.subplots(figsize=(7,3))

for i, row in enumerate(all_Fids.T):   # .T → iterate over N, values per p
    ax.bar(
        x + (i - 1.5) * width,           # shift bars (4 bars)
        row,
        width,
        label=titles[i],
        color=colors[i],
        edgecolor='black',
        linewidth=0.3
    )
    
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
ax.grid(axis='x', visible=False)


ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.set_xlabel(r'pulse number $p$', fontsize=17)
ax.set_ylabel(r'$F(N)$', fontsize=17)
ax.set_ylim(np.nanmin(all_Fids) - 0.01, 1.0 + 0.01)

ax.legend(title=r'target state $|N\rangle$', bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=13, title_fontsize=14, frameon=True)

ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'highest_fidelities.png'), bbox_inches='tight', dpi=300)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LOOKUP  —  enter p and N to display fidelity + parameters
# ─────────────────────────────────────────────────────────────────────────────

all_Params = np.load(os.path.join(data_dir, 'all_parameters.npy'), allow_pickle=True).item()

while True:
    try:
        p_in = int(input("\nEnter pulse number p (3–6, or 0 to quit): "))
    except ValueError:
        print("  Please enter an integer.")
        continue

    if p_in == 0:
        break

    if p_in not in range(3, 7):
        print("  p must be 3, 4, 5, or 6.")
        continue

    try:
        N_in = int(input("Enter target Fock state N (1–4): "))
    except ValueError:
        print("  Please enter an integer.")
        continue

    if N_in not in range(1, 5):
        print("  N must be 1, 2, 3, or 4.")
        continue

    fidelity = all_Fids[p_in - 3, N_in - 1]
    print(f"\n  p = {p_in},  N = {N_in}")
    print(f"  Fidelity : {fidelity:.6f}")

    if (p_in, N_in) in all_Params:
        params = all_Params[(p_in, N_in)]          # raw parameters (2p-1,)
        r_vals = to_dB(params[0::2])               # even indices → r in dB
        t_vals = params[1::2] / (2 * np.pi)        # odd  indices → t in units of 2π
        print(f"  Parameters [normal units] {params}):")
        print(f"  Parameters (2p-1 = {len(params)}):")
        print(f"    r  [dB]   : {np.array2string(r_vals, precision=4, separator=', ')}")
        print(f"    t  [2π]   : {np.array2string(t_vals, precision=4, separator=', ')}")
    else:
        print("  (No parameter data stored for this combination.)")
