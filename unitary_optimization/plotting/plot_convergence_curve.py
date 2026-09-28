import numpy as np
import matplotlib.pyplot as plt
import os
### we put ylabel manually here

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  only change things here
# ─────────────────────────────────────────────────────────────────────────────

p        = 4      # pulse number (4, 5, or 6)
target_N = 2      # Fock state |N⟩
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimization_histories')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
#   Requires run_optimization.py to have been run with save_probs_histories = True
#   (saves per-iteration histories to data/manuscript/optimization_histories).
#   fin_Probs    : final fidelity of each of the 100 runs  — shape (100,)
#   best_history : full fidelity history of the best run   — shape (iterations,)
# ─────────────────────────────────────────────────────────────────────────────

probs_histories = np.load(os.path.join(DATA_DIR, f'Probs_histories_p{p}N{target_N}.npy'), allow_pickle=True)
fin_Probs    = np.array([h[-1] for h in probs_histories])
sorter = np.argsort(fin_Probs) 

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
#   (a) sorted final fidelities over all 100 initializations
#   (b) infidelity  1 - F(iter)  of the best run
# ─────────────────────────────────────────────────────────────────────────────

fontsize, labelsize = 25, 20
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=350)

# (a) sorted fidelities
y = fin_Probs[sorter]
x = np.arange(1, y.size + 1)
ax[0].plot(x, y, 'o', markersize=5)
ax[0].set_xlim(0, y.size + 1)
ax[0].set_xticks([1, 25, 50, 75, 100])
ax[0].set_xlabel(r'Initializations $\vec{\theta}_i$', fontsize=fontsize)
ax[0].set_ylabel(r'$\langle 2|\hat{\rho}_\mathrm{s}|2\rangle$', fontsize=fontsize)
ax[0].tick_params(axis='x', labelsize=labelsize)
ax[0].tick_params(axis='y', labelsize=labelsize)

# (b) infidelity curve of the best run
best_history = np.array(probs_histories[sorter[-1]])
ax[1].plot(np.arange(1, len(best_history) + 1), 1 - best_history, color='darkgray', linewidth=4)
ax[1].set_yscale('log')
ax[1].set_xlabel('Iterations', fontsize=fontsize)
ax[1].set_ylabel(r'$\mathcal{I}(\vec{\theta})$', fontsize=fontsize)
ax[1].tick_params(axis='x', labelsize=labelsize)
ax[1].tick_params(axis='y', labelsize=labelsize)

# subfigure labels
for ax_, label in zip(ax.flat, ['(a)', '(b)']):
    ax_.text(-0.232, 1., label, transform=ax_.transAxes,
             fontsize=20, fontweight='bold', va='top')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, f'convergence_p{p}N{target_N}.png'), bbox_inches='tight', dpi=300)
plt.show()
