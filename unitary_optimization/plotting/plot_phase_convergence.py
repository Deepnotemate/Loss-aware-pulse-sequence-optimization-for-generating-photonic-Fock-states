import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os


def to_dB(r):
    """Convert squeezing parameter r (natural units) to dB."""
    return -10 * np.log10(np.exp(-2 * r))


p        = 4      # pulse number
target_N = 2       # Fock state |N⟩
epsilon  = 0.01    # tolerance [rad] for a phase difference to count as converged to a multiple of pi
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimization_histories')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
#   Requires run_optimization.py to have been run with optimize_phases = True
#   and save_probs_histories = save_params_histories = True (appends the
#   "_with_optimized_phases" tag and saves per-iteration histories, not finals,
#   to data/manuscript/optimization_histories).
#   fin_Parms : final parameter vector of each of the 100 runs, shape (3p-1, 100)
#   fin_Probs : final fidelity of each of the 100 runs,          shape (100,)
# ─────────────────────────────────────────────────────────────────────────────

probs_histories = np.load(os.path.join(DATA_DIR, f'Probs_histories_p{p}N{target_N}_with_optimized_phases.npy'),
                           allow_pickle=True)
parms_histories = np.load(os.path.join(DATA_DIR, f'Parms_histories_p{p}N{target_N}_with_optimized_phases.npy'),
                           allow_pickle=True)

fin_Probs = np.array([h[-1] for h in probs_histories])          # last entry of each run's history
fin_Parms = np.array([h[-1] for h in parms_histories]).T        # shape (3p-1, 100)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE-DIFFERENCE ANALYSIS
#   Parameter layout per run: [r1, phi1, t1, r2, phi2, t2, ..., r_p, phi_p],
#   so the p phases sit at every 3rd entry starting at index 1. For each run we
#   compute the p-1 consecutive phase differences and how far each one deviates
#   (mod pi) from the nearest multiple of pi. A run is flagged as "converged to
#   a binary-phase pattern" if all its deviations are below `epsilon`.
# ─────────────────────────────────────────────────────────────────────────────

phi_params       = fin_Parms[1::3, :]                             # shape (p, 100)
phase_diffs      = np.diff(phi_params, axis=0)                    # shape (p-1, 100), raw phase differences
diffs_mod_pi     = phase_diffs % np.pi                            # each difference reduced into [0, pi)
deviation        = np.minimum(diffs_mod_pi, np.pi - diffs_mod_pi) # distance of each diff to the nearer of 0 or pi

converged = np.all(deviation < epsilon, axis=0)   # True for a run only if ALL its p-1 deviations are < epsilon
print(f'{converged.sum()} / {converged.size} runs converged to a binary-phase pattern '
      f'(all phase differences within {epsilon} rad of a multiple of pi)')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
#   (a) Scatter plot of the sorted final fidelities, highlighted if the run
#       converged to a binary-phase pattern.
#   (b) Phase-parameter trajectories phi_1, ..., phi_p over iterations for a
#       selected configuration.
# ─────────────────────────────────────────────────────────────────────────────

sorter           = np.argsort(fin_Probs)
y                = fin_Probs[sorter]
converged_sorted = converged[sorter]

# count how many of the best runs (from the top, descending fidelity) are ALL converged
# before hitting the first non-converged run, i.e. where the streak breaks
from_top = converged_sorted[::-1]
break_idx = np.argmin(from_top) if not from_top.all() else from_top.size
print(f'Among the best {break_idx} / {from_top.size} runs, all converged to a binary-phase pattern')


selected_idx    = sorter[-2]                                # run index shown in (b); change to inspect another run
selected_parms  = np.array(parms_histories[selected_idx])   # shape (iterations, 3p-1)
selected_phases = selected_parms[:, 1::3]                   # shape (iterations, p)
iterations      = np.arange(1, selected_phases.shape[0] + 1)

selected_final_parms = fin_Parms[:, selected_idx]           # final parameter vector of the selected run, shape (3p-1,)
r_vals   = to_dB(selected_final_parms[0::3])                # r in dB
phi_vals = selected_final_parms[1::3]                        # phases [rad]
t_vals   = selected_final_parms[2::3] / (2 * np.pi)          # interaction times [2π]

print(f'\nSelected run (fidelity = {fin_Probs[selected_idx]:.6f}) parameters:')
print(f'  r   [dB] : {np.array2string(r_vals, precision=4, separator=", ")}')
print(f'  phi [rad]: {np.array2string(phi_vals, precision=4, separator=", ")}')
print(f'  t   [2π] : {np.array2string(t_vals, precision=4, separator=", ")}')

fontsize, labelsize = 25, 20
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=350)

# (a) sorted final fidelities
x = np.arange(1, y.size + 1)
ax[0].plot(x[~converged_sorted], y[~converged_sorted], 'o', markersize=5, color='tab:blue')
ax[0].plot(x[converged_sorted],  y[converged_sorted],  'o', markersize=5, color='tab:orange')
ax[0].set_xlim(0, y.size + 1)
ax[0].set_xticks([1, 25, 50, 75, 100])
ax[0].set_xlabel(r'Initializations $\vec{\theta}_i$', fontsize=fontsize)
ax[0].set_ylabel(r'$\langle 2|\hat{\rho}_\mathrm{s}|2\rangle$', fontsize=fontsize)
ax[0].tick_params(axis='x', labelsize=labelsize)
ax[0].tick_params(axis='y', labelsize=labelsize)

legend_elements = [
    Patch(facecolor='tab:blue', label=r'arbitrary $\Delta\phi_j$'),
    Patch(facecolor='tab:orange', label=r'$\Delta\phi_j \in \{0, \pi\}$'),
]
ax[0].legend(handles=legend_elements, fontsize=16, loc='upper left')

# (b) phase-parameter trajectories of the selected run
for j in range(selected_phases.shape[1]):
    ax[1].plot(iterations, selected_phases[:, j] / np.pi, linewidth=3, label=fr'$\phi_{j + 1}$')
ax[1].set_xlabel('Iterations', fontsize=fontsize)
ax[1].set_ylabel(r'$\phi_j\ /\ \pi$', fontsize=fontsize)
ax[1].tick_params(axis='x', labelsize=labelsize)
ax[1].tick_params(axis='y', labelsize=labelsize)
ax[1].legend(fontsize=16)

for ax_, label in zip(ax.flat, ['(a)', '(b)']):
    ax_.text(-0.232, 1., label, transform=ax_.transAxes,
              fontsize=20, fontweight='bold', va='top')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, f'phase_convergence_p{p}N{target_N}.png'), bbox_inches='tight', dpi=300)
plt.show()