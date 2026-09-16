import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Unitary_Optimization


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  only change things here
# ─────────────────────────────────────────────────────────────────────────────

p         = 5     # number of pulses
target_N  = 1     # Fock state |N⟩ we want to prepare
num_tests = 5      # number of highest-fidelity configurations to run convergence tests on

# Hilbert-space truncation dimensions used for the convergence test
# (optimization was run at dim=60; we check whether that was sufficient)
dims = np.arange(30,330,30)


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_IN_DIR  = Path(__file__).parent.parent / "data" / "auxiliary" / "optimization_results"
FIG_DIR      = Path(__file__).parent.parent / "figures" / "auxiliary" / "convergence_of_optimal_results"
DATA_OUT_DIR = Path(__file__).parent.parent / "data" / "manuscript" / "optimal_configurations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
#    fin_Probs: shape (100,), final fidelity of each run.
#    Parms: shape (2p-1, 100), final optimized parameter vectors.
# ─────────────────────────────────────────────────────────────────────────────

fin_probs = np.load(DATA_IN_DIR / f'Probs_p{p}N{target_N}.npy')   # shape (100,)
all_parms = np.load(DATA_IN_DIR / f'Parms_p{p}N{target_N}.npy')        # shape (2p-1, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SELECT TOP-num_tests CONFIGURATIONS
#    Sort by final fidelity (ascending) and keep the best num_tests runs.
# ─────────────────────────────────────────────────────────────────────────────

top_idx   = np.argsort(fin_probs)[-num_tests:]   # indices of best runs (ascending)
top_probs = fin_probs[top_idx]                   # shape (num_tests,)
top_parms = all_parms[:, top_idx]                # shape (2p-1, num_tests)

print(f'Top-{num_tests} fidelities at dim=60 for p={p}, N={target_N}:')
for i, (idx, prob) in enumerate(zip(top_idx, top_probs)):
    print(f'  [{i}] run {idx:3d}  F = {prob:.6f}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVERGENCE TEST
#    Re-evaluate each of the top-num_tests parameter sets at increasing
#    Hilbert-space truncation dimensions to check whether the fidelity
#    stabilizes (i.e. dim=60 was sufficient during optimization).
# ─────────────────────────────────────────────────────────────────────────────

def convergence_test(p, target_N, params, dims):
    """Evaluate the fidelity of a fixed parameter set at multiple truncation dims."""
    fixed_phases = [0, 1] * (p // 2)
    if p % 2 != 0:
        fixed_phases.append(fixed_phases[-2])
    fidelities = []
    for dim in dims:
        # num_iterations=0: no further optimization, just evaluate the fidelity
        opti = Unitary_Optimization(1, dim, fixed_phases, 'Adam', [0, 0, 0],
                            p, target_N, num_iterations=0, verbose=False)
        fidelities.append(opti.calculate_Fidelity(params, params, True))
    return fidelities

conv_fidelities = []
for j in range(num_tests):
    print(f'Running convergence test {j + 1}/{num_tests} ...')
    conv_fidelities.append(convergence_test(p, target_N, top_parms[:, j], dims))

conv_fidelities = np.array(conv_fidelities)   # shape: (num_tests, len(dims))


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT CONVERGENCE
#    Skip unphysical configurations (negative interaction times → tot_t = -1).
#    Label: run index, fidelity from optimization, fidelity at largest dim.
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots()

for i in range(num_tests):
    t_vals = top_parms[:, i][1::2]   # interaction times (odd indices)
    if (t_vals < 0).any():
        print(f'  Skipping config [{i}]: unphysical (negative interaction time).')
        continue
    ax.plot(dims, conv_fidelities[i], 'o-',
            label=(f'[{i}]  F_opt={top_probs[i]:.4f},  '
                   f'F(dim={dims[-1]})={conv_fidelities[i, -1]:.4f}'))

ax.set_title(f'Convergence test  —  p={p}, N={target_N}')
ax.set_xlabel('Truncation dimension d')
ax.set_ylabel('Fidelity')
ax.legend()
fig.tight_layout()

fig_path = FIG_DIR / f'convergence_p{p}N{target_N}.svg'
fig.savefig(fig_path)
plt.show()
print(f'Saved figure → {fig_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. IDENTIFY BEST CONFIGURATION AFTER CONVERGENCE TEST
#    The best configuration is the one with the highest fidelity at the
#    largest truncation dimension (dim = dims[-1]).
# ─────────────────────────────────────────────────────────────────────────────

conv_fids_final = conv_fidelities[:, -1]     # fidelity at dim=dims[-1], shape (num_tests,)

# Mask out unphysical configurations (negative interaction times) before picking best
physical_mask = np.array([(top_parms[:, i][1::2] >= 0).all() for i in range(num_tests)])
if not physical_mask.any():
    raise RuntimeError('All top configurations have negative interaction times — no valid result to save.')

masked_fids     = np.where(physical_mask, conv_fids_final, -np.inf)
best_idx        = int(np.argmax(masked_fids))
best_fidelity   = conv_fids_final[best_idx]
best_params     = top_parms[:, best_idx]

print(f'\nBest configuration after convergence test:')
print(f'  Index in top-{num_tests}         : {best_idx}')
print(f'  F_opt  (dim=60)         : {top_probs[best_idx]:.6f}')
print(f'  F_conv (dim={dims[-1]})       : {best_fidelity:.6f}')
print(f'  Parameters              : {best_params}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE RESULTS TO data/paper/optimal_configurations/
#
#    all_fidelities.npy  — shape (4, 4), rows = p ∈ {3,4,5,6}, cols = N ∈ {1,2,3,4}
#                          row 0 (p=3) holds reference values from the PRL paper.
#    all_parameters.npy  — dict keyed by (p, N), value = best parameter vector
# ─────────────────────────────────────────────────────────────────────────────

# PRL paper reference fidelities for p=3, N=1..4
prl_paper_fidelities = [0.98306544, 0.92511628, 0.84335183, 0.74493255]

ALL_FID_PATH   = DATA_OUT_DIR / 'all_fidelities.npy'
ALL_PARMS_PATH = DATA_OUT_DIR / 'all_parameters.npy'

# Load or initialize the fidelity matrix
try:
    all_fid = np.load(ALL_FID_PATH)
    if all_fid.shape != (4, 4):
        all_fid = np.full((4, 4), np.nan)
except FileNotFoundError:
    all_fid = np.full((4, 4), np.nan)

all_fid[0, :]                 = prl_paper_fidelities   # row 0 = p=3 (PRL reference)
all_fid[p - 3, target_N - 1] = best_fidelity
np.save(ALL_FID_PATH, all_fid)
print(f'\nSaved best fidelity {best_fidelity:.6f} → {ALL_FID_PATH}  [row {p - 3}, col {target_N - 1}]')

# Load or initialize the parameter dict
try:
    all_parms_dict = np.load(ALL_PARMS_PATH, allow_pickle=True).item()
except FileNotFoundError:
    all_parms_dict = {}

all_parms_dict[(p, target_N)] = best_params
np.save(ALL_PARMS_PATH, all_parms_dict)
print(f'Saved best params          → {ALL_PARMS_PATH}  key=({p}, {target_N})')
