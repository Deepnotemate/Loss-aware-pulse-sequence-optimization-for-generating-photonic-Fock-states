import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import Unitary_Optimization

DATA_DIR = Path(__file__).parent.parent / "data" / "auxiliary" / "optimization_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR = Path(__file__).parent.parent / "data" / "manuscript" / "optimization_histories"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  only change things here
# ─────────────────────────────────────────────────────────────────────────────

# Parse command-line arguments: python run_optimization.py p=4 N=2
_cli = {k: int(v) for k, v in (arg.split('=') for arg in sys.argv[1:])}
p              = _cli.get('p', 4)    # number of pulses
target_N       = _cli.get('N', 2)    # Fock state |N⟩ we want to prepare
num_iterations = 600                 # maximum number of iterations per initialization

optimize_phases        = False  # if True, phases are randomly initialized and optimized jointly with r, t; otherwise, phases are fixed
save_probs_histories   = False  # if False, only final fidelities are saved
save_params_histories  = False  # if False, only final parameters are saved

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE RANDOM INITIAL PARAMETERS
#    Each run starts from a different random initial point to improve the
#    chance of finding the global optimum.
#    Parameter layout per run (fixed phases):     [r1, t1, r2, t2, ..., r_p]
#    Parameter layout per run (randomly initialized phases): [r1, phi1, t1, r2, phi2, t2, ..., r_p, phi_p]
#      rᵢ ∈ [0, 1.7]   — squeezing amplitudes
#      tᵢ ∈ [0, 2π]    — interaction times between pulses
#      phiᵢ ∈ [0, 2π]  — pulse phases (only if optimize_phases)
# ─────────────────────────────────────────────────────────────────────────────

init_parameters = []
for i in range(100):
    r_vals = np.random.uniform(0, 1.7, p)           # squeezing amplitudes
    phi_vals = np.random.uniform(0, 2*np.pi, p)     # phases 
    t_vals = np.random.uniform(0, 2 * np.pi, p - 1) # interaction times

    if optimize_phases:
        # Interleave r, phi and t: [r1, phi1, t1, r2, phi2, t2, ..., r_p, phi_p]
        params = np.empty(3 * p - 1)
        params[0::3] = r_vals
        params[1::3] = phi_vals
        params[2::3] = t_vals
    else:
        # Interleave r and t: [r1, t1, r2, t2, ..., r_p]
        params = np.empty(2 * p - 1)
        params[::2]  = r_vals
        params[1::2] = t_vals
    init_parameters.append(params)

inits = np.array(init_parameters)   # shape: (100, 3p-1) if optimize_phases else (100, 2p-1)
print(f'shape of initial parameters: {inits.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN GRADIENT DESCENT
#    For each of the 100 random initializations, run Adam gradient descent.
#    A run stops early if the gradient norm drops below `stopping_condition`.
# ─────────────────────────────────────────────────────────────────────────────

def Run_Optimization(opti, inits):
    """Run gradient descent from each initial point; return per-run parameters
    (full history if save_params_histories else final vector only), per-run
    fidelities (full history if save_probs_histories else final value only)
    and per-run stopping flags."""
    stopping_condition = 0.01e-3 
    parms    = []   # per run: parameter history over iterations if save_params_histories else final parameter vector 
    probs    = []   # per run: fidelity history over iterations if save_probs_histories else final fidelity
    stps_met = []   # per run: whether stopping condition was met
    for i in tqdm(range(100)):
        p_final, probability_history, parameters, stp_cnd_met = opti.run_gradient_descent(
            inits[i, :], stopping_condition,
            save_Probs=save_probs_histories, save_params=save_params_histories)
        parms.append(parameters if save_params_histories else p_final)
        probs.append(probability_history if save_probs_histories else opti.calculate_Fidelity(None, p_final, from_params=True))
        stps_met.append(stp_cnd_met)
    return parms, probs, stps_met


# ─────────────────────────────────────────────────────────────────────────────
# 3. SET UP OPTIMIZER
#    If optimize_phases is True, phases are free parameters, optimized jointly
#    with r and t. Otherwise, phases are fixed, alternating [0, π, 0, π, ...]
#    (even index → 0, odd → π).
#    dim=60: Hilbert-space truncation (signal × idler × atom).
# ─────────────────────────────────────────────────────────────────────────────

dim         = 60
Omega       = 1                       # single-photon Rabi frequency
optimizer   = 'Adam'
hyperparams = [0.02, 0.9, 0.999]      # [learning rate, β1, β2]

if optimize_phases:
    fixed_phases = False    # phases are treated as free optimization parameters
else:
    fixed_phases = [0, 1] * (p // 2)     # alternating 0/π phases
    if p % 2 != 0:
        fixed_phases.append(fixed_phases[-2])   # extra phase for odd p

opti = Unitary_Optimization(Omega, dim, fixed_phases, optimizer, hyperparams,
                    p, target_N, num_iterations, verbose=True)

parms, probs, _ = Run_Optimization(opti, inits)


# ─────────────────────────────────────────────────────────────────────────────
# 4. COLLECT RESULTS
#    save_probs_histories:   True  -> save fidelites over iterations, per run
#                            False -> save final fidelity per run
#    save_params_histories:  True  -> save parameters over iterations, per run
#                            False -> save final parameters per run
# ─────────────────────────────────────────────────────────────────────────────

if optimize_phases:
    phase_tag = '_with_optimized_phases'   # marks filenames of runs with phases optimized as free parameters
else:
    phase_tag = ''

if save_probs_histories:
    # probs is a list of arrays with potentially different lengths (early stopping),
    # so it must be saved as an object array.
    np.save(HISTORY_DIR / f'Probs_histories_p{p}N{target_N}{phase_tag}',
            np.array(probs, dtype=object), allow_pickle=True)
else:
    # probs is a list of final fidelities per run
    np.save(DATA_DIR / f'Probs_p{p}N{target_N}{phase_tag}', np.array(probs))


if save_params_histories:
    # parms holds one parameter history per run (ragged due to early stopping) -> object array
    np.save(HISTORY_DIR / f'Parms_histories_p{p}N{target_N}{phase_tag}',
            np.array(parms, dtype=object), allow_pickle=True)
else:
    # parms holds the final parameter vector per run
    np.save(DATA_DIR / f'Parms_p{p}N{target_N}{phase_tag}', np.array(parms).T)

print(f'Saved results for p={p}, N={target_N}{phase_tag}')



