import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import Unitary_Optimization


DATA_DIR = Path(__file__).parent.parent / "data" / "auxiliary" / "optimization_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parse command-line arguments: python run_optimization.py p=4 N=2
_cli = {k: int(v) for k, v in (arg.split('=') for arg in sys.argv[1:])}
p              = _cli.get('p', 4)    # number of pulses
target_N       = _cli.get('N', 2)   # Fock state |N⟩ we want to prepare
num_iterations = 600
save_probs_histories = False  # if False, only final fidelities are saved

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE RANDOM INITIAL PARAMETERS
#    Each run starts from a different random initial point to improve the
#    chance of finding the global optimum.
#    Parameter layout per run: [r1, t1, r2, t2, ..., r_p]
#      rᵢ ∈ [0, 1.7]   — squeezing amplitudes
#      tᵢ ∈ [0, 2π]    — interaction times between pulses
# ─────────────────────────────────────────────────────────────────────────────

init_parameters = []
for i in range(100):
    r_vals = np.random.uniform(0, 1.7, p)           # squeezing amplitudes
    phi_vals = np.random.uniform(0, 2*np.pi, p)     # phases 
    t_vals = np.random.uniform(0, 2 * np.pi, p - 1) # interaction times

    # Interleave r and t: [r1, t1, r2, t2, ..., r_p]
    params = np.empty(2 * p - 1)
    params[::2]  = r_vals
    params[1::2] = t_vals 
    init_parameters.append(params)

inits = np.array(init_parameters)   # shape: (100, 2p-1)
print(f'shape of initial parameters: {inits.shape}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN GRADIENT DESCENT
#    For each of the 100 random initializations, run Adam gradient descent.
#    A run stops early if the gradient norm drops below `stopping_condition`.
# ─────────────────────────────────────────────────────────────────────────────

def Run_Optimization(opti, inits):
    """Run gradient descent from each initial point; return final params and fidelity histories."""
    stopping_condition = 0.01e-3 
    parms    = []   # final optimised parameter vector per run
    Probs    = []   # full fidelity history (or final value) per run
    stps_met = []   # whether stopping condition was met
    for i in tqdm(range(100)):
        p, P, _, stp_cnd_met = opti.run_gradient_descent(inits[i, :], stopping_condition,
                                                        save_Probs=save_probs_histories, save_params=False)
        parms.append(p)
        Probs.append(P if save_probs_histories else opti.calculate_Fidelity(None, p, from_params=True))
        stps_met.append(stp_cnd_met)
    return parms, Probs, stps_met


# ─────────────────────────────────────────────────────────────────────────────
# 3. SET UP OPTIMIZER
#    Phases are fixed alternating [0, π, 0, π, ...] (even index → 0, odd → π).
#    dim=60: Hilbert-space truncation (signal × idler × atom).
# ─────────────────────────────────────────────────────────────────────────────

dim         = 60
Omega       = 1                       # single-photon Rabi frequency
optimizer   = 'Adam'
hyperparams = [0.02, 0.9, 0.999]      # [learning rate, β1, β2]

fixed_phases = [0, 1] * (p // 2)     # alternating 0/π phases
if p % 2 != 0:
    fixed_phases.append(fixed_phases[-2])   # extra phase for odd p

opti = Unitary_Optimization(Omega, dim, fixed_phases, optimizer, hyperparams,
                    p, target_N, num_iterations, verbose=True)

parms, Probs, _ = Run_Optimization(opti, inits)


# ─────────────────────────────────────────────────────────────────────────────
# 4. COLLECT RESULTS
#    Store the full fidelity history of every run and the final parameter
#    vector of each run.
# ─────────────────────────────────────────────────────────────────────────────

fin_Parms = np.zeros((2 * p - 1, 100))
for i in range(100):
    fin_Parms[:, i] = parms[i]

if save_probs_histories:
    # Probs is a list of arrays with potentially different lengths (early stopping),
    # so it must be saved as an object array.
    np.save(DATA_DIR / f'Probs_histories_p{p}N{target_N}',
            np.array(Probs, dtype=object), allow_pickle=True)
else:
    np.save(DATA_DIR / f'Probs_p{p}N{target_N}', np.array(Probs))
np.save(DATA_DIR / f'Parms_p{p}N{target_N}', fin_Parms)

print(f'Saved results for p={p}, N={target_N}')
