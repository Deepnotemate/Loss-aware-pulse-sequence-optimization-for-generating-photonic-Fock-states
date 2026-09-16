import glob
import os
import numpy as np

from utils import Loss_Aware_Optimization

# ─── Configuration ────────────────────────────────────────────────────────────
Omega = 1.0
dim = 60
cutoff = 3
dt_val = 0.01
loss_type = 'cavity' # 'cavity' or 'atom'
target_N = 2 # 1 or 2
p = 4
# ──────────────────────────────────────────────────────────────────────────────

opti = Loss_Aware_Optimization(Omega, dim, cutoff, dt_val, loss_type, 0, target_N)

_BASE         = os.path.join(os.path.dirname(__file__), '..')
_UNITARY_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'unitary_optimization')

# --- PRL 3-pulse parameters ---
if target_N == 1:
    prl_params = np.array([0.54801525, 0, 6.97433569, 1.48056221, np.pi, 1.19380521, 1.42645147, 0])
elif target_N == 2:
    prl_params = np.array([0.93254696, 0, 8.85929128, 1.48056221, np.pi, 2.136283, 1.26181663, 0])

# --- Load unitary optimal parameters ---
_u_params = np.load(os.path.join(_UNITARY_BASE, 'data', 'manuscript', 'optimal_configurations',
                                  'all_parameters.npy'), allow_pickle=True).item()[(p, target_N)]
_phases = ([0, np.pi] * p)[:p]
U_params = np.empty(3 * p - 1)
U_params[0::3] = _u_params[0::2]  # r_i
U_params[1::3] = _phases          # phi_i
U_params[2::3] = _u_params[1::2]  # t_i

# --- Load loss-aware optimal parameters (all gamma variants for this loss_type and N) ---
_lme_files = sorted(glob.glob(os.path.join(_BASE, 'data', 'manuscript', 'optimal_parameters',
                                            f'{loss_type}_decay_N={target_N}_gamma=*.npy')))
LME_params_list = [np.load(f, allow_pickle=True) for f in _lme_files]

param_configs = [prl_params, U_params] + LME_params_list

# --- decay rates ###
gamma_array = np.linspace(0.00, 0.05, 15)
print(f'Scanning dissipation rates for N={target_N}, loss_type={loss_type} decay ...')

Fs = [] # container for fidelity-arrays for each parameter configuration
for params in param_configs:
    fidelities = []
    for gamma in gamma_array:
        opti.gamma = gamma
        r_params, phi_params, t_params = params[0::3], params[1::3], np.cumsum(np.append(0, params[2::3]))
        U_NLs = [opti.create_U_NL(r_params[i], phi_params[i]) for i in range(len(r_params))]
        rhos = opti.dynamics_sub_unitaries(params, U_NLs, t_params)
        fidelity = opti.calculate_Fidelity(rhos[-1])
        fidelities.append(fidelity)
    Fs.append(fidelities)

print(f'Saving results ...')
save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'dissipation_rate_scans')
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, f'decay_rates_{loss_type}_decay_N={target_N}'), gamma_array)
np.save(os.path.join(save_dir, f'fidelities_{loss_type}_decay_N={target_N}'), Fs)
