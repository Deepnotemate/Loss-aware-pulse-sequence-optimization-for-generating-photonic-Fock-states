import os
import numpy as np

from utils import Loss_Aware_Optimization

# ─── Configuration ────────────────────────────────────────────────────────────
Omega = 1.0
dim = 60
cutoff = 3
dt_val = 0.01
loss_type = 'atom' # 'cavity' or 'atom'
gamma = 0.05 if loss_type == 'atom' else 0.03
target_N = 1
p = 4
# ──────────────────────────────────────────────────────────────────────────────

opti = Loss_Aware_Optimization(Omega, dim, cutoff, dt_val, loss_type, gamma, target_N)


# --- Load parameters ---
_BASE         = os.path.join(os.path.dirname(__file__), '..')
_UNITARY_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'unitary_optimization')

# Load loss-aware optimal parameters
N1p4_LME = np.load(os.path.join(_BASE, 'data', 'manuscript', 'optimal_parameters',
                                 f'{loss_type}_decay_N={target_N}_gamma={0.05 if loss_type == "atom" else 0.01}.npy'),
                   allow_pickle=True) # N=1 optimiziation under cavity decay was optimized at gamma=0.01

# Load unitary optimal params [r1,t1,...] and interleave alternating phases [0,π,0,π,...]
_u_params = np.load(os.path.join(_UNITARY_BASE, 'data', 'manuscript', 'optimal_configurations',
                                  'all_parameters.npy'), allow_pickle=True).item()[(p, target_N)]
_phases = ([0, np.pi] * p)[:p]
N1p4_U_optimal = np.empty(3 * p - 1)
N1p4_U_optimal[0::3] = _u_params[0::2]   # r_i
N1p4_U_optimal[1::3] = _phases            # phi_i
N1p4_U_optimal[2::3] = _u_params[1::2]   # t_i


# --- append extra values for fith pulse with r_5 = 0 for plotting ---
if loss_type == 'atom':
    fintime = 2.4 # total interaction time
elif loss_type == 'cavity':
    fintime = 3.0 # total interaction time
t4_LME = fintime*2*np.pi-np.sum(N1p4_LME[2::3])
params_LME =  np.array([N1p4_LME.tolist() + [t4_LME, 0, 0]])[0]
t4_U = fintime*2*np.pi-np.sum(N1p4_U_optimal[2::3])
params_U =  np.array([N1p4_U_optimal.tolist() + [t4_U, 0, 0]])[0] 

# --- create unitary squeezing operators ---
r_params_LME, phi_params_LME, t_params_LME = params_LME[0::3], params_LME[1::3], np.cumsum(np.append(0, params_LME[2::3])) # cumsum, because we need to have actual times in dynamics (not delays between times)
U_NLs_LME = [opti.create_U_NL(r_params_LME[i], phi_params_LME[i]) for i in range(len(r_params_LME))]
r_params_U, phi_params_U, t_params_U = params_U[0::3], params_U[1::3], np.cumsum(np.append(0, params_U[2::3])) 
U_NLs_U = [opti.create_U_NL(r_params_U[i], phi_params_U[i]) for i in range(len(r_params_U))]

import time 
t1 = time.time() 
# --- calculate dynamics --- 
print(f'Calculating dynamics for loss-aware optimum and unitary optimum for {loss_type} decay with loss rate gamma={gamma} ...')
timelist_LME, diags_LME, rho_p_LME = opti.dynamics_all_time_steps(params_LME, U_NLs_LME, t_params_LME)
timelist_U, diags_U, rho_p_U = opti.dynamics_all_time_steps(params_U, U_NLs_U, t_params_U)
t2 = time.time() 
print(f'Dynamics calculation took {t2-t1:.2f} seconds.') 

# --- save diagonal elements of density matrices over time for plotting ---
save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'composite_state_dynamics', f'{loss_type}_decay')
os.makedirs(save_dir, exist_ok=True)

print('Saving data...')
np.save(os.path.join(save_dir, 'timelist_LME'), np.array(timelist_LME, dtype=np.float64))
np.save(os.path.join(save_dir, 'diags_LME'),    np.array(diags_LME,    dtype=np.float64))
np.save(os.path.join(save_dir, 'timelist_U'),   np.array(timelist_U,   dtype=np.float64))
np.save(os.path.join(save_dir, 'diags_U'),      np.array(diags_U,      dtype=np.float64))
np.save(os.path.join(save_dir, 'Fidelity_LME'), opti.calculate_Fidelity(rho_p_LME))
np.save(os.path.join(save_dir, 'Fidelity_U'),   opti.calculate_Fidelity(rho_p_U))

