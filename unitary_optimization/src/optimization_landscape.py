import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import Unitary_Optimization


# --- Load parameters ---
all_Params = np.load(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimal_configurations', 'all_parameters.npy'),
    allow_pickle=True).item()

p_in, N_in = 4, 2
params_no_phases = all_Params[(p_in, N_in)]

# Convert to [r1, phi1, t1, r2, phi2, t2, ...] format
phases = [np.pi, 0, 0, np.pi]
params = np.zeros(3 * p_in - 1)
params[::3]  = np.abs(params_no_phases[::2])
params[1::3] = phases
params[2::3] = params_no_phases[1::2]

# --- Set up optimizer ---
dim = 100
pulse_num = 4
target_N = 2
Omega, optimizer, hyperparams, num_iterations = 1, 'Adam', [0.02, 0.9, 0.999], 600

opti = Unitary_Optimization(Omega, dim, False, optimizer, hyperparams, pulse_num, target_N, num_iterations, True)

# --- Compute optimization landscape slices ---
r_var   = 2.5
t_var   = np.pi * 2 * 3
phi_var = np.pi
num_points_r, num_points_phi, num_points_t = 301, 301, 301

print('Computing optimization landscape slices...')
slices, ranges = opti.loss_landscape(params, r_var, t_var, phi_var,
                                     num_points_r, num_points_phi, num_points_t)

# --- Save ---
out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimization_landscape')
os.makedirs(out_dir, exist_ok=True)

np.savez(
    os.path.join(out_dir, 'optimization_landscape.npz'),
    slices    = np.array(slices),
    ranges    = np.array(ranges),
    params    = params,
    target_N  = np.array(target_N),
)

print(f"Saved optimization_landscape.npz to {out_dir}")
