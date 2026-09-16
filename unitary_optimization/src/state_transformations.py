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
params = all_Params[(p_in, N_in)]

# --- Set up optimizer ---
dim = 100
pulse_num = 4
target_N = 2
Omega, optimizer, hyperparams, num_iterations = 1, 'Vanilla GD', [0.02, 0.9, 0.999], 600
fixed_phases = [1, 0, 1, 0]

opti = Unitary_Optimization(Omega, dim, fixed_phases, optimizer, hyperparams, pulse_num, target_N, num_iterations, True)

# --- Compute state sequence ---
Units = opti.create_unitaries(params)[0]
state = np.eye(2 * dim)[0]
states = [np.abs(state) ** 2]
for U in Units:
    state = U @ state
    states.append(np.abs(state) ** 2)
states = np.array(states)

# --- Split into ground / excited components: shape (8, 2, 2*dim) ---
# axis 0: state index 0-7 | axis 1: 0=ground, 1=excited | axis 2: vector
states_ge = np.zeros((8, 2, 2 * dim))
for j in range(8):
    states_ge[j, 0, ::2]  = states[j, ::2]   # ground
    states_ge[j, 1, 1::2] = states[j, 1::2]  # excited

# --- Compute display parameters (r in dB, t, effective phases) ---
def to_decibel_converter(r):
    return -10 * np.log10(np.exp(-2 * r))

r_raw  = params[::2].copy()   # [r1, r2, r3, r4]
t_vals = params[1::2].copy()  # [t1, t2, t3]
phases = list(fixed_phases)   # 0 = phase 0, 1 = phase pi
for i in range(len(r_raw)):
    if r_raw[i] < 0:
        r_raw[i]  = -r_raw[i]
        phases[i] = (phases[i] + 1) % 2
r_dB = to_decibel_converter(r_raw)

# --- Save ---
out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'state_transformations')
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, 'states_ge.npy'), states_ge)
np.save(os.path.join(out_dir, 'r_dB.npy'),      r_dB)
np.save(os.path.join(out_dir, 't_vals.npy'),     t_vals)
np.save(os.path.join(out_dir, 'phases.npy'),     np.array(phases))

print(f"Saved states_ge {states_ge.shape}, r_dB={np.round(r_dB,2)}, t={np.round(t_vals,2)}, phases={phases} to {out_dir}")
