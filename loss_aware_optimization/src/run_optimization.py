import os
import sys
import json
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import Loss_Aware_Optimization

# ─── Configuration ────────────────────────────────────────────────────────────
Omega          = 1.0
dim            = 60
cutoff         = 3
dt_val         = 0.01
loss_type      = 'cavity'   # 'atom' or 'cavity'
gamma          = 0.01
target_N       = 1

num_iterations = 300
no_inits       = 5
hyperparams    = [0.9, 0.999, 0.05]
stopping_cond  = 0.001
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_idx", type=int, required=True,
                        help="Starting index into the initializations array.")
    args = parser.parse_args()
    init_idx = args.init_idx

    # --- load initializations ---
    data_dir  = os.path.join(os.path.dirname(__file__), '..', 'data', 'auxiliary', 'optimization_results')
    init_file = os.path.join(data_dir, f'initializations_{loss_type}.npy')
    initializations = np.load(init_file)

    # --- build optimizer ---
    opti = Loss_Aware_Optimization(Omega, dim, cutoff, dt_val, loss_type, gamma, target_N)

    # --- run optimizations ---
    all_Probs, all_Params, stopping_conds, t_fitted_arr = [], [], [], []
    for i in range(no_inits):
        print(f'running optimization {i+1}/{no_inits}  (init_idx={init_idx + i})...')
        Probs, Params, stp_met, t_fitted = opti.single_GD_run(
            num_iterations, initializations[init_idx + i], hyperparams, stopping_cond
        )
        all_Probs.append(Probs)
        all_Params.append(Params)
        stopping_conds.append(stp_met) # array with entries True if stopping condition was met, False if max iterations reached
        t_fitted_arr.append(t_fitted) # array with entries True if any of the time delays was fitted to zero
    stopping_conds = np.array(stopping_conds) # convert to numpy array for easier saving
    t_fitted_arr   = np.array(t_fitted_arr)

    # --- save results at correct index positions ---
    out_dir = os.path.join(data_dir, f'{loss_type}_decay_N={target_N}_gamma={gamma}')
    os.makedirs(out_dir, exist_ok=True)

    # save all results as dicts {global_init_index → result}
    # loading existing file and inserting at the right key ensures that
    # parallel runs (different init_idx) accumulate correctly in one file
    files = [('Probs.pkl', all_Probs), ('Params.pkl', all_Params),
             ('stopping_conds.pkl', stopping_conds), ('t_fitted.pkl', t_fitted_arr)]
    for fname, batch in files:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
        data.update({init_idx + i: v for i, v in enumerate(batch)})
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    settings = {
        'Omega':            Omega,
        'dim':              dim,
        'cutoff':           cutoff,
        'dt_val':           dt_val,
        'loss_type':        loss_type,
        'gamma':            gamma,
        'target_N':         target_N,
        'num_iterations':   num_iterations,
        'hyperparams_Adam': hyperparams,
        'stopping_cond':    stopping_cond,
    }
    settings_path = os.path.join(out_dir, 'settings.json')
    if not os.path.exists(settings_path):
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)

    print(f'results saved to {out_dir}')