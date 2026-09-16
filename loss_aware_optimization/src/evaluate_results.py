import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Configuration -----------------------------------------------------------
loss_type = 'cavity'   # 'atom' or 'cavity'
N         = 1
gamma     = 0.01
# -----------------------------------------------------------------------------

_BASE = os.path.join(os.path.dirname(__file__), '..')


def to_decibel_converter(r):
    return -10 * np.log10(np.exp(-2 * r))


def load_results(loss_type, N, gamma):
    data_dir = os.path.join(_BASE, 'data', 'auxiliary', 'optimization_results',
                            f'{loss_type}_decay_N={N}_gamma={gamma}')

    def _pkl(fname):
        with open(os.path.join(data_dir, fname), 'rb') as f:
            return pickle.load(f)

    Probs   = _pkl('Probs.pkl')
    Params  = _pkl('Params.pkl')
    stp_cnd = _pkl('stopping_conds.pkl')
    t_fit   = _pkl('t_fitted.pkl')

    with open(os.path.join(data_dir, 'settings.json')) as f:
        settings = json.load(f)

    return Probs, Params, stp_cnd, t_fit, settings


def get_sorted_results(Probs, Params, stp_cnd, t_fit):
    """Sort runs by ascending final fidelity."""
    keys = sorted(Probs.keys())
    fin_probs, fin_params, stp_arr, tfit_arr = [], [], [], []

    for k in keys:
        hist = Probs[k]
        if len(hist) > 0:
            fin_probs.append(hist[-1])
            fin_params.append(Params[k][-1])
        else:
            fin_probs.append(-0.1)
            fin_params.append(np.full_like(Params[k][0] if len(Params[k]) > 0 else [0], np.nan))
        stp_arr.append(bool(stp_cnd[k]))
        tfit_arr.append(bool(t_fit[k]))

    fin_probs  = np.array(fin_probs)
    fin_params = np.array(fin_params, dtype=object)
    stp_arr    = np.array(stp_arr)
    tfit_arr   = np.array(tfit_arr)
    keys       = np.array(keys)

    sorter = np.argsort(fin_probs)
    return (fin_probs[sorter], fin_params[sorter],
            stp_arr[sorter], tfit_arr[sorter], keys[sorter])


def print_params(best_params, fin_probs, loss_type, N, gamma):
    r_params = np.array(best_params[::3],  dtype=float)
    t_params = np.array(best_params[2::3], dtype=float)
    r_dB     = to_decibel_converter(r_params)

    print(f'\n{"="*60}')
    print(f' Best-run parameters  |  {loss_type} decay  |  N={N}  |  gamma={gamma} Omega')
    print(f'{"="*60}')
    print(f'  Best fidelity : {np.max(fin_probs):.6f}')
    print(f'\n  Squeezing strengths (r_i raw | dB):')
    for i, (r, rdB) in enumerate(zip(r_params, r_dB)):
        print(f'    r_{i+1} = {r:8.4f}   ->   {rdB:6.2f} dB')
    print(f'\n  Time delays (Omega^-1 | units of 2pi/Omega):')
    for i, t in enumerate(t_params):
        print(f'    t_{i+1} = {t:8.4f} Omega^-1   ->   {t / (2*np.pi):.4f} x (2pi/Omega)')
    print(f'{"="*60}\n')


def save_optimal_params(best_params, loss_type, N, gamma):
    out_dir = os.path.join(_BASE, 'data', 'manuscript', 'optimal_parameters')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{loss_type}_decay_N={N}_gamma={gamma}.npy')
    np.save(out_path, best_params)
    print(f'  Optimal parameters saved -> {os.path.relpath(out_path, _BASE)}')


def plot_results(Probs, fin_probs, fin_params, stp_arr, tfit_arr, sorted_keys, loss_type, N, gamma):
    best_key     = sorted_keys[-1]
    best_history = Probs[best_key]
    best_params  = fin_params[-1]

    r_params   = np.array(best_params[::3],  dtype=float)
    phi_raw    = np.array(best_params[1::3], dtype=float)
    # absorb negative r sign into phase: r<0 is equivalent to r>0 with phi+pi
    phi_params = (phi_raw + np.where(r_params < 0, np.pi, 0.0)) % (2 * np.pi)
    t_params   = np.array(best_params[2::3], dtype=float)
    t_params_2pi = t_params / (2 * np.pi)
    r_dB_vals  = to_decibel_converter(np.abs(r_params))
    num_runs   = len(fin_probs)
    num_p      = len(r_params)
    num_t      = len(t_params)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.suptitle(
        f'{loss_type} decay  |  N={N}  |  $\\gamma={gamma}\\,\\Omega$  '
        f'|  best F = {np.max(fin_probs):.4f}',
        fontsize=12, y=1.02)

    # --- left panel: final fidelity of all runs ------------------------------
    ax = axes[0]
    x  = np.arange(1, num_runs + 1)
    dot_colors = ['tomato' if t else ('mediumseagreen' if s else 'steelblue')
                  for s, t in zip(stp_arr, tfit_arr)]
    ax.scatter(x, fin_probs, color=dot_colors, s=55, zorder=3,
               edgecolors='k', linewidths=0.3)
    ax.set_xlabel('GD run (sorted by fidelity)', fontsize=12)
    ax.set_ylabel(rf'final $F(N={N})$', fontsize=12)
    ax.set_title('all runs')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_xlim(0, num_runs + 1)
    ax.legend(handles=[
        mpatches.Patch(color='mediumseagreen', label='stopping cond. met'),
        mpatches.Patch(color='steelblue',      label='max iter reached'),
        mpatches.Patch(color='tomato',         label='time fitted to 0'),
    ], fontsize=11, loc='lower right')

    # --- middle panel: convergence curve of best run -------------------------
    ax = axes[1]
    iters = np.arange(1, len(best_history) + 1)
    ax.plot(iters, best_history, linewidth=1.5, color='steelblue')
    ax.set_xlabel('iteration', fontsize=12)
    ax.set_ylabel(rf'$F(N={N})$', fontsize=12)
    ax.set_title('convergence (best run)')
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- right panel: best-run parameters as bar plot ------------------------
    ax  = axes[2]
    ax2 = ax.twinx()
    bar_w = 0.35

    # r-params
    x_r = np.arange(num_p) - bar_w / 2
    ax.bar(x_r, r_dB_vals, width=bar_w, color='steelblue', alpha=0.8, zorder=3)

    # annotate each r-bar with its phase value
    for i, (rdb, phi) in enumerate(zip(r_dB_vals, phi_params)):
        lbl = r'$\Phi{=}\pi$' if np.isclose(phi, np.pi) else r'$\Phi{=}0$'
        pad = abs(rdb) * 0.02 + 0.1  # small gap between label and bar top
        # place label just above (positive bar) or below (negative bar) the bar edge
        ax.text(x_r[i], rdb + (pad if rdb >= 0 else -pad),
                lbl, ha='center', va='bottom' if rdb >= 0 else 'top',
                fontsize=9, color='black')

    # pad only the top so phi labels don't clip; bottom stays at 0
    y_hi = max(np.max(r_dB_vals), 1)
    ax.set_ylim(0, y_hi * 1.3)

    # t-params
    x_t = np.arange(num_t) + bar_w / 2
    ax2.bar(x_t, t_params_2pi, width=bar_w, color='darkorange', alpha=0.8, zorder=3)

    tick_labels = [rf'$r_{i+1}$/$t_{i+1}$' if i < num_t else rf'$r_{i+1}$'
                   for i in range(num_p)]
    ax.set_xticks(np.arange(num_p))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('pulse index', fontsize=12)
    ax.set_ylabel(r'squeezing $r_i$ [dB]', color='steelblue', fontsize=12)
    ax2.set_ylabel(r'time delay $t_i\;[2\pi/\Omega]$', color='darkorange', fontsize=12)
    ax.tick_params(axis='y', colors='steelblue', labelsize=11)
    ax2.tick_params(axis='y', colors='darkorange', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title('parameters (best run)', fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue',  alpha=0.8, label=r'$r_i$ [dB]'),
        mpatches.Patch(color='darkorange', alpha=0.8, label=r'$t_i\;[2\pi/\Omega]$'),
    ], fontsize=10, loc='upper left', bbox_to_anchor=(1.12, 1), borderaxespad=0)

    plt.tight_layout()

    fig_dir = os.path.join(_BASE, 'figures', 'auxiliary', 'optimization_results')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'{loss_type}_decay_N={N}_gamma={gamma}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Figure saved -> {os.path.relpath(fig_path, _BASE)}')

    plt.show()


if __name__ == '__main__':
    Probs, Params, stp_cnd, t_fit, settings = load_results(loss_type, N, gamma)
    fin_probs, fin_params, stp_arr, tfit_arr, sorted_keys = get_sorted_results(
        Probs, Params, stp_cnd, t_fit)
    best_params = fin_params[-1]

    print_params(best_params, fin_probs, loss_type, N, gamma)
    save_optimal_params(best_params, loss_type, N, gamma)
    plot_results(Probs, fin_probs, fin_params, stp_arr, tfit_arr, sorted_keys, loss_type, N, gamma)
