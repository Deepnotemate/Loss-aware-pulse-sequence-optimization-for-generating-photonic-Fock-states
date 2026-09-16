import os
import sys
import numpy as np
import pylab as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import Loss_Aware_Optimization


# --- Configuration ---
loss_type = 'atom'  # 'atom' or 'cavity'
target_N         = 1
p = 4
create_labels = Loss_Aware_Optimization.create_labels_atom if loss_type == 'atom' else Loss_Aware_Optimization.create_labels_cavity

# --- paths ---
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'composite_state_dynamics', f'{loss_type}_decay')
fig_dir  = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')
os.makedirs(fig_dir, exist_ok=True)

# --- load data ---
diags_LME    = np.load(os.path.join(data_dir, 'diags_LME.npy'))
timelist_LME = np.load(os.path.join(data_dir, 'timelist_LME.npy'))
diags_U      = np.load(os.path.join(data_dir, 'diags_U.npy'))
timelist_U   = np.load(os.path.join(data_dir, 'timelist_U.npy'))
fidelity_LME = np.load(os.path.join(data_dir, 'Fidelity_LME.npy'))
fidelity_U   = np.load(os.path.join(data_dir, 'Fidelity_U.npy'))
print(f'Fidelity after last pulse for LME: {fidelity_LME:.6f}')
print(f'Fidelity after last pulse for U:   {fidelity_U:.6f}')

# --- basis ---
dim    = 60
cutoff = 3
label_dics, label_list = create_labels(dim, cutoff)

# --- pulse parameters (for pulse-position markers) ---
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
N1p4_U_optimal[1::3] = _phases           # phi_i
N1p4_U_optimal[2::3] = _u_params[1::2]   # t_i

t_params_LME = np.cumsum(np.append(0, N1p4_LME[2::3]))
t_params_U   = np.cumsum(np.append(0, N1p4_U_optimal[2::3]))

# --- plotting colors ---
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# --- loss-type-specific settings ---
if loss_type == 'atom':
    def is_featured_state(d):
        # highlight states with the target signal photon number N, as well as the ground state |0,0,g>
        return (d['signal'] == target_N) or (d['signal'] == 0 and d['idler'] == 0 and d['atom'] == 'g')
    counter_color_start = 3    # first 3 color slots are reserved for hardcoded states; extras start here
    other_threshold     = 400  # draw the "other states" legend entry at the 400th non-featured state
    legend_anchor       = (1.32, 1.6)
elif loss_type == 'cavity':
    def is_featured_state(d):
        ground = d['signal'] == 0 and d['idler'] == 0 and d['atom'] == 'g'
        # highlight states with target signal photon number N and up to 2 idler photons,
        # but exclude the state |2,1,e> since its contribution is negligible and it would clutter the legend
        signal = (d['signal'] == target_N and d['idler'] in [0, 1, 2]
                  and not (d['idler'] == 2 and d['signal'] == 1 and d['atom'] == 'e'))
        return ground or signal
    counter_color_start = 4    # one more hardcoded color than in the atom case; extras start at index 4
    other_threshold     = 100  # fewer non-featured states expected, so label appears earlier
    legend_anchor       = (1.32, 1.79)

# --- figure ---
fig, ax = plt.subplots(2, 1, figsize=(8, 7))

# plot pulses
for axis, t_params in zip(ax, [t_params_LME, t_params_U]):
    pulse_plots = axis.plot(
        np.array([t_params[i] * np.ones(10) for i in range(len(t_params))]).T / 2 / np.pi,
        np.array([np.linspace(0, 1, 10)      for i in range(len(t_params))]).T,
        c='grey', linestyle='--', linewidth=2)
    pulse_plots[0].set_label("pulses")
    [line.set_label("_nolegend_") for line in pulse_plots[1:]]
    pulse_plots[-1].set_linewidth(3.5)

# plot state populations
# pass 1: all states as grey background
ax[0].plot(timelist_LME / 2 / np.pi, diags_LME,
            linestyle=':', linewidth=2, color='grey', alpha=0.5)
ax[1].plot(timelist_U   / 2 / np.pi, diags_U,
            linestyle=':', linewidth=2, color='grey', alpha=0.5)

# pass 2: overplot featured states with colors
counter_color = counter_color_start
counter_other = 0

for i in range(diags_LME.shape[1]):
    d = label_dics[i]
    if is_featured_state(d):
        lbl = rf'$|{d["idler"]}, {d["signal"]}, {d["atom"]}\rangle$'
        # hardcoded colors for the states 00g, 11g, 10e, then cycle through the rest
        if d['signal'] == 0 and d['idler'] == 0 and d['atom'] == 'g':
            color = colors[0]
        elif d['signal'] == 1 and d['idler'] == 1 and d['atom'] == 'g':
            color = colors[1]
        elif d['signal'] == 1 and d['idler'] == 0 and d['atom'] == 'e':
            color = colors[2]
        else:
            color = colors[counter_color]
            counter_color += 1

        ax[0].plot(timelist_LME / 2 / np.pi, diags_LME[:, i], color=color, label=lbl)
        ax[1].plot(timelist_U   / 2 / np.pi, diags_U[:, i],   color=color, label=lbl)
    else:
        counter_other += 1
        if counter_other == other_threshold:
            # dummy entry to place "other states" at the correct position in the legend
            ax[1].plot([], [], linestyle=':', linewidth=2, color='grey', alpha=0.5,
                       label="other\nstates")

# --- axis labels & formatting ---
ax[0].set_ylabel(r'Probability', fontsize=20)
ax[0].tick_params(axis='both', labelsize=19)

ax[1].set_xlabel(r'time [$\frac{2\pi}{\Omega}]$', fontsize=20)
ax[1].set_ylabel(r'Probability', fontsize=20)
ax[1].legend(bbox_to_anchor=legend_anchor, fontsize=17)
ax[1].tick_params(axis='both', labelsize=19)

for ax_, label in zip(ax.flat, ['(a)', '(b)']):
    ax_.text(-0.12, 1.15, label, transform=ax_.transAxes,
             fontsize=19, fontweight='bold', va='top')

plt.subplots_adjust(hspace=0.3)

plt.savefig(os.path.join(fig_dir, f'{loss_type}_decay_dynamics.png'), bbox_inches='tight', dpi=300)
plt.show()

