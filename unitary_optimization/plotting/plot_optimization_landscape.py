import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load data ---
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'optimization_landscape')
data = np.load(os.path.join(data_dir, 'optimization_landscape.npz'))

slices    = data['slices']    # shape (num_params, num_points)
ranges    = data['ranges']    # shape (num_params, num_points)
params    = data['params']    # optimal params in [r1,phi1,t1,...] format
pulse_num = (len(params) + 1) // 3
target_N  = int(data['target_N'])

# --- Parameter index helpers ---
r_indices   = [0 + 3 * i for i in range(pulse_num)]
phi_indices = [1 + 3 * i for i in range(pulse_num)]
t_indices   = [2 + 3 * i for i in range(pulse_num - 1)]

# --- Plot ---
fontsize = 20
colors     = plt.get_cmap('tab10').colors
fig, ax    = plt.subplots(1, 3, figsize=(15, 4))

handles, leg_labels = [], []

def to_decibel_converter(r):
    return -10 * np.log10(np.exp(-2 * r))

# ---------- r slices ----------
for i in range(pulse_num):
    # --- Main curve ---
    line, = ax[0].plot(
        to_decibel_converter(ranges[r_indices[i]]),
        slices[r_indices[i]],
        color=colors[i % len(colors)],
        linewidth=2,
        label=rf'$j={i+1}$',
    )
    handles.append(line)
    leg_labels.append(line.get_label())
    # --- Vertical line at optimal parameter value ---
    ax[0].axvline(
        to_decibel_converter(params[::3][i]),
        linestyle='--',
        color=colors[i % len(colors)],
        linewidth=2,
        alpha=0.5,
    )

ax[0].set_xlabel(r'$r_j\ \mathrm{[dB]}$', fontsize=fontsize)
ax[0].set_ylabel(rf'$\langle {target_N} | \rho_s | {target_N} \rangle$', fontsize=fontsize)
ax[0].set_xlim(0, to_decibel_converter(ranges[r_indices[1]][-1]))
ax[0].set_ylim(0, 1.0)
ax[0].tick_params(axis='both', labelsize=14, length=6, width=1.5)

# ---------- phi slices ----------
# counter1/counter2: ensure only the first axvline at a shared x-position is solid ('-'),
# subsequent ones at the same position are dashed ('--') to remain distinguishable.
counter1 = counter2 = False
for i in range(pulse_num):
    # --- Main curve ---
    if params[1::3][i] == 0:
        # if phi = 0, roll the curve so that we can plot it in the range [0, 2pi] instead of [-pi, pi]
        shift = -int((len(ranges[phi_indices[i]]) - 1) / 2) #
        ax[1].plot(
            np.linspace(0, 2 * np.pi, len(slices[phi_indices[i]])),
            np.roll(slices[phi_indices[i]], shift),
            color=colors[i % len(colors)],
            linewidth=2,
            label=rf'$j={i+1}$',
        )
        # --- Vertical line at optimal parameter value ---
        linestyle = '--' if counter2 else '-'
        ax[1].axvline(
            0,
            linestyle=linestyle,
            color=colors[i % len(colors)],
            linewidth=4,
            alpha=0.5,
        )
        if not counter2:
            counter2 = True
    else:
        ax[1].plot(
            ranges[phi_indices[i]],
            slices[phi_indices[i]],
            color=colors[i % len(colors)],
            linewidth=2,
            label=rf'$j={i+1}$',
        )
        # --- Vertical line at optimal parameter value ---
        linestyle = '--' if counter1 else '-'
        ax[1].axvline(
            params[1::3][i],
            linestyle=linestyle,
            linewidth=2,
            color=colors[i % len(colors)],
            alpha=0.5,
        )
        if not counter1:
            counter1 = True

ax[1].set_xlabel(rf'$\phi_j$', fontsize=fontsize)
ax[1].set_ylabel(rf'$\langle {target_N} | \rho_s | {target_N} \rangle$', fontsize=fontsize)
ax[1].set_xticks([0, np.pi, 2 * np.pi])
ax[1].set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
ax[1].set_xlim(ranges[phi_indices[0]][0], ranges[phi_indices[0]][-1])
ax[1].set_ylim(0, 1.0)
ax[1].tick_params(axis='both', labelsize=14, length=6, width=1.5)

# ---------- t slices ----------
for i in range(pulse_num - 1):
    # --- Main curve ---
    ax[2].plot(
        ranges[t_indices[i]] / 2 / np.pi,
        slices[t_indices[i]],
        color=colors[i % len(colors)],
        linewidth=2,
        label=rf'$j={i+1}$',
    )
    # --- Vertical line at optimal parameter value ---
    ax[2].axvline(
        params[2::3][i] / 2 / np.pi,
        linestyle='--',
        color=colors[i % len(colors)],
        linewidth=2,
        alpha=0.5,
    )

ax[2].set_xlabel(r'$t_j \; \left[\frac{2\pi}{\Omega}\right]$', fontsize=fontsize)
ax[2].set_ylabel(rf'$\langle {target_N} | \rho_s | {target_N} \rangle$', fontsize=fontsize)
ax[2].set_xlim(0, ranges[t_indices[1]][-1] / 2 / np.pi)
ax[2].set_ylim(0, 1.0)
ax[2].tick_params(axis='both', labelsize=14, length=6, width=1.5)

# ---------- subfigure labels ----------
for ax_, label in zip(ax.flat, ['(a)', '(b)', '(c)']):
    ax_.text(-0.225, 1.02, label,
             transform=ax_.transAxes,
             fontsize=17, fontweight='bold', va='top')

fig.legend(handles, leg_labels, fontsize=14, bbox_to_anchor=(0.323, 0.922), ncol=1)
plt.tight_layout(pad=2)

# --- Save figure ---
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'optimization_landscape.png'), dpi=300, bbox_inches='tight')
print(f"Figure saved to {out_dir}")
plt.show()
