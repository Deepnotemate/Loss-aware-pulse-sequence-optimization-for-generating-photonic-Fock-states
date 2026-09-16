import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

# --- Load data ---
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'state_transformations')

# states_ge: shape (8, 2, 2*dim) — axis 0: state index, axis 1: 0=ground/1=excited
states_ge = np.load(os.path.join(data_dir, 'states_ge.npy'))
dim = states_ge.shape[2] // 2

# --- Layout parameters ---
# subplot_order[i] = (row, col) for state index i
subplot_order = [(0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0)]
xlimits       = [10,   45,   45,   45,   45,   150,  150,  15]
# xticks per state index (None = auto)
xtick_configs = [
    [0, 5, 10, 15, 20],
    np.arange(0, 50, 10),
    np.arange(0, 50, 10),
    np.arange(0, 50, 10),
    np.arange(0, 50, 10),
    None,
    None,
    np.arange(0, 16, 5),
]

tol_colors = [
    "#264653",  # deep teal
    "#F4A261",  # orange
]
fontsize = 20
fsize = 15

# --- Figure ---
fig, ax = plt.subplots(2, 4, figsize=(18, 7))

for i, (row, col) in enumerate(subplot_order):
    n = np.arange(states_ge.shape[2])
    label_g = r'$P_g(n)$' if i == 3 else None
    label_e = r'$P_e(n)$' if i == 3 else None
    ax[row, col].bar(n, states_ge[i, 0], color=tol_colors[1], label=label_g)
    ax[row, col].bar(n, states_ge[i, 1], color=tol_colors[0], label=label_e)
    if i == 3:
        ax[row, col].legend(loc='upper right', fontsize=17)

# --- Axes labels & ticks ---
for i in range(2):
    for j in range(4):
        ax[i, j].set_xlabel(r'$N_\mathrm{tot}$', fontsize=fontsize)
        ax[i, j].tick_params(axis='x', labelsize=18)
        ax[i, j].tick_params(axis='y', labelsize=18)

for i, (row, col) in enumerate(subplot_order):
    ax[row, col].set_xlim(-0.5, xlimits[i])
    if xtick_configs[i] is not None:
        ax[row, col].set_xticks(xtick_configs[i])
    ax[row, col].tick_params(axis='both', labelsize=18, length=6, width=1.5)

ax[0, 0].set_ylabel('Probability', fontsize=fontsize)
ax[1, 0].set_ylabel('Probability', fontsize=fontsize)

# --- Subfigure labels ---
labels = ['(a)', '(b)', '(c)', '(d)', '(h)', '(g)', '(f)', '(e)']
for ax_, label in zip(ax.flat, labels):
    ax_.text(-0.15, 1.15, label,
             transform=ax_.transAxes,
             fontsize=17, fontweight='bold', va='top')

plt.tight_layout()

# --- Load display parameters ---
r_dB   = np.load(os.path.join(data_dir, 'r_dB.npy'))    # [r1,r2,r3,r4] in dB
t_vals = np.load(os.path.join(data_dir, 't_vals.npy'))  # [t1,t2,t3]
phases = np.load(os.path.join(data_dir, 'phases.npy'))  # [0|1, ...], 1 means pi

def nl_label(i):  # i is 1-based
    phi_str = r'\pi' if phases[i - 1] == 1 else '0'
    return (rf'$r_{i} = {r_dB[i-1]:.2f}\,\mathrm{{dB}}$'
            + '\n' + rf'$\phi_{i} = {phi_str}$')

def ls_label(i):  # i is 1-based
    return rf'$t_{i} = {t_vals[i-1]:.2f}\,\Omega^{{-1}}$'


# --- Arrow connector ---
def draw_arrow(kind, r1, c1, r2, c2, text_above=None, text_below=None, color='black'):
    """kind: 'h' horizontal forward, 'hl' horizontal backward, 'v' vertical."""
    pos1 = ax[r1, c1].get_position()
    pos2 = ax[r2, c2].get_position()
    if kind == 'h':
        x1, y1 = pos1.x1 - 0.05, (pos1.y0 + pos1.y1) / 2 + 0.03
        x2, y2 = pos2.x0 - 0.01, (pos2.y0 + pos2.y1) / 2 + 0.03
        style = '-|>'
        ta_kw = dict(ha='left',   va='bottom', x=(x1+x2)/2 - 0.060, y=(y1+y2)/2 + 0.01)
        tb_kw = dict(ha='left',   va='top',    x=(x1+x2)/2 - 0.075, y=(y1+y2)/2 - 0.01)
    elif kind == 'hl':
        x1, y1 = pos1.x1 - 0.07, (pos1.y0 + pos1.y1) / 2 + 0.05
        x2, y2 = pos2.x0 - 0.04, (pos2.y0 + pos2.y1) / 2 + 0.05
        style = '<|-'
        ta_kw = dict(ha='right',  va='bottom', x=(x1+x2)/2 + 0.010, y=(y1+y2)/2 + 0.01)
        tb_kw = dict(ha='left',   va='top',    x=(x1+x2)/2 - 0.055, y=(y1+y2)/2 - 0.01)
    elif kind == 'v':
        x1 = x2 = (pos1.x0 + pos1.x1) / 2
        y1, y2  = pos1.y0 - 0.13, pos2.y1 - 0.142
        style = '-|>'
        ta_kw = dict(ha='center', va='top', x=(x1+x2)/2 - 0.015, y=(y1+y2)/2 - 0.005)
        tb_kw = dict(ha='center', va='top', x=(x1+x2)/2 + 0.040, y=(y1+y2)/2 - 0.011)
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), transform=fig.transFigure,
                                   arrowstyle=style, mutation_scale=20,
                                   linewidth=3, color=color, alpha=0.5))
    if text_above:
        fig.text(ta_kw.pop('x'), ta_kw.pop('y'), text_above, fontsize=fsize, color='black', **ta_kw)
    if text_below:
        fig.text(tb_kw.pop('x'), tb_kw.pop('y'), text_below, fontsize=fsize, color='black', **tb_kw)


# --- Draw all connectors ---
# (kind, r1, c1, r2, c2, label_above, label_below, color)
connectors = [
    ('h',  0, 0, 0, 1, r'$\hat{U}_\mathrm{NL}$',  nl_label(1), 'grey'),
    ('h',  0, 1, 0, 2, r'$\hat{U}_\mathrm{2LS}$', ls_label(1), 'black'),
    ('h',  0, 2, 0, 3, r'$\hat{U}_\mathrm{NL}$',  nl_label(2), 'grey'),
    ('v',  0, 3, 1, 3, r'$\hat{U}_\mathrm{2LS}$', ls_label(2), 'black'),
    ('hl', 1, 2, 1, 3, r'$\hat{U}_\mathrm{NL}$',  nl_label(3), 'grey'),
    ('hl', 1, 1, 1, 2, r'$\hat{U}_\mathrm{2LS}$', ls_label(3), 'black'),
    ('hl', 1, 0, 1, 1, r'$\hat{U}_\mathrm{NL}$',  nl_label(4), 'grey'),
]

for kind, r1, c1, r2, c2, ta, tb, color in connectors:
    draw_arrow(kind, r1, c1, r2, c2, text_above=ta, text_below=tb, color=color)

# --- Save figure ---
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'state_transformations.png'), bbox_inches='tight', dpi=300)
print(f"Figure saved to {out_dir}")
plt.show()
