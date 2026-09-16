import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'manuscript', 'dissipation_rate_scans')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'manuscript')

# --- Load fidelity data ---
# Each file has shape (num_configs, num_gamma_points):
#   index 0: 3-pulse unitary optimum
#   index 1: 4-pulse unitary optimum
#   index 2: 4-pulse loss-aware optimum
fids_atom_N1   = np.load(os.path.join(data_dir, 'fidelities_atom_decay_N=1.npy'))
fids_cavity_N1 = np.load(os.path.join(data_dir, 'fidelities_cavity_decay_N=1.npy'))
# index 0: 3-pulse, 1: 4-pulse unitary, 2: loss-aware gamma=0.01, 3: loss-aware gamma=0.03
fids_cavity_N2 = np.load(os.path.join(data_dir, 'fidelities_cavity_decay_N=2.npy'))

# --- Load decay rate arrays ---
gamma_atom_N1   = np.load(os.path.join(data_dir, 'decay_rates_atom_decay_N=1.npy'))
gamma_cavity_N1 = np.load(os.path.join(data_dir, 'decay_rates_cavity_decay_N=1.npy'))
gamma_cavity_N2 = np.load(os.path.join(data_dir, 'decay_rates_cavity_decay_N=2.npy'))

c = 'FireBrick'
fontsize, legendsize, ticksize = 16, 12, 12

# ── Figure 1: N=1, two-panel (atom decay top, cavity decay bottom) ──────────
N = 1
fig, ax = plt.subplots(2, figsize=(6, 7))

# --- top panel: atom decay ---
ax[0].plot(gamma_atom_N1, fids_atom_N1[0], linestyle='--', linewidth=2, label='3-pulse unitary optimium')
ax[0].plot(gamma_atom_N1, fids_atom_N1[1], linestyle='--', linewidth=2, label='4-pulse unitary optimum')
ax[0].plot(gamma_atom_N1, fids_atom_N1[2], linestyle='--', linewidth=2, label='4-pulse loss-aware optimum')

# vertical marker at the gamma_a value used during optimization
ax[0].plot(np.ones(10)*0.05, np.linspace(0.9, 1, 10), linestyle=':', color=c)

ax[0].set_xlabel(r'$\gamma_\mathrm{a}$ [$\Omega$]', fontsize=fontsize)
ax[0].set_ylabel('$F(N=1)$', fontsize=fontsize)
ax[0].tick_params(axis='both', which='major', labelsize=ticksize)
ax[0].grid(True, linestyle=':', alpha=0.7)
ax[0].legend(fontsize=legendsize, frameon=True)

# --- bottom panel: cavity decay ---
ax[1].plot(gamma_cavity_N1, fids_cavity_N1[0], linestyle='--', linewidth=2, label='3-pulse optimium')
ax[1].plot(gamma_cavity_N1, fids_cavity_N1[1], linestyle='--', linewidth=2, label='4-pulse unitary optimum')
ax[1].plot(gamma_cavity_N1, fids_cavity_N1[2], linestyle='--', linewidth=2, label='4-pulse loss-aware optimum')

# vertical marker at the gamma_s value used during optimization
ax[1].plot(np.ones(10)*0.01, np.linspace(0.5, 1, 10), linestyle=':', color=c)

ax[1].set_xlabel(r'$\gamma_\mathrm{s}$ [$\Omega$]', fontsize=fontsize)
ax[1].set_ylabel(f'$F(N={N})$', fontsize=fontsize)
ax[1].tick_params(axis='both', which='major', labelsize=ticksize)
ax[1].grid(True, linestyle=':', alpha=0.7)

# --- subfigure labels ---
for ax_, label in zip(ax.flat, ['(a)', '(b)']):
    ax_.text(-0.12, 1.15, label,
             transform=ax_.transAxes,
             fontsize=15,
             fontweight='bold',
             va='top')

plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(figs_dir, 'N=1_comparison_of_dissipation_rates.png'), dpi=300)
plt.show()

# ── Figure 2: N=2, single panel (cavity decay) ──────────────────────────────
N = 2
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(gamma_cavity_N2, fids_cavity_N2[0], linestyle='--', linewidth=2, label='3-pulse unitary optimium')
ax.plot(gamma_cavity_N2, fids_cavity_N2[1], linestyle='--', linewidth=2, label='4-pulse unitary optimum')
ax.plot(gamma_cavity_N2, fids_cavity_N2[3], linestyle='--', linewidth=2, label='4-pulse loss-aware \n' + r'optimum ($\gamma_\mathrm{s}=0.03\,\Omega$)')
ax.plot(gamma_cavity_N2, fids_cavity_N2[2], linestyle='--', linewidth=2, label='4-pulse loss-aware \n' + r'optimum ($\gamma_\mathrm{s}=0.01\,\Omega$)')

ax.set_xlabel(r'$\gamma_\mathrm{s}$ [$\Omega$]', fontsize=20)
ax.set_ylabel(f'$F(N={N})$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=17)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'N=2_comparison_of_dissipation_rates.png'), dpi=300)
plt.show()
