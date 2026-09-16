# Loss-aware pulse sequence optimization for generating photonic Fock states

This repository contains the code and data accompanying the manuscript:

> **Loss-aware pulse sequence optimization for generating photonic Fock states**  
> B. Stodd, P. Tiwari, R. Sondenheimer, S. Saravi, M. Gärttner  

---

## Overview

We optimize multi-pulse driving protocols for the preparation of photonic Fock states in a hybrid cavity system consisting of a nonlinear crystal and a two-level system (2LS). 

The repository is divided into two independent modules:

| Module | Description |
|--------|-------------|
| `unitary_optimization/` | Gradient-descent optimization assuming ideal (lossless) unitary dynamics |
| `loss_aware_optimization/` | Optimization incorporating dissipation via the Lindblad master equation (LME) |

---

## Repository structure

```
.
├── unitary_optimization/
│   ├── src/
│   │   ├── utils.py                  # Unitary_Optimization class (dynamics, gradient, optimizer)
│   │   ├── run_optimization.py       # Run gradient-descent optimization 
│   │   ├── convergence_testing.py    # Verify Fock-space truncation convergence
│   │   ├── optimization_landscape.py # Compute fidelity landscape over parameter slices
│   │   └── state_transformations.py  # Compute state evolution
│   ├── plotting/
│   │   ├── plot_highest_fidelities.py   # Fig. 2: optimal fidelities vs. pulse number
│   │   ├── plot_optimization_landscape.py # Fig. 4: optimization landscape
│   │   ├── plot_state_trafos.py         # Fig. 3: composite state populations
│   │   └── plot_convergence_curve.py    # Fig. 9: final fidelities and convergence of best GD run
│   ├── data/
│   │   ├── auxiliary/optimization_results/  # Raw optimization output (all GD runs)
│   │   └── manuscript/                      # Processed data used in figures
│   └── figures/
│
└── loss_aware_optimization/
    ├── src/
    │   ├── utils.py                  # Loss_Aware_Optimization class (LME dynamics, gradient, optimizer)
    │   ├── run_optimization.py       # Run loss-aware gradient-descent
    │   ├── evaluate_results.py       # Load results, print/save optimal parameters, auxiliary plots
    │   ├── compute_dynamics.py       # Compute dynamics for optimal parameters
    │   └── scan_dissipation_rates.py # Scan fidelity over a range of decay rates  
    ├── plotting/
    │   ├── plot_dynamics.py          # Figs. 6, 7: time evolution of composite state populations
    │   └── plot_dissipation_rate_scan.py # Fig. 5, 8: fidelity vs. dissipation rate
    ├── data/
    │   ├── auxiliary/                # Raw optimization output
    │   └── manuscript/               # Optimal parameters and processed data used in figures
    └── figures/
```

---

## Dependencies

Install all required packages via:

```bash
pip install -r requirements.txt
```

---

## Execution order

Scripts marked **(cluster)** were run on an HPC cluster to parallelize many independent optimization runs; all others run locally. Run each script from its containing folder with `python script.py`; rows marked "edit config" require changing a few variables near the top of the file first.
> **Note:** All plotting scripts read pre-computed data already included in `data/`, so the optimization jobs do not need to be rerun to reproduce the manuscript figures.


### Unitary optimization

| # | Script | What it does | Run |
|---|--------|---------------|-----|
| 1 | `src/run_optimization.py` (cluster) | GD optimization for one `(p, N)` combination; saves fidelities/parameters for each run to `data/auxiliary/optimization_results/`. | `python run_optimization.py p=4 N=2` |
| 2 | `src/convergence_testing.py` (edit config) | Checks Fock-space truncation convergence for the best runs; saves the optimal parameter set to `data/manuscript/optimal_configurations/`. | set `p`, `target_N`, `num_tests`, then `python convergence_testing.py` |
| 3 | `src/optimization_landscape.py` (edit config) | Computes fidelity-landscape slices around the optimum; saves to `data/manuscript/optimization_landscape/`. | set `p_in`, `N_in`, then `python optimization_landscape.py` |
| 4 | `src/state_transformations.py` (edit config) | Computes the state evolution through the pulse sequence; saves to `data/manuscript/state_transformations/`. | set `p_in`, `N_in`, then `python state_transformations.py` |
| 5 | `plotting/plot_highest_fidelities.py` | Fig. 2 — optimal fidelities versus pulse number; interactively displays the fidelity and pulse parameters for a selected `(p, N)` configuration. | `python plot_highest_fidelities.py` |
| 6 | `plotting/plot_optimization_landscape.py` | Fig. 4 — optimization landscape. | `python plot_optimization_landscape.py` |
| 7 | `plotting/plot_state_trafos.py` | Fig. 3 — composite state populations. | `python plot_state_trafos.py` |
| 8 | `plotting/plot_convergence_curve.py` | Fig. 9 — sorted final fidelities across all initializations and the infidelity convergence of the best GD run. | `python plot_convergence_curve.py` |

### Loss-aware optimization

| # | Script | What it does | Run |
|---|--------|---------------|-----|
| 1 | `src/run_optimization.py` (cluster, edit config) | Loss-aware GD for a given `loss_type`, `gamma`, `target_N`; writes results and the optimization settings (`settings.json`) to `data/auxiliary/optimization_results/{loss_type}_decay_N={N}_gamma={gamma}/`. | Configure the optimization hyperparameters in the script's `Configuration` section, then run `python run_optimization.py --init_idx 0` (use different `--init_idx` values for parallel runs). |
| 2 | `src/evaluate_results.py` (edit config) | Ranks runs by fidelity, prints the best parameters, and saves them to `data/manuscript/optimal_parameters/`; creates auxiliary evaluation plot.| set `loss_type`, `N`, `gamma`, then `python evaluate_results.py` |
| 3 | `src/compute_dynamics.py` (edit config) | Computes time evolution for the optimal loss-aware and optimal unitary parameters, respectively; saves to `data/manuscript/composite_state_dynamics/`. | Configure the simulation parameters in the script's `Configuration` section, then run `python compute_dynamics.py`. |
| 4 | `src/scan_dissipation_rates.py` (edit config) | Scans fidelity over a range of decay rates; saves to `data/manuscript/dissipation_rate_scans/`. | Configure the simulation parameters in the script's `Configuration` section, then run `python scan_dissipation_rates.py`. |
| 5 | `plotting/plot_dynamics.py` | Figs. 6, 7 — state-population dynamics; prints fidelity after last pulse. | Set `loss_type` in the script's `Configuration` section, then run `python plot_dynamics.py` |
| 6 | `plotting/plot_dissipation_rate_scan.py` | Figs. 5, 8 — fidelity vs. dissipation rate. | `python plot_dissipation_rate_scan.py` |


---

## Contact

Benjamin Stodd — benjamin.stodd@uni-jena.de

