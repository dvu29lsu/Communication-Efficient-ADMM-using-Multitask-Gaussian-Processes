# MTGP-ADMM: communication-efficient sharing ADMM with GP gradient surrogates

Code for the ICC 2027 experiments. `N` agents jointly solve the sharing problem

```text
minimize   sum_i f_i(x_i)  +  zeta * || sum_i x_i ||_1
```

with sharing ADMM (Boyd et al.). In each iteration, agent `i` must report the gradient of the Moreau envelope of `f_i` at its local point `z_i`, which costs one exact local solve plus one message to the coordinator. This repository replaces most of those exact queries with a Gaussian-process surrogate. The agent queries for real only when the surrogate's predictive variance exceeds an adaptive threshold.

Three methods are compared:

| Method | Gradient source | Provider |
|---|---|---|
| **Vanilla ADMM** | Exact Moreau gradient every iteration (every agent communicates every round) | `admm.moreau_oracle.MoreauOracle` |
| **SOGP-ADMM** | One independent derivative GP per agent | `query_method/sogp/sogp_provider.py` |
| **MTGP-ADMM** | One multi-task derivative GP (LMC kernel) shared by all agents, so agents borrow strength from each other | `query_method/mogp/mtgp_provider.py` |
| **Cluster MTGP-ADMM** | One MTGP per agent cluster | `query_method/mogp/cluster_mtgp_provider.py` |

## How the GP query works?

For each agent, the GP providers (`MTGPMoreauGradProvider`, `SOGPMoreauProvider`) do the following:

1. **Burn-in** (`k < burn_in`): query exactly and store `(z_i, [f_gamma(z_i), grad f_gamma(z_i)])`.
2. **Activate**: once every agent has at least `min_points` samples, fit the GP on values and gradients jointly.
3. **Predict**: at each later iteration, predict the gradient mean and variance at `z_i`. Project the mean onto an intersection of ellipsoids built from past samples (`utils/compute_ellipsoids.py`, `utils/projection_to_ell.py`).
4. **Gate**: compare `max(var)` with a geometrically decaying threshold
   `psi_i^k = iota * V_i^{k0} * alpha^(k - k0)` (`mtgp_threshold.py` / `sogp/threshold.py`).
   - If the variance is above the threshold, query exactly, add the point to the buffer, and refit the GP.
   - Otherwise, use the GP prediction and send nothing.

A provider is a callable `provider(i, z_i, query_flag, k) -> (grad, did_query, stats[, grad_err])`. `SharingADMM` uses the `did_query` flag to count communication.

## Local objective families

Each agent's `f_i` is built from `(theta_i, Gamma_i)` in `admm/local_models.py`. Select the family with `local_model=`:

| `local_model` | `f_i(x)` with `u = x - theta_i` | Prox solver |
|---|---|---|
| `"quadratic"` | `u' Gamma_i u` | Closed form (`admm/exact_quadsolver.py`) |
| `"poly4"` | `u' Gamma_i u + lambda4 * sum(u^4)` | `scipy.optimize.minimize` |
| `"dir_poly4"` | `u' Gamma_i u + lambda4 * sum((A u)^4)`, where `A` is an `r x p` direction matrix | `scipy.optimize.minimize` |

## Repository layout

```text
admm/
  sharing_admm.py       SharingADMM: x/z/u updates, residuals, stopping rule, per-iteration logs
  moreau_oracle.py      MoreauOracle (prox + Moreau value/grad); build_local_models / build_moreau_oracles
  local_models.py       Quadratic, Poly4, DirectionalPoly4 objectives (value, grad, hess)
  exact_quadsolver.py   closed-form Moreau gradient for quadratics
  proximal_ops.py       soft_threshold
utils/
  config.py             ADMMConfig (zeta, rho, eps_abs, eps_rel, max_iters, burn_in)
  coordinator.py        Coordinator: wraps SharingADMM and aggregates communication statistics
  json_processing.py    process_output_to_jsonl: per-iteration log writer
  csv_writer.py         summary rows for sweeps / repeated runs
  compute_ellipsoids.py, projection_to_ell.py   gradient-consistency projection
agents_loader/
  agent_load.py         load_data_from_mat (Set*.mat) and load_data_from_jsonl (clustered data)
gp/                     single-output derivative GP (SOGP) model and trainer
mogp/                   multi-task derivative GP: LMC gradient kernel, ExactLCMGrad model, trainer
query_method/
  sogp/                 SOGP provider, threshold, decision rule
  mogp/                 MTGP and cluster-MTGP providers, threshold, task-specific noise likelihoods
data/10 Agents/{p5,p10}/Set1..Set10.mat    10-agent problems in dimension p = 5 or 10
sand_box/               clustered JSONL datasets (set1_{one,two,three}_cluster.jsonl) and analysis notebooks
results/                experiment outputs (JSONL logs, sweep summaries)
run_exp/plot_results.py plotting helper
run_*.py                experiment entry points (see below)
```

## Setup

Python 3.10+ is recommended. Install the dependencies:

```bash
pip install numpy scipy pandas matplotlib torch gpytorch linear_operator
pip install cvxpy
```

## Running experiments

All scripts use relative paths (`data/...`, `results/...`, `sand_box/...`), so **run them from this directory**:

```bash
cd ADMM_MTGP
python run_mtgp_admm.py
```

The scripts take no command-line arguments. Set datasets, hyperparameters, and output paths by editing the constants at the top of each script and inside its `main()`.

### Unclustered `.mat` datasets (`data/10 Agents/p5`)

| Script | What it does |
|---|---|
| `run_vanilla_admm_store_infor.py` | Vanilla ADMM on a single set |
| `run_sogp_admm.py` | SOGP-ADMM on a single set |
| `run_mtgp_admm.py` | MTGP-ADMM on a single set |
| `run_fullset_{vanilla,sogp,mtgp}.py` | Run one method on every `Set*.mat` in the data folder |
| `run_n_times.py` | Repeat SOGP/MTGP runs over several sets, writing CSV summaries |
| `run_sweep_{sogp,mtgp}_admm.py` | Sweep the threshold parameters `iota` and `alpha`, writing to `results/*sweep_*` |

### Clustered JSONL datasets (`sand_box/set1_*_cluster.jsonl`)

| Script | What it does |
|---|---|
| `run_vanilla_admm_cluster.py` | Vanilla ADMM with a selectable `local_model` |
| `run_sogp_admm_cluster.py` | SOGP-ADMM with a selectable `local_model` |
| `run_mtgp_admm_cluster.py` | Global MTGP-ADMM on clustered data |
| `run_multi_cluster_mtgp.py` | Cluster-specific MTGP-ADMM, plus per-cluster communication counts |
| `run_sweep_{sogp,mtgp}_admm_cluster.py` | `iota` / `alpha` sweeps on clustered data |

Each clustered JSONL file holds one record with `gamma_list` (`N x p x p`), `theta_list` (`N x p`) and `labels` (`N`). These files come from the generator in `../cluster_specific_functions_generating_updated/cluster_specific_functions_generating/`, which has its own README.

### Key hyperparameters

| Name | Where | Meaning | Typical values |
|---|---|---|---|
| `rho` | `ADMMConfig` | ADMM penalty; the Moreau smoothing parameter is `1/rho` | 10 |
| `zeta` | `ADMMConfig` | Weight of the l1 sharing term | 1.0 |
| `eps_abs`, `eps_rel` | `ADMMConfig` | Stopping tolerances for the primal/dual residuals | 1e-4 / 1e-3 |
| `burn_in` | script | Number of exact queries per agent before the GP activates | 5 |
| `iota` | provider | Initial threshold as a multiple of the first predicted max variance | MTGP 1.0–1.3, SOGP ~0.6 |
| `alpha` | provider | Per-iteration decay of the threshold | 0.96 (MTGP), 0.8 (SOGP) |
| `num_latents` (`Q`), `rank` | MTGP | Number of LMC latent processes and rank of the task mixing matrix | 3–5, 1 |
| `train_iter`, `lr` | GP | Adam iterations and learning rate for hyperparameter fitting | 300, 0.03 |
| `lambda4`, `A` | local model | Weight and directions of the quartic term | 0.5 |

## Output format

`process_output_to_jsonl` writes one JSON line per ADMM iteration:

```json
{"iteration": 0, "primal_residual": ..., "dual_residual": ..., "obj_val": ...,
 "comm_per_iter": 10, "grads_error_per_iter": [...], "other_stats": [...]}
```

- `comm_per_iter` is the number of agents that queried exactly in that iteration.
- `grads_error_per_iter` is `||g_true - g_GP||_2` for each agent.
  - The GP providers compute the exact gradient for this diagnostic even when they skip a query. Count communication from `comm_per_iter`, not from wall-clock time.
- `other_stats` holds per-agent `var_max`, `psi_k` and `do_query`.

The dict that `Coordinator.run` returns also contains `comm_total`, `comm_per_agent`, `comm_cum`, `converged` and `elapsed`.

## Notes

- Seeds are fixed through `utils/set_seed.py`.
- The default dtype is `torch.float32`, and computation runs on CPU.
- The scripts print a lot of per-agent, per-iteration output. This is intentional for debugging.
- `data_generation/` is empty. Use the sibling `cluster_specific_functions_generating_updated/` project to generate data.
