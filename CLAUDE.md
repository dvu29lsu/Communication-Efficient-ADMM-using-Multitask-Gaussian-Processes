# CLAUDE.md

## How we work

- **Role:** act as an experienced researcher with strong programming and machine-learning skills. You are the user's co-worker, pair programmer and fellow researcher, not just a code generator. Raise research concerns (correctness, fairness of comparisons, leakage) as well as code issues.
- **Plan first:** before implementing anything, present a clear plan and wait for the user's approval.
- **One step at a time:** implement one function or one file per step. Then stop, show the change, and wait for review before moving on. Don't edit several files in one go.
- **Code quality:**
  - Write clear, clean, well-commented code; comments explain *why*, not just *what*.
  - Each function does one thing.
  - Don't over-engineer: no extra abstraction layers, config systems or compatibility shims unless asked.
  - Follow the existing style of this repo: plain functions and classes, numpy/torch idioms, and the current naming conventions.
- **Who runs what:** the user runs the ADMM algorithm scripts (vanilla, SOGP, MTGP runners and sweeps). Don't run them unless asked. Cheap checks are fine: imports, constructing objects, loading data, single oracle calls.

## What this repository is

Communication-efficient sharing ADMM with Gaussian-process surrogates (ICC 2027 work). `N` agents solve

```text
min_{x_1..x_N}  sum_i f_i(x_i) + zeta * || sum_i x_i ||_1
```

Each ADMM iteration, agent `i` must supply the gradient of the Moreau envelope of `f_i` at its query point. An exact answer costs one local prox solve plus one message to the coordinator. A GP predicts that gradient instead, and the agent queries exactly only when the predictive variance exceeds a decaying threshold. The goal is fewer total exact queries at the same convergence and accuracy.

Methods (use these names consistently): **exact/vanilla ADMM**, **SOGP-ADMM** (one derivative GP per agent), **global MTGP-ADMM** (one LMC multitask derivative GP over all agents), **cluster-wise MTGP-ADMM** (one MTGP per known cluster).

### Relation to `../ADMM_MTGP`

- `../ADMM_MTGP` is the user's earlier sandbox repo. It produced the ICC paper results, but it is over-engineered: subprocess sweep orchestration, multiple profiles and source registries, archived compatibility layers.
- **This repo is the clean rebuild.** It is where runs are rederived and results analyzed.
- Read `../ADMM_MTGP` for ideas, algorithms, conventions and paper settings. Its `PROJECT_KNOWLEDGE.md` has the research decision log. Don't copy its structure wholesale.
- Prefer small, readable modules and explicit configuration here.
- When porting behavior from ADMM_MTGP, say what changes numerically.
- Don't modify `../ADMM_MTGP` from this repo's tasks.

## Running

- Run every script **from this repo root** in the `experiment_env` conda env (the base `python` has no torch), e.g. `conda run -n experiment_env python run_vanilla_admm_cluster.py`. Imports are root-relative (`from admm...`, `from utils...`), and data and output paths are relative (`data_generation/...`, `test/...`).
- The scripts have no CLI. Edit the constants and `main()` in each script to change the dataset, hyperparameters, `local_model`, and output path.
- Active entry points and their example datasets:
  - `run_vanilla_admm_cluster.py`: `homogeneous.json`
  - `run_sogp_admm_cluster.py`: `homogeneous.json`
  - `run_mtgp_admm_cluster.py` (global MTGP): `homogeneous.json`
  - `run_multi_cluster_mtgp.py` (cluster-wise MTGP, plus per-cluster query counts): `separated_2_clusters.json`, with `adapt_latents_to_cluster=True`. Its active setting is `dir_poly4`; the other three runners default to `quadratic`.
- Shared runner settings: `ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-6, eps_rel=1e-3)`, `set_seed(123)`, `burn_in=5`, float32, CPU.
- Test output goes to `test/test_<method>_<dataset>.jsonl`. The folder is created in `main()`, so importing a runner creates no folders. Never overwrite existing files in `results/` or `miscellaneous/old_results/`.
- Dependencies: numpy, scipy, pandas, torch, gpytorch, linear_operator, **cvxpy**. cvxpy is required by every GP run because `utils/projection_to_ell.py` imports it. There is no requirements file.
- There is no test suite. Output is very verbose by design (per-agent, per-iteration prints).

## Data

- `sand_box/set1_{one,two,three}_cluster.jsonl` are single-record JSON Lines files with keys `gamma_list` (N×p×p), `theta_list` (N×p) and `labels` (N). They are read by `agents_loader.load_data_from_jsonl`, which returns `labels` as a pandas Series. Callers use `labels[0]`.
- `data/10 Agents/p{5,10}/Set*.mat` (keys `Phi`, `theta`) are read by `load_data_from_mat`. These are the unclustered legacy problems.
- `data_generation/data_generation.ipynb` is the new, simplified generator. It is a rewrite of ADMM_MTGP's `cluster_specific_functions_generating_updated/` package. It starts from `Set1.mat` and uses the normalized distance `w_theta·‖Δθ‖/s_theta + w_gamma·AIRM(Γ_i,Γ_j)/s_gamma`, medoid/prototype selection, radius-rejection sampling and contracted-prototype overlap.
  - It writes six regimes (10 agents, p = 5): `homogeneous`, `heterogeneous`, `separated_{2,3}_clusters` and `overlapping_{2,3}_clusters` (`.json`).
  - These are the **datasets the runners use**. They are `indent=2` JSON objects, read with `agents_loader.load_data_from_json`, which returns numpy arrays `gamma_list` (N,p,p), `theta_list` (N,p) and `labels` (N,) int.
- `data_generation/generate_direction_matrix.py` writes `data_generation/directional_quartic_dim5.json`. This is the fixed orthogonal 5×5 `A` for `dir_poly4`.
  - It is made by QR of a standard-normal matrix, seed 20260813, with signs set so `diag(R) > 0`. This is the same recipe and seed as the paper, and the result matches ADMM_MTGP's matrix to 2e-16.
  - Load it with `load_direction_matrix`, which checks that it is square and orthogonal. One `A` is shared by every agent, dataset and method.
- In the paper, ADMM_MTGP used six accepted datasets in `../ADMM_MTGP/function_parameters/K0{1,2,3}/.../params.jsonl`. Those files are not the same samples as the notebook outputs.

## Algorithm and code map

Data flow: loader → `admm.moreau_oracle.build_local_models` → `utils.coordinator.Coordinator(gamma, theta, ADMMConfig, burn_in, local_models)` → `admm.sharing_admm.SharingADMM.run_moreau_admm(provider, burn_in)`.

- **ADMM update, x-first (x → y → u)**, in `SharingADMM.update_step`:
  `q_i = x_i + ȳ − x̄ − ū` (query point from the current state);
  `x_i ← q_i − β_i/ρ`;
  `ȳ ← soft_threshold(x̄ + ū, ζ/ρ)`;
  `y_i ← x_i − x̄ + ȳ`;
  `ū ← ū + x̄ − ȳ`.
  - `y_*` are the auxiliary variables. Commented-out legacy code calls them `z`.
  - Stopping requires both `r ≤ eps_pri` and `s ≤ eps_dual` (Boyd-style tolerances with `sqrt(N·p)·eps_abs`).
  - This update is mathematically identical to ADMM_MTGP's `admm/sharing_admm_xyu.py`. That version also records state histories (`x_hist`, `y_blocks_hist`, `u_bar_hist`, `z_list_hist`, `moreau_grad_hist`) and a primal objective `Σf_i(x_i) + ζ‖Σx_i‖₁`.
- **Provider contract:** `provider(i, z_i, query_flag, k) -> (grad, did_query, stats[, grad_err])`.
  - `query_flag = k < burn_in`.
  - Communication is counted only from `did_query`, which feeds `comm_mask_per_iter` and `comm_per_iter`.
  - `Coordinator.run` adds `comm_total`, `comm_per_agent`, `comm_cum` and `comm_rounds_iter`.
- **Local objectives** (`admm/local_models.py`), with `u = x − θ`:
  - `quadratic`: `uᵀΓu`
  - `poly4`: `+ λ4 Σu⁴`
  - `dir_poly4`: `+ λ4 Σ(Au)⁴`; requires `A`
- **Exact oracle** (`admm/moreau_oracle.py`):
  - `MoreauOracle(model, rho)` is the single exact oracle shared by every method; no caller passes solver settings.
  - **Quadratic:** closed-form Cholesky solve (`exact_quadsolver.py`).
  - **`poly4` / `dir_poly4`:** `_prox_quartic` solves the stationarity equation `∇f(x) + ρ(x − z) = 0` with `scipy.optimize.root(hybr)`, using Jacobian `∇²f + ρI` and starting from `x0 = z`. This is the paper's solver.
    - Fixed constants: `QUARTIC_ROOT_XTOL = 1e-10`, `QUARTIC_ROOT_MAXFEV = 300`, `QUARTIC_STATIONARITY_TOL = 1e-8`.
    - It raises `RuntimeError` unless the ∞-norm residual is ≤ 1e-8. Measured residuals are ~1e-11, at about 0.1 ms per solve.
    - Any other model type raises `TypeError`.
  - `moreau_value_and_grad` returns `(f_γ(z), ρ(z − prox))`.
- **GP target:** every observation is `[f_γ(z), ∇f_γ(z)]` (envelope value plus all gradient coordinates), modeled jointly with `RBFKernelGrad`. Both GP models use a zero mean, frozen.
  - SOGP: `gp/sogp_derivative_model.py`, with `MultitaskGaussianLikelihood(p+1)`.
  - MTGP: `mogp/` (`LMCGradKernel` = `Σ_q IndexKernel_q ⊗ RBFKernelGrad_q`). Task id is appended as the last input column; outputs are flattened per point.
- **Provider lifecycle.** All three providers live in `query_method/`. The cluster provider maps a global agent to `(cluster, local idx)` and delegates to one `MTGPMoreauGradProvider` per cluster. With `adapt_latents_to_cluster=True`, it uses `Q_c = ceil(n_c/2)`.
  1. **Burn-in:** query exactly and store `(z, y)` in `Z[i]`/`Y[i]`.
  2. **First fit:** once `k ≥ burn_in` and every agent has `min_points` samples.
  3. **Predict:** latent (noise-free) mean and variance, with exact predictive variance.
  4. **Project:** project the mean onto the intersection of up to 5 ellipsoids built from past samples (`utils/compute_ellipsoids.py`, `utils/projection_to_ell.py`). cvxpy/SCS is used only when the mean lies outside.
  5. **Gate:** query if `max(var) > psi_i^k = iota · V_i^{k0} · alpha^(k−k0)`, where `k0` is the agent's first GP decision (`*/threshold.py`).
  6. **Refit:** after a query, SOGP refits that agent immediately. MTGP refits once, at the start of the next iteration (when `i == 0`).
- **Output:** `utils/json_processing.process_output_to_jsonl` writes one JSON line per iteration: `primal_residual`, `dual_residual`, `obj_val`, `comm_per_iter`, `grads_error_per_iter` and `other_stats` (`g_var`, `var_max`, `psi_k`, `do_query`).

## Rules for the method

- **No oracle leakage.** Providers compute a hidden exact gradient at every prediction, only to log `grad_err`. It must never affect the query decision, the GP training data, or the gradient used by non-query agents.
- **`grad_err` is the raw prediction error, measured before projection**, in both SOGP and MTGP: `‖g_true − g_mean‖` straight from the GP. This compares the predictive ability of the models without the projection step. It is `0.0` on burn-in and exact-query steps, so exclude those (via `comm_per_iter` / `do_query`) when averaging.
- Keep the x-first ordering and the per-agent max-variance gate. ADMM_MTGP's y-first ordering and joint trace-budget query (STEP-GP theory track) are a separate algorithm; never mix them in.
- Changing `rho` changes the Moreau-gradient target. Don't reuse GP observations across different `rho` values.
- Fix seeds: all four runners call `set_seed(123)` at import; the GP runners also call `torch.set_default_dtype(torch.float32)`. CPU is the default.

## Known inconsistencies in the current code

All of these were verified in source. Fix them deliberately and note the numerical effect; don't fix them silently in passing.

- **`grad_err` is `0.0` when no prediction is used** (burn-in and query steps). The paper convention marks these as NaN. The user chose to keep `0.0`, so analysis must drop those entries.
- **Refit schedule.**
  - MTGP (global and cluster-wise) now takes separate `train_iter` (first fit) and `refit_iter` (each later refit, default 100) arguments, exposed in `run_mtgp_global` / `run_mtgp_clustered`. The old `self.train_iters = 100` override is commented out.
  - SOGP still refits with the full `train_iters` every time; it has no `refit_iter` yet.
- **Double exact solves.** On a query, providers solve the prox again even though the hidden exact gradient was just computed. ADMM_MTGP reuses it (`eval_exact_error`).
- **`eval_exact_error` is ignored.** `run_sogp_admm_cluster.py` passes `eval_exact_error=False`, but `SOGPMoreauProvider` only stores the flag and always computes the hidden exact gradient.
- **GP numerics intentionally differ from the paper.** The user chose these settings; keep them.
  - Here: exact predictive variance (`fast_pred_var(False)`), `max_cholesky_size(10000)`, MTGP noise floor `GreaterThan(1e-3)`, zero mean frozen.
  - Paper: `fast_pred_var=True`, `max_cholesky_size(2000)` for MTGP, noise floor 5e-4.
- **Leftover noise.**
  - `query_method/mogp/mtgp_provider.py` has an unused `from scipy.optimize import minimize`.
  - The quadratic oracle path prints "Using exact solver for quadratic local model" on every solve.
- **`run_mtgp_admm_cluster.py` is quadratic-only.** It passes no `local_models` to `Coordinator` and no `local_model` to the provider.
- **Ignored or dead parameters.**
  - `ADMMConfig.burn_in` is unused; the `Coordinator(burn_in=...)` argument is the one that's used.
  - The MTGP `num_tasks` argument is ignored; `N` is used.
  - `ExactLCMGrad.task_noise` references an undefined `noise_ceiling` and is never called.
  - `utils/logs.py`, `utils/metrics.py`, `utils/exp_io.py` and `query_method/mogp/custom_likelihood.py` are unused.
  - `query_method/mogp/__init.__.py` is misnamed; imports work as a namespace package.
- **The README is stale.**
  - It says `cd ADMM_MTGP` and lists `run_mtgp_admm.py`, `run_fullset_*`, `run_n_times.py`, `run_sweep_*` and `run_vanilla_admm_store_infor.py`. Those scripts are in `miscellaneous/old_scripts/` and need the repo root on `PYTHONPATH`.
  - It says `data_generation/` is empty; it is not.
  - `run_exp/plot_results.py` expects CSVs (`results/{vanilla,sogp}/Set1/iter_metrics.csv`) that the current runners don't write.

## Paper reference configuration (from ADMM_MTGP, for rederivation)

Frozen ICC profile, not this repo's current defaults:

- **ADMM:** x-first ordering, `rho=10`, `zeta=1`, `eps_abs=1e-6`, `eps_rel=1e-3`, at most 200 iterations, `burn_in=5`, float32, `fast_pred_var`.
- **SOGP:** 100 initial and 100 refit Adam steps, lr 0.01.
- **MTGP (global and cluster-wise):** 300 initial and 100 refit steps, lr 0.03, `rank=1`; global `Q=5`, cluster-wise `Q_c=ceil(n_c/2)`.
- **Objectives:** quadratic, coordinate quartic and directional quartic, with `lambda4=0.5`.
- **Sweep grid:** `iota ∈ {0.5,…,1.5}`, `alpha ∈ {0.80,…,0.99}`.
- **Regimes:** K1 homogeneous, K1 heterogeneous, K2/K3 separated, K2/K3 overlapping. Skip cluster-wise MTGP on K1, where it equals global MTGP.
- **Final selection:**
  - `iota ∈ [0.5, 1.0]`; SOGP `alpha ∈ [0.80, 0.90]`, MTGP `alpha ∈ [0.90, 0.99]`.
  - Require convergence, objective relative error < 0.05%, `n_vanilla < n_model < 2·n_vanilla`, and fewer total exact queries than vanilla.
  - Keep the three closest distinct iteration counts above vanilla, then rank by total queries, then objective error, then iterations.
  - Query rate and query-active iterations are diagnostics only.

## Direction for new code

Things worth borrowing from ADMM_MTGP:

- explicit `refit_iters`
- an `eval_exact_error` flag
- NaN for "no prediction"
- a `verbose` switch
- the objective-family mapping (`quadratic`/`coordinate_quartic`/`directional_quartic` → `quadratic`/`poly4`/`dir_poly4`)
- solver state histories plus the primal objective
- one summary CSV row plus one `.npz` trajectory per run, recording every setting
- `--device auto|cpu|cuda`

Keep it simple: one runner per method, plain config, no subprocess orchestration, no compatibility shims for retired variants.
