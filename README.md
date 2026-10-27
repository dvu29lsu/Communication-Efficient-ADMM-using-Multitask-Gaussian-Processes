# Communication-Efficient ADMM using Multitask Gaussian Processes

In sharing ADMM, each agent sends the gradient of its Moreau envelope to the coordinator at every iteration. That requires one exact local solve and one message per agent per iteration. Here, a Gaussian process learns these gradients from past queries, and an agent only queries exactly (and communicates) when the GP's predictive variance is above an adaptive threshold.

We compare four methods:

- **Vanilla ADMM**: every agent queries exactly at every iteration (the baseline).
- **SOGP-ADMM**: one independent derivative GP per agent.
- **MTGP-ADMM**: one multitask derivative GP shared by all agents.
- **Cluster-wise MTGP-ADMM**: one multitask GP per cluster of similar agents.

## Requirements

Python 3.11. Install the packages with

```bash
pip install -r requirements.txt
```

Everything runs on CPU. The GP models use float64.

## Data

The datasets are in `data_generation/`. Each one has 10 agents in dimension 5:

- `homogeneous.json`
- `heterogeneous.json`
- `overlapping_2_clusters.json`, `overlapping_3_clusters.json`
- `separated_2_clusters.json`, `separated_3_clusters.json`

Each file stores `gamma_list` (10 x 5 x 5), `theta_list` (10 x 5) and `labels` (the cluster of each agent). Agent `i` has the quadratic local function `f_i(x) = (x - theta_i)^T Gamma_i (x - theta_i)`.

## Running the quadratic experiments

Run the scripts from the repository root:

```bash
python run_vanilla_admm_cluster.py   # vanilla ADMM
python run_sogp_admm_cluster.py      # SOGP-ADMM
python run_mtgp_admm_cluster.py      # MTGP-ADMM
python run_multi_cluster_mtgp.py     # cluster-wise MTGP-ADMM
```

The scripts have no command-line arguments. To change the dataset, edit `data_path` in the `main()` of each script. Keep `local_model = "quadratic"`.

All scripts share these settings:

- ADMM: `zeta = 1.0`, `rho = 10`, `max_iters = 150`, `eps_abs = 1e-6`, `eps_rel = 1e-3`
- GP: `burn_in = 5` exact queries per agent before the GP starts
- seed: 123

Default GP settings for each method:

| Method | iota | alpha | Training |
|---|---|---|---|
| SOGP-ADMM | 0.8 | 0.96 | 100 Adam iterations, lr 0.01 |
| MTGP-ADMM | 1.0 | 0.96 | 5 latent processes, 300 Adam iterations, lr 0.03 |
| Cluster-wise MTGP-ADMM | 1.0 | 0.96 | ceil(cluster size / 2) latent processes, 300 Adam iterations, lr 0.03 |

The query threshold is `psi_k = iota * V_k0 * alpha^(k - k0)`, where `V_k0` is the first predicted max variance of the agent.

## Output

Each run writes `test/test_<method>_<dataset>.jsonl` with one line per ADMM iteration:

- `primal_residual`, `dual_residual`, `obj_val`
- `comm_per_iter`: number of agents that queried exactly in this iteration
- `grads_error_per_iter`: error between the GP gradient (before projection) and the exact gradient for each agent; it is 0 when the agent queried
- `other_stats`: per-agent `g_var`, `var_max`, `psi_k` and `do_query`
