## adapt the cluster specific mtgp -> cluster data

## create a list of MTGP -> when call the multi_cluster_mtgp as the provider -> based on the index (i)
## => select the model -> predict -> return the result

## need map the global index, query -> local cluster index -> query
## inherite the global MTGP in side the cluster mtpg

import torch
import numpy as np

from admm import local_models

# threshold adaptive and decision rule
from .mtgp_provider import MTGPMoreauGradProvider

class ClusterMTGPProvider:
    def __init__(
        self,
        gamma_list,
        theta_list,
        labels,
        rho: float,
        zeta: float,
        min_points: int = 5,
        iota: float = 1.0,
        alpha: float = 0.96,
        train_iter: int = 300,
        lr: float = 0.03,
        num_latents: int = 5,
        rank: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        adapt_latents_to_cluster: bool = False,
        noise_specific_task: bool = False,
        local_model: str = "quadratic", # "quadratic" or "poly4"
        lambda4: float = 0.05,
        direct_matrix=None,
        solver_method="trust-constr",
        solver_tol=1e-6,
        solver_maxiter=500
        
    ):
        self.gamma_list = gamma_list
        self.theta_list = theta_list
        self.labels = np.asarray(labels, dtype=int).reshape(-1)

        self.rho = float(rho)
        self.zeta = float(zeta)
        self.min_points = int(min_points)
        self.iota = float(iota)
        self.alpha = float(alpha)

        # mtgp model
        self.train_iter = int(train_iter)
        self.lr = float(lr)
        self.num_latents = int(num_latents)
        self.rank = int(rank)
        self.device = device
        self.dtype = dtype
        self.adapt_latents_to_cluster = bool(adapt_latents_to_cluster)
        self.noise_specific_task = bool(noise_specific_task)

        self.N = len(gamma_list)

        #define the local function 

        self.local_model = local_model
        self.lambda4 = float(lambda4)
        self.solver_method = solver_method
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter
        self.direct_matrix = direct_matrix

        # cluster indexing
        self.cluster_ids =  sorted(np.unique(self.labels).tolist()) # sorted

        # mapping: cluster id -> list of global agent index

        # set of agent for each cluster sorted by cluster id
        self.cluster_to_global = {
            c: np.where(self.labels == c)[0].tolist() for c in self.cluster_ids
        }

        self.global_to_local = {}

        # cluster_id -> provider
        self.providers = {}

        for c in self.cluster_ids:
            global_ids = self.cluster_to_global[c] ## global agent idx in the cluster

            cluster_size = len(global_ids) ## num agent in cluster

            if cluster_size == 0:
                continue

            # getting the gamma and theta for the cluster
            clust_gamma = [self.gamma_list[g] for g in global_ids]
            clust_theta = [self.theta_list[g] for g in global_ids]

            # map the global idx to local idx inside the cluster
            # ex: cluster 0 has global agent [0, 2, 5] -> local idx [0, 1, 2]

            for local_idx, global_idx in enumerate(global_ids):
                ## global idx -> (cluster id, local idx)
                self.global_to_local[global_idx] = (c, local_idx) 

            # num latents each clusters
            q_cluster = self.num_latents

            # max latent = cluster size
            if self.adapt_latents_to_cluster:
                # q_cluster = half and round up the num agents in cluster
                q_cluster = int(np.ceil(cluster_size / 2))
                # q_cluster = min(cluster_size, self.num_latents)

            print("Number latents for cluster {} with size {}: {}".format(c, cluster_size, q_cluster))

            ## define the mtgp for the cluster
            print("Function form for the cluster MTGP provider:", self.local_model)
            self.providers[c] = MTGPMoreauGradProvider(
                gamma_list=clust_gamma,
                theta_list=clust_theta,
                rho=self.rho,
                zeta=self.zeta,
                min_points=self.min_points,
                iota=self.iota,
                alpha=self.alpha,
                train_iter=self.train_iter,
                lr=self.lr,
                num_latents=q_cluster,
                rank=self.rank,
                device=self.device,
                dtype=self.dtype,
                specific_task_noise=self.noise_specific_task,
                local_model=self.local_model,
                lambda4=self.lambda4,
                solver_method=self.solver_method,
                solver_tol=self.solver_tol,
                solver_maxiter=self.solver_maxiter,
                direct_matrix=self.direct_matrix
            )

    def get_cluster_summary(self):
        return {
            "cluster_ids": [c for c in self.cluster_ids],

            "cluster_to_global": {
                c: [g for g in ids]
                for c, ids in self.cluster_to_global.items()
            },

            "cluster_sizes": {
                s: len(ids)
                for s, ids in self.cluster_to_global.items()
            },
        }

    def __call__(self, i: int, z_i: np.ndarray, query_flag: bool, k: int , *args, **kargs):
        
        if i not in self.global_to_local:
            raise ValueError(f"Global index {i} not found in any cluster or check the data query data type")
        
        # get the mapping global idx -> cluster id and local idx of the agent in the cluster
        cluster_id, local_idx = self.global_to_local[i]
        provider = self.providers[cluster_id]

        print(f"Prediction for the agent {i + 1} using the MTGP of cluster {cluster_id} with its local idx {local_idx}")
        
        # get return from the agent specific provider (cluster mtgp)
        ret = provider(local_idx, z_i, query_flag, k, *args, **kargs)

        return ret