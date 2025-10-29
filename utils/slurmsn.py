# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

import os
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy

def setup_slurm_training():
    """
    Set up SLURM-specific configurations for distributed training
    """
    # Check if running on SLURM
    if "SLURM_JOB_ID" in os.environ:
        # Get SLURM job ID
        job_id = os.environ["SLURM_JOB_ID"]
        
        # Configure PyTorch Lightning for SLURM
        slurm_env = SLURMEnvironment()
        strategy = DDPStrategy(
            cluster_environment=slurm_env,
            find_unused_parameters=True
        )
        
        # Get the number of GPUs per node
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
        
        # Get the number of nodes
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        
        return {
            "strategy": strategy,
            "devices": gpus_per_node,
            "num_nodes": num_nodes,
            "job_id": job_id
        }
    else:
        # Not running on SLURM
        return {
            "strategy": "ddp",
            "devices": 1,
            "num_nodes": 1,
            "job_id": None
        }