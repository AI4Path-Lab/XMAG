# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

import os
import yaml
import argparse
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb


from models.distiller import PathologyDistiller
from data.dataset import PathologyDataModule
from utils.slurm import setup_slurm_training

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 * 1024)  # Increase to 100MB


def main(args):
    # load configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Save a copy of the config file to output directory
    config_backup_path = os.path.join(args.output_dir, 'config.yaml')
    shutil.copy2(args.config, config_backup_path)
    print(f"Config file saved to: {config_backup_path}")

    # setup seed for reproducibility
    pl.seed_everything(config['project']['seed'], workers=True)

    # Initialize data module
    data_module = PathologyDataModule(config)

    # Initialize model
    model = PathologyDistiller(config)

    # Setup callbacks
    callbacks = [ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename="epoch_{epoch:02d}-val_loss-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    ), 
    # Checkpoint every N epochs (e.g., every 5 epochs)
    ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints', 'periodic'),
        filename="periodic_epoch_{epoch:02d}",
        every_n_epochs=5,  # Save every 5 epochs
        save_top_k=-1,  # Keep all periodic checkpoints
        auto_insert_metric_name=False
    ),
    LearningRateMonitor(logging_interval='step')]

    # Setup logger
    wandb.login(key="xxxxx")
    
    run_name = args.run_name if args.run_name else f"{config['project']['name']}_{config['project']['seed']}"
    tags = args.tags.split(',') if args.tags else None
    
    # Set up SLURM configuration
    slurm_config = setup_slurm_training()
    
    # Prepare initial config with SLURM info
    initial_config = config.copy()
    if slurm_config["job_id"]:
        initial_config.update({
            "slurm_job_id": slurm_config["job_id"],
            "total_gpus": slurm_config["total_gpus"],
            "effective_batch_size": config['data']['batch_size'] * slurm_config["total_gpus"]
        })
    
    wandb_logger = WandbLogger(
        project=config['project']['name'],
        name=run_name,
        save_dir=os.path.join(args.output_dir, 'logs'),
        tags=tags,
        config=initial_config  # Pass the complete config here
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['scheduler']['max_epochs'],
        precision=config['project']['precision'],
        gradient_clip_val=config['trainer']['gradient_clip_val'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        val_check_interval=config['trainer']['val_check_interval'],
        callbacks=callbacks,
        logger=wandb_logger,
        strategy=slurm_config["strategy"],
        devices=slurm_config["devices"],
        num_nodes=slurm_config["num_nodes"],
    )

    # Print training setting including number of GPUs and nodes and total batch size
    print(f"Training with {slurm_config['total_gpus']} GPUs across {slurm_config['num_nodes']} nodes.")
    print(f"Effective batch size: {config['data']['batch_size'] * slurm_config['total_gpus']}")
    print(f"Run name: {run_name}")
    print(f"Tags: {tags}")

    # # Initialize Trainer
    # from pytorch_lightning.strategies import DDPStrategy
    # from pytorch_lightning.plugins.environments import SLURMEnvironment

    # slurm_env = SLURMEnvironment()

    # trainer = pl.Trainer(
    #     max_epochs=config['scheduler']['max_epochs'],
    #     precision=config['project']['precision'],
    #     gradient_clip_val=config['trainer']['gradient_clip_val'],
    #     log_every_n_steps=config['trainer']['log_every_n_steps'],
    #     val_check_interval=config['trainer']['val_check_interval'],
    #     callbacks=callbacks,
    #     logger=wandb_logger,
    #     accelerator="gpu", devices=2, num_nodes=2, strategy=DDPStrategy(
    #         cluster_environment=slurm_env,
    #         find_unused_parameters=True
    #     )
    # )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pathology distillation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save outputs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run in wandb")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for wandb run")
    args = parser.parse_args()

    # Set the output directory in the config
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
