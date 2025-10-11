# @Description: Training script for RadioMambaNet v14. Simplified version with single model configuration,
#              removing multi-scale model selection logic from v12.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Import local modules
from dataset import RadioMambaNetDataset
from model import Model as RadioModel

# Import Mamba for the check
from mamba_ssm import Mamba

try:
    from torchmetrics import StructuralSimilarityIndexMeasure
except ImportError:
    StructuralSimilarityIndexMeasure = None
    print("Warning: torchmetrics not found. SSIM loss and metric will be unavailable.")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# The LightningRadioModel and RadioMapSeerDataModule classes remain identical
# in logic to your v11. I am including them here for completeness with
# updated import paths.


class LightningRadioModel(pl.LightningModule):
    def __init__(self, model_params, training_config):
        super().__init__()
        self.save_hyperparameters()
        self.model_params = model_params
        self.training_config = training_config

        self.model = RadioModel(
            in_channels=self.model_params['in_channels'],
            out_channels=self.model_params['out_channels'],
            dims=self.model_params['dims'],
            depths=self.model_params['depths'],
            ssm_d_state=self.model_params['ssm_d_state'],
            ssm_d_conv=self.model_params['ssm_d_conv'],
            ssm_expand=self.model_params['ssm_expand']
        )
        self.learning_rate = self.training_config['learning_rate']
        self.weight_decay = self.training_config['weight_decay']

        self.l1_loss_fn = nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()

        if StructuralSimilarityIndexMeasure is not None:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        else:
            self.ssim_metric = None

        loss_weights_cfg = self.training_config.get('loss_weights', {})
        self.l1_weight = loss_weights_cfg.get('l1', 0.0)
        self.mse_weight = loss_weights_cfg.get('mse', 1.0)
        self.ssim_weight = loss_weights_cfg.get('ssim', 0.0)
        self.gradient_weight = loss_weights_cfg.get('gradient', 0.0)

        sobel_x_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x_kernel)
        self.register_buffer('sobel_y', sobel_y_kernel)

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _calculate_gradient_loss(self, preds, targets):
        preds_c = preds if preds.ndim == 4 and preds.size(
            1) == 1 else preds.unsqueeze(1)
        targets_c = targets if targets.ndim == 4 and targets.size(
            1) == 1 else targets.unsqueeze(1)
        pred_grad_x = F.conv2d(preds_c, self.sobel_x, padding='same')
        pred_grad_y = F.conv2d(preds_c, self.sobel_y, padding='same')
        target_grad_x = F.conv2d(targets_c, self.sobel_x, padding='same')
        target_grad_y = F.conv2d(targets_c, self.sobel_y, padding='same')
        return (self.l1_loss_fn(pred_grad_x, target_grad_x) + self.l1_loss_fn(pred_grad_y, target_grad_y)) / 2.0

    def _calculate_combined_loss(self, outputs, targets):
        outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
        loss_l1_val = self.l1_loss_fn(
            outputs_clamped, targets) if self.l1_weight > 0 else torch.tensor(0.0, device=outputs.device)
        loss_mse_val = self.mse_loss_fn(
            outputs_clamped, targets) if self.mse_weight > 0 else torch.tensor(0.0, device=outputs.device)
        loss_gradient_val = self._calculate_gradient_loss(
            outputs_clamped, targets) if self.gradient_weight > 0 else torch.tensor(0.0, device=outputs.device)

        ssim_score_val = None
        loss_ssim_val = torch.tensor(0.0, device=outputs.device)
        if self.ssim_metric is not None and self.ssim_weight > 0:
            self.ssim_metric = self.ssim_metric.to(outputs_clamped.device)
            ssim_score_val = self.ssim_metric(outputs_clamped, targets)
            loss_ssim_val = 1.0 - ssim_score_val

        total_loss = (self.l1_weight * loss_l1_val +
                      self.mse_weight * loss_mse_val +
                      self.ssim_weight * loss_ssim_val +
                      self.gradient_weight * loss_gradient_val)
        return total_loss, loss_l1_val, loss_mse_val, ssim_score_val, loss_gradient_val

    def training_step(self, batch, batch_idx):
        inputs, targets, *_ = batch
        outputs = self(inputs)
        total_loss, train_l1, train_mse, train_ssim_score, train_gradient = self._calculate_combined_loss(
            outputs, targets)
        self.log_dict({
            'train_total_loss': total_loss,
            'train_l1_loss': train_l1,
            'train_mse_loss': train_mse,
            'train_ssim_score': train_ssim_score,
            'train_gradient_loss': train_gradient
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, *_ = batch
        outputs = self(inputs)
        total_loss, val_l1, val_mse, val_ssim_score, val_gradient = self._calculate_combined_loss(
            outputs, targets)
        self.log_dict({
            'val_total_loss': total_loss,
            'val_l1_loss': val_l1,
            'val_mse_loss': val_mse,
            'val_ssim_score': val_ssim_score,
            'val_gradient_loss': val_gradient
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.trainer.global_rank == 0 and batch_idx == 0:
            self.validation_step_outputs.append({
                'inputs': inputs.cpu().detach(),
                'targets': targets.cpu().detach(),
                'preds': torch.clamp(outputs.cpu().detach(), 0.0, 1.0)
            })
        return total_loss

    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0 and self.validation_step_outputs:
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, targets, *_ = batch
        outputs = self(inputs)
        total_loss, test_l1, test_mse, test_ssim_score, test_gradient = self._calculate_combined_loss(
            outputs, targets)
        self.log_dict({
            'test_total_loss': total_loss,
            'test_l1_loss': test_l1,
            'test_mse_loss': test_mse,
            'test_ssim_score': test_ssim_score,
            'test_gradient_loss': test_gradient
        }, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=self.hparams.training_config['lr_scheduler_patience'], verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss"}}


class RadioMapSeerDataModule(pl.LightningDataModule):
    def __init__(self, data_config: dict, seed: int = 42):
        super().__init__()
        self.dir_dataset = data_config['dataset_root_dir']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']
        self.seed = seed
        self.dataset_params = {
            "dir_dataset": self.dir_dataset,
            "numTx": data_config['num_tx_per_map'],
            "carsSimul": data_config.get('cars_simul', 'no'),
            "carsInput": data_config.get('cars_input', 'no'),
            "transform": transforms.ToTensor()
        }
        maps_inds_shuffled = np.arange(0, 700, 1, dtype=np.int16)
        np.random.seed(self.seed)
        np.random.shuffle(maps_inds_shuffled)
        self.maps_inds_shuffled = maps_inds_shuffled

    def prepare_data(self):
        if not os.path.exists(self.dir_dataset):
            raise FileNotFoundError(
                f"Dataset path {self.dir_dataset} not found.")

    def setup(self, stage: str = None):
        dataset_args = {**self.dataset_params,
                        "maps_inds": self.maps_inds_shuffled}
        if stage == 'fit' or stage is None:
            self.train_dataset = RadioMambaNetDataset(
                phase="train", **dataset_args)
            self.val_dataset = RadioMambaNetDataset(
                phase="val", **dataset_args)
        if stage == 'test' or stage is None:
            self.test_dataset = RadioMambaNetDataset(
                phase="test", **dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)

# The ValidationImageSaver class logic is unchanged.


class ValidationImageSaver(Callback):
    def __init__(self, save_dir, num_samples=4, log_to_tensorboard=True):
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.log_to_tensorboard = log_to_tensorboard

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if trainer.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.global_rank != 0 or not pl_module.validation_step_outputs:
            return

        batch_data = pl_module.validation_step_outputs[0]
        targets, preds = batch_data['targets'], batch_data['preds']
        num_to_show = min(self.num_samples, targets.size(0))

        combined_grid = make_grid(
            torch.cat([targets[:num_to_show], preds[:num_to_show]]), nrow=num_to_show)

        # Log to TensorBoard
        if self.log_to_tensorboard and trainer.logger and hasattr(trainer.logger.experiment, 'add_image'):
            trainer.logger.experiment.add_image(
                "Validation/GT_vs_Pred", combined_grid, global_step=trainer.global_step)

        # Save to file
        fig, ax = plt.subplots(figsize=(num_to_show * 3, 7))
        ax.imshow(combined_grid.permute(1, 2, 0).cpu().numpy())
        ax.set_title(
            f"Ground Truth (top) vs. Predictions (bottom) at Step {trainer.global_step}")
        ax.axis('off')
        save_path = os.path.join(
            self.save_dir, f"step_{trainer.global_step}_val_samples.png")
        plt.savefig(save_path)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RadioMambaNet v14")
    parser.add_argument('--config', type=str, default='../configs/config_withcars.yaml',
                        help='Path to the YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config['seed'], workers=True)

    if Mamba is None:
        print("Fatal: `mamba_ssm` is required for this model. Please install it.")
        exit(1)

    # Build model parameters directly from config (no more multi-scale)
    model_params = {
        'in_channels': config['Model']['in_channels'],
        'out_channels': config['Model']['out_channels'],
        'dims': config['Model']['dims'],
        'depths': config['Model']['depths'],
        'ssm_d_state': config['Model']['ssm_d_state'],
        'ssm_d_conv': config['Model']['ssm_d_conv'],
        'ssm_expand': config['Model']['ssm_expand']
    }

    # Use training config directly
    training_config = config['training'].copy()

    # Use data config directly
    data_config = config['data'].copy()

    data_module = RadioMapSeerDataModule(
        data_config=data_config, seed=config['seed'])
    lightning_model = LightningRadioModel(
        model_params=model_params, training_config=training_config)

    # Setup callbacks (no more model level substitution)
    callbacks_list = []
    callbacks_cfg = config.get('callbacks', {})

    if 'checkpoint_best' in callbacks_cfg:
        callbacks_list.append(ModelCheckpoint(**callbacks_cfg['checkpoint_best']))

    if 'checkpoint_latest' in callbacks_cfg:
        callbacks_list.append(ModelCheckpoint(**callbacks_cfg['checkpoint_latest']))

    if 'early_stopping' in callbacks_cfg:
        callbacks_list.append(EarlyStopping(**callbacks_cfg['early_stopping']))

    if 'save_validation_images' in callbacks_cfg:
        callbacks_list.append(ValidationImageSaver(**callbacks_cfg['save_validation_images']))

    # Setup trainer config directly
    trainer_cfg = config['trainer_config'].copy()

    # Setup logging
    logging_cfg = config['logging']['tensorboard'].copy()

    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks_list,
        logger=pl.loggers.TensorBoardLogger(**logging_cfg)
    )

    # Check if resume from checkpoint is specified
    resume_checkpoint = training_config.get('resume_from_checkpoint', '')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.fit(lightning_model, datamodule=data_module,
                    ckpt_path=resume_checkpoint)
    else:
        if resume_checkpoint:
            print(
                f"Warning: Checkpoint path '{resume_checkpoint}' not found. Starting from scratch.")
        trainer.fit(lightning_model, datamodule=data_module)
