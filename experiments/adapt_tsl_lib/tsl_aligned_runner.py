#!/usr/bin/env python3
"""TSL-Aligned Runner for Liulian Transformer Models.

This experimental script runs Liulian's Transformer with training loop
exactly matching TSL's exp_long_term_forecasting.py to eliminate variance
from implementation differences.

Key alignment points:
1. Same random seed handling
2. Same data loader construction
3. Same training loop order (optimizer.zero_grad, forward, loss, backward, step)
4. Same LR adjustment timing (after early stopping check)
5. Same early stopping on vali_loss only
6. Same batch construction for decoder input

Usage:
    python tsl_aligned_runner.py --config experiments/etth1/transformer_config.yaml
"""

import argparse
import logging
import os
import random
import sys
import time
import warnings
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from liulian.config import load_config
from liulian.data.data_factory import create_dataloader
from liulian.models.torch.transformer import TransformerAdapter
from liulian.models.torch.training_utils import EarlyStopping
from liulian.optim.lr_schedulers import adjust_learning_rate

# Import float64-aligned dataloader for TSL matching
from experiments.adapt_tsl_lib.tsl_float64_dataloader import create_tsl_aligned_dataloader

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_seed(seed: int = 2021):
    """Set random seeds exactly like TSL run.py."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set CUBLAS workspace config for deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable PyTorch deterministic algorithms (may raise errors for non-deterministic ops)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Not supported on all PyTorch versions


def create_dataloader_tsl_style(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    """Create DataLoader matching TSL's data_provider."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,  # TSL uses drop_last=True
    )


class TSLAlignedTrainer:
    """Training loop that exactly mirrors TSL exp_long_term_forecasting.py."""

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pred_len = config.get('pred_len', 96)
        self.label_len = config.get('label_len', 48)
        self.features = config.get('features', 'M')
        self.use_amp = config.get('use_amp', False)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/tsl_aligned')

    def _select_criterion(self):
        """TSL uses plain MSELoss."""
        return nn.MSELoss()

    def _select_optimizer(self, model):
        """TSL uses Adam with configured learning_rate."""
        return optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.0001))

    def vali(self, model, vali_loader, criterion):
        """Validation exactly like TSL."""
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()  # Keep on CPU initially like TSL

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input construction exactly like TSL
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        model.train()
        return total_loss

    def train(self, model, train_loader, vali_loader, test_loader):
        """Training loop exactly like TSL."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_epochs = self.config.get('train_epochs', 10)
        patience = self.config.get('patience', 3)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_mode=True)

        model_optim = self._select_optimizer(model)
        criterion = self._select_criterion()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        history = []

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                
                # EXACTLY like TSL: zero_grad FIRST
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input exactly like TSL
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.features == 'MS' else 0
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, -self.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(f"iters: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_time_taken = time.time() - epoch_time
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(model, vali_loader, criterion)
            test_loss = self.vali(model, test_loader, criterion)

            logger.info(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss_avg,
                'vali_loss': vali_loss,
                'test_loss': test_loss,
                'time': epoch_time_taken,
            })

            # EXACTLY like TSL: early_stopping BEFORE adjust_learning_rate
            early_stopping(vali_loss, model, self.checkpoint_dir)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            # EXACTLY like TSL: adjust_learning_rate AFTER early_stopping check
            adjust_learning_rate(model_optim, epoch + 1, self.config)

        # Load best model
        best_model_path = os.path.join(self.checkpoint_dir, 'checkpoint')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            logger.info(f"Loaded best model from {best_model_path}")

        return history, best_model_path


def main():
    # IMPORTANT: Set seed FIRST, before argparse like TSL does
    # This ensures the random state is identical when model weights are initialized
    fix_seed(2021)  # Default seed, will be overridden by --seed if provided
    
    parser = argparse.ArgumentParser(description='TSL-Aligned Liulian Runner')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed (default: 2021 like TSL)')
    parser.add_argument('--train_epochs', type=int, default=None, help='Override train epochs')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.train_epochs is not None:
        config['train_epochs'] = args.train_epochs

    # Reset seed if different from default (TSL uses 2021)
    if args.seed != 2021:
        fix_seed(args.seed)

    logger.info(f"Running TSL-aligned training with seed={args.seed}")
    logger.info(f"Config: {args.config}")

    # Create dataloaders using factory
    data_name = config.get('data', 'ETTh1')
    
    # Use dataset-specific paths matching TSL config
    DATASET_PATHS = {
        'ETTh1': ('./dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('./dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('./dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('./dataset/ETT-small/', 'ETTm2.csv', 7),
        'weather': ('./dataset/weather/', 'weather.csv', 21),
        'electricity': ('./dataset/electricity/', 'electricity.csv', 321),
        'traffic': ('./dataset/traffic/', 'traffic.csv', 862),
        'exchange_rate': ('./dataset/exchange_rate/', 'exchange_rate.csv', 8),
        'illness': ('./dataset/illness/', 'national_illness.csv', 7),
    }
    
    if data_name in DATASET_PATHS:
        root_path, data_path, enc_in = DATASET_PATHS[data_name]
    else:
        root_path = config.get('root_path', 'dataset')
        data_path = config.get('data_path', f'{data_name}.csv')
        enc_in = config.get('enc_in', 7)
    
    logger.info(f"Creating model and dataloaders: {data_name}")
    logger.info(f"Path: {root_path}/{data_path}, enc_in: {enc_in}")

    seq_len = config.get('seq_len', 96)
    label_len = config.get('label_len', 48)
    pred_len = config.get('pred_len', 96)
    batch_size = config.get('batch_size', 32)

    # Create MODEL FIRST (before dataloaders) to match TSL's random state consumption order
    model_config = {
        'seq_len': seq_len,
        'pred_len': pred_len,
        'label_len': label_len,
        'enc_in': enc_in,
        'dec_in': enc_in,
        'c_out': enc_in,
        'd_model': config.get('d_model', 512),
        'n_heads': config.get('n_heads', 8),
        'e_layers': config.get('e_layers', 2),
        'd_layers': config.get('d_layers', 1),
        'd_ff': config.get('d_ff', 2048),
        'factor': config.get('factor', 1),
        'dropout': config.get('dropout', 0.05),
        'activation': config.get('activation', 'gelu'),
        'embed': config.get('embed', 'timeF'),
        'freq': config.get('freq', 'h'),
        'task_name': 'long_term_forecast',
    }

    adapter = TransformerAdapter(model_config)
    model = adapter._model  # Access internal model for raw torch training
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Model created on {device}")
    logger.info(f"Model config: enc_in={enc_in}, d_model={model_config['d_model']}, "
                f"n_heads={model_config['n_heads']}, e_layers={model_config['e_layers']}")
    
    # Extract root_path and data_path for dataloaders (enc_in already extracted above)
    if data_name in DATASET_PATHS:
        root_path, data_path, _ = DATASET_PATHS[data_name]
    else:
        root_path = config.get('root_path', 'dataset')
        data_path = config.get('data_path', f'{data_name}.csv')

    # Create DATALOADERS after model (but model already created above)
    # Use float64-aligned dataloader to match TSL exactly
    use_float64 = config.get('use_float64', True)
    
    if use_float64:
        logger.info("Using float64-aligned dataloader (matching TSL)")
        train_loader = create_tsl_aligned_dataloader(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        val_loader = create_tsl_aligned_dataloader(
            root_path=root_path,
            data_path=data_path,
            flag='val',
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        
        test_loader = create_tsl_aligned_dataloader(
            root_path=root_path,
            data_path=data_path,
            flag='test',
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
    else:
        # Use Liulian's native float32 dataloaders
        logger.info("Using Liulian float32 dataloader")
        train_loader = create_dataloader(
            data_name=data_name,
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(seq_len, label_len, pred_len),
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            scale=True,
            timeenc=1 if config.get('embed', 'timeF') == 'timeF' else 0,
            freq=config.get('freq', 'h'),
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True,
        )
        
        val_loader = create_dataloader(
            data_name=data_name,
            root_path=root_path,
            data_path=data_path,
            flag='val',
            size=(seq_len, label_len, pred_len),
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            scale=True,
            timeenc=1 if config.get('embed', 'timeF') == 'timeF' else 0,
            freq=config.get('freq', 'h'),
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )
        
        test_loader = create_dataloader(
            data_name=data_name,
            root_path=root_path,
            data_path=data_path,
            flag='test',
            size=(seq_len, label_len, pred_len),
            features=config.get('features', 'M'),
            target=config.get('target', 'OT'),
            scale=True,
            timeenc=1 if config.get('embed', 'timeF') == 'timeF' else 0,
            freq=config.get('freq', 'h'),
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Create trainer config
    trainer_config = {
        'pred_len': config.get('pred_len', 96),
        'label_len': config.get('label_len', 48),
        'features': config.get('features', 'M'),
        'train_epochs': config.get('train_epochs', 10),
        'patience': config.get('patience', 3),
        'learning_rate': config.get('learning_rate', 0.0001),
        'lradj': config.get('lradj', 'type1'),
        'use_amp': config.get('use_amp', False),
        'checkpoint_dir': f'checkpoints/tsl_aligned_{data_name}_transformer',
        'use_float64': config.get('use_float64', True),  # Match TSL's internal dtype
    }

    trainer = TSLAlignedTrainer(trainer_config, device=str(device))
    history, best_model_path = trainer.train(model, train_loader, val_loader, test_loader)

    # Get best epoch metrics
    best_epoch = min(history, key=lambda x: x['vali_loss'])
    
    # Compute test loss with best model
    model.eval()
    criterion = nn.MSELoss()
    final_test_loss = trainer.vali(model, test_loader, criterion)
    
    # Print final results
    logger.info("=" * 60)
    logger.info("TSL-Aligned Training Complete")
    logger.info(f"Total Epochs Run: {len(history)}")
    logger.info(f"Best Epoch (lowest vali): {best_epoch['epoch']}")
    logger.info(f"Best Vali Loss (MSE): {best_epoch['vali_loss']:.6f}")
    logger.info(f"Final Test Loss (MSE, using best model): {final_test_loss:.6f}")
    logger.info("=" * 60)

    # Print in format compatible with compare script
    print(f"Liulian final MSE: {final_test_loss:.6f}")
    print(f"Liulian epochs run: {len(history)}")


if __name__ == '__main__':
    main()
