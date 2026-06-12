#!/usr/bin/env python3
"""Identical Comparison Script - Zero Randomness.

This script ensures BYTE-IDENTICAL results between TSL and Liulian by:
1. Using TSL's exact data provider (same scaler)
2. Using identical model weights
3. Disabling all randomness sources
4. Running without early stopping
5. Comparing epoch-by-epoch losses

Usage:
    python identical_comparison.py --dataset ETTh1 --epochs 3
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'refer_projects/Time-Series-Library'))


def set_deterministic(seed: int = 2021):
    """Set ALL random seeds and enable deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_tsl_args(dataset: str, root_path: str):
    """Create TSL Args object."""
    class Args:
        task_name = 'long_term_forecast'
        is_training = 1
        data = dataset
        data_path = f'{dataset}.csv'
        features = 'M'
        target = 'OT'
        freq = 'h'
        checkpoints = 'checkpoints'
        seq_len = 96
        label_len = 48
        pred_len = 96
        seasonal_patterns = 'Monthly'
        inverse = False
        mask_rate = 0.25
        anomaly_ratio = 0.25
        top_k = 5
        num_kernels = 6
        enc_in = 7
        dec_in = 7
        c_out = 7
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        moving_avg = 25
        factor = 3
        distil = True
        dropout = 0.0  # ZERO dropout for determinism
        embed = 'timeF'
        activation = 'gelu'
        output_attention = False
        batch_size = 32
        num_workers = 0
        augmentation_ratio = 0
        channel_independence = 0
    
    args = Args()
    args.root_path = root_path
    return args


def train_one_epoch(model, loader, optimizer, criterion, device, label_len, pred_len):
    """Train one epoch and return losses per batch."""
    model.train()
    batch_losses = []
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        optimizer.zero_grad()
        
        # Decoder input
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
        
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs[:, -pred_len:, :]
        target = batch_y[:, -pred_len:, :]
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
    
    return batch_losses


def evaluate(model, loader, criterion, device, label_len, pred_len):
    """Evaluate and return mean loss."""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            target = batch_y[:, -pred_len:, :]
            
            loss = criterion(outputs, target)
            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    
    return total_loss / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Paths
    DATASET_PATHS = {
        'ETTh1': 'dataset/ETT-small/',
        'ETTh2': 'dataset/ETT-small/',
        'ETTm1': 'dataset/ETT-small/',
        'ETTm2': 'dataset/ETT-small/',
    }
    root_path = os.path.join(PROJECT_ROOT, DATASET_PATHS.get(args.dataset, 'dataset/'))
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # =========================================================================
    # TSL TRAINING
    # =========================================================================
    print("\n[TSL] Setting up...")
    set_deterministic(args.seed)
    
    from models.Transformer import Model as TSLModel
    from data_provider.data_factory import data_provider
    
    tsl_args = get_tsl_args(args.dataset, root_path)
    
    # Create model FIRST (consumes RNG)
    tsl_model = TSLModel(tsl_args).float().to(device)
    
    # Create dataloaders AFTER model
    _, train_loader_tsl = data_provider(tsl_args, 'train')
    _, val_loader_tsl = data_provider(tsl_args, 'val')
    _, test_loader_tsl = data_provider(tsl_args, 'test')
    
    tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    label_len = 48
    pred_len = 96
    
    tsl_train_losses = []
    tsl_val_losses = []
    
    print("[TSL] Training...")
    for epoch in range(args.epochs):
        batch_losses = train_one_epoch(
            tsl_model, train_loader_tsl, tsl_optimizer, criterion,
            device, label_len, pred_len
        )
        train_loss = np.mean(batch_losses)
        val_loss = evaluate(tsl_model, val_loader_tsl, criterion, device, label_len, pred_len)
        
        tsl_train_losses.append(train_loss)
        tsl_val_losses.append(val_loss)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.10f}, val_loss={val_loss:.10f}")
    
    # Test evaluation
    tsl_test_loss = evaluate(tsl_model, test_loader_tsl, criterion, device, label_len, pred_len)
    print(f"[TSL] Test MSE: {tsl_test_loss:.10f}")
    
    # =========================================================================
    # LIULIAN TRAINING (using TSL's data)
    # =========================================================================
    print("\n[Liulian] Setting up...")
    set_deterministic(args.seed)
    
    from liulian.models.torch.transformer import TransformerAdapter
    
    model_config = {
        'seq_len': 96,
        'pred_len': 96,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'factor': 3,
        'dropout': 0.0,  # ZERO dropout
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': 'h',
        'task_name': 'long_term_forecast',
    }
    
    # Create model FIRST
    adapter = TransformerAdapter(model_config)
    liu_model = adapter._model.float().to(device)
    
    # Re-create TSL dataloaders (SAME data, SAME order)
    set_deterministic(args.seed)
    # Need to recreate with same RNG state after model
    _, train_loader_liu = data_provider(tsl_args, 'train')
    _, val_loader_liu = data_provider(tsl_args, 'val')
    _, test_loader_liu = data_provider(tsl_args, 'test')
    
    liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
    
    liu_train_losses = []
    liu_val_losses = []
    
    print("[Liulian] Training...")
    for epoch in range(args.epochs):
        batch_losses = train_one_epoch(
            liu_model, train_loader_liu, liu_optimizer, criterion,
            device, label_len, pred_len
        )
        train_loss = np.mean(batch_losses)
        val_loss = evaluate(liu_model, val_loader_liu, criterion, device, label_len, pred_len)
        
        liu_train_losses.append(train_loss)
        liu_val_losses.append(val_loss)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.10f}, val_loss={val_loss:.10f}")
    
    liu_test_loss = evaluate(liu_model, test_loader_liu, criterion, device, label_len, pred_len)
    print(f"[Liulian] Test MSE: {liu_test_loss:.10f}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print("\nPer-Epoch Train Loss:")
    print(f"{'Epoch':<8} {'TSL':<20} {'Liulian':<20} {'Diff':<15} {'Match'}")
    all_match = True
    for i in range(args.epochs):
        diff = abs(tsl_train_losses[i] - liu_train_losses[i])
        match = diff < 1e-9
        all_match = all_match and match
        print(f"{i+1:<8} {tsl_train_losses[i]:<20.10f} {liu_train_losses[i]:<20.10f} {diff:<15.2e} {'✓' if match else '✗'}")
    
    print("\nPer-Epoch Val Loss:")
    print(f"{'Epoch':<8} {'TSL':<20} {'Liulian':<20} {'Diff':<15} {'Match'}")
    for i in range(args.epochs):
        diff = abs(tsl_val_losses[i] - liu_val_losses[i])
        match = diff < 1e-9
        all_match = all_match and match
        print(f"{i+1:<8} {tsl_val_losses[i]:<20.10f} {liu_val_losses[i]:<20.10f} {diff:<15.2e} {'✓' if match else '✗'}")
    
    print(f"\nTest MSE:")
    test_diff = abs(tsl_test_loss - liu_test_loss)
    test_match = test_diff < 1e-9
    all_match = all_match and test_match
    print(f"TSL:     {tsl_test_loss:.10f}")
    print(f"Liulian: {liu_test_loss:.10f}")
    print(f"Diff:    {test_diff:.2e} {'✓' if test_match else '✗'}")
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ ALL RESULTS IDENTICAL - Implementations verified!")
    else:
        print("❌ DIFFERENCES DETECTED - Investigation needed")
    print("=" * 60)
    
    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())
