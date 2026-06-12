#!/usr/bin/env python3
"""Compare TSL vs Liulian with IDENTICAL results mode.

This script runs both TSL and Liulian Transformer models with all
randomness eliminated to achieve byte-identical results.

Requirements for identical results:
1. Dropout = 0.0 (no stochasticity)
2. Deterministic mode enabled
3. Float64 data (matching TSL's internal dtype)
4. Same batch objects shared between models
5. No early stopping (fixed epochs)
6. Same seed order: seed -> model -> dataloader

Usage:
    python compare_identical.py --pairs ETTh1_Transformer --epochs 5
"""

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'refer_projects' / 'Time-Series-Library'))


def set_deterministic(seed: int = 2021):
    """Enable full deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# Dataset configurations
DATASET_CONFIGS = {
    'ETTh1': {
        'root_path': 'dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'enc_in': 7,
        'freq': 'h',
    },
    'ETTh2': {
        'root_path': 'dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'enc_in': 7,
        'freq': 'h',
    },
    'ETTm1': {
        'root_path': 'dataset/ETT-small/',
        'data_path': 'ETTm1.csv',
        'enc_in': 7,
        'freq': 't',
    },
    'ETTm2': {
        'root_path': 'dataset/ETT-small/',
        'data_path': 'ETTm2.csv',
        'enc_in': 7,
        'freq': 't',
    },
    'Weather': {
        'root_path': 'dataset/weather/',
        'data_path': 'weather.csv',
        'enc_in': 21,
        'freq': 'h',
    },
}


def create_tsl_model(args, device):
    """Create TSL Transformer model."""
    from models.Transformer import Model as TSLModel
    model = TSLModel(args).float().to(device)
    return model


def create_liulian_model(config, device):
    """Create Liulian Transformer model."""
    from liulian.models.torch.transformer import TransformerAdapter
    adapter = TransformerAdapter(config)
    model = adapter._model.float().to(device)
    return model


def create_dataloader(dataset_name: str, flag: str, batch_size: int = 32, shuffle: bool = True):
    """Create float64-aligned dataloader matching TSL."""
    from experiments.adapt_tsl_lib.tsl_float64_dataloader import create_tsl_aligned_dataloader
    
    cfg = DATASET_CONFIGS[dataset_name]
    root_path = str(PROJECT_ROOT / cfg['root_path'])
    
    loader = create_tsl_aligned_dataloader(
        root_path=root_path,
        data_path=cfg['data_path'],
        flag=flag,
        seq_len=96,
        label_len=48,
        pred_len=96,
        features='M',
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        freq=cfg['freq'],  # Use correct freq from config
    )
    return loader


def train_epoch(model, batches, optimizer, criterion, device, label_len=48, pred_len=96):
    """Train one epoch on given batches."""
    model.train()
    losses = []
    
    for batch in batches:
        batch_x, batch_y, batch_x_mark, batch_y_mark = [b.float().to(device) for b in batch]
        
        optimizer.zero_grad()
        dec_inp = torch.cat([
            batch_y[:, :label_len, :],
            torch.zeros_like(batch_y[:, -pred_len:, :])
        ], dim=1).to(device)
        
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def evaluate(model, batches, criterion, device, label_len=48, pred_len=96):
    """Evaluate on given batches."""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in batches:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [b.float().to(device) for b in batch]
            
            dec_inp = torch.cat([
                batch_y[:, :label_len, :],
                torch.zeros_like(batch_y[:, -pred_len:, :])
            ], dim=1).to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    
    return total_loss / count


def run_comparison(dataset_name: str, epochs: int = 5, seed: int = 2021):
    """Run identical comparison for a dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = DATASET_CONFIGS[dataset_name]
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")
    print(f"{'='*70}")
    
    # TSL Args (dropout=0.0 for determinism)
    class TSLArgs:
        task_name = 'long_term_forecast'
        seq_len = 96
        label_len = 48
        pred_len = 96
        enc_in = cfg['enc_in']
        dec_in = cfg['enc_in']
        c_out = cfg['enc_in']
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 3
        dropout = 0.0  # CRITICAL: No dropout
        activation = 'gelu'
        embed = 'timeF'
        freq = cfg['freq']
        output_attention = False
    
    tsl_args = TSLArgs()
    
    # Liulian config (dropout=0.0 for determinism)
    liulian_config = {
        'seq_len': 96,
        'pred_len': 96,
        'label_len': 48,
        'enc_in': cfg['enc_in'],
        'dec_in': cfg['enc_in'],
        'c_out': cfg['enc_in'],
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'factor': 3,
        'dropout': 0.0,  # CRITICAL: No dropout
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': cfg['freq'],
        'task_name': 'long_term_forecast',
    }
    
    criterion = nn.MSELoss()
    
    # Create models (same seed = same weights)
    print("\n[1] Creating models...")
    set_deterministic(seed)
    tsl_model = create_tsl_model(tsl_args, device)
    tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
    
    set_deterministic(seed)
    liu_model = create_liulian_model(liulian_config, device)
    liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
    
    # Verify initial weights match
    tsl_w = tsl_model.enc_embedding.value_embedding.tokenConv.weight
    liu_w = liu_model.enc_embedding.value_embedding.tokenConv.weight
    w_match = torch.allclose(tsl_w, liu_w)
    print(f"   Initial weights match: {w_match}")
    
    if not w_match:
        print("   ERROR: Initial weights don't match!")
        return False
    
    # Training
    print("\n[2] Training...")
    all_identical = True
    
    for epoch in range(epochs):
        # Create fresh dataloader with same seed offset
        set_deterministic(seed + epoch)
        train_loader = create_dataloader(dataset_name, 'train', shuffle=True)
        train_batches = list(train_loader)
        
        set_deterministic(seed)  # Reset for validation
        val_loader = create_dataloader(dataset_name, 'val', shuffle=False)
        val_batches = list(val_loader)
        
        # Train both on SAME batches
        tsl_losses = train_epoch(tsl_model, train_batches, tsl_optimizer, criterion, device)
        liu_losses = train_epoch(liu_model, train_batches, liu_optimizer, criterion, device)
        
        # Validate both on SAME batches
        tsl_val = evaluate(tsl_model, val_batches, criterion, device)
        liu_val = evaluate(liu_model, val_batches, criterion, device)
        
        # Check differences
        train_diff = max(abs(t - l) for t, l in zip(tsl_losses, liu_losses))
        val_diff = abs(tsl_val - liu_val)
        epoch_match = train_diff < 1e-9 and val_diff < 1e-9
        all_identical = all_identical and epoch_match
        
        status = '✓' if epoch_match else '✗'
        print(f"   Epoch {epoch+1}: train={np.mean(tsl_losses):.6f} val={tsl_val:.6f} "
              f"| train_diff={train_diff:.1e} val_diff={val_diff:.1e} {status}")
    
    # Test evaluation
    print("\n[3] Test evaluation...")
    set_deterministic(seed)
    test_loader = create_dataloader(dataset_name, 'test', shuffle=False)
    test_batches = list(test_loader)
    
    tsl_test = evaluate(tsl_model, test_batches, criterion, device)
    liu_test = evaluate(liu_model, test_batches, criterion, device)
    test_diff = abs(tsl_test - liu_test)
    
    print(f"   TSL Test MSE:     {tsl_test:.10f}")
    print(f"   Liulian Test MSE: {liu_test:.10f}")
    print(f"   Difference:       {test_diff:.2e}")
    
    all_identical = all_identical and (test_diff < 1e-9)
    
    # Final result
    print(f"\n{'='*70}")
    if all_identical:
        print(f"✅ {dataset_name}_Transformer: IDENTICAL")
    else:
        print(f"❌ {dataset_name}_Transformer: DIFFERENT")
    print(f"{'='*70}")
    
    return all_identical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pairs', nargs='+', default=['ETTh1_Transformer'],
        help='Experiment pairs to run (e.g., ETTh1_Transformer ETTh2_Transformer)'
    )
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    args = parser.parse_args()
    
    # Parse dataset names from pair names
    datasets = []
    for pair in args.pairs:
        if '_Transformer' in pair:
            dataset = pair.replace('_Transformer', '')
            if dataset in DATASET_CONFIGS:
                datasets.append(dataset)
            else:
                print(f"Warning: Unknown dataset {dataset}, skipping {pair}")
    
    if not datasets:
        print("No valid Transformer pairs found.")
        print(f"Available: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    # Run comparisons
    results = {}
    for dataset in datasets:
        try:
            identical = run_comparison(dataset, args.epochs, args.seed)
            results[f"{dataset}_Transformer"] = identical
        except Exception as e:
            print(f"Error running {dataset}: {e}")
            results[f"{dataset}_Transformer"] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for pair, identical in results.items():
        status = "✅ IDENTICAL" if identical else "❌ DIFFERENT"
        print(f"  {pair}: {status}")
    
    all_pass = all(results.values())
    print("="*70)
    if all_pass:
        print("✅ ALL PAIRS IDENTICAL")
    else:
        print("❌ SOME PAIRS HAVE DIFFERENCES")
    print("="*70)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
