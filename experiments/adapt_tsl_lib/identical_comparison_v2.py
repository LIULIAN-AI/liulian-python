#!/usr/bin/env python3
"""Identical Comparison V2 - Share EXACT same data between both models.

Key insight: To get identical results, both models must:
1. Have identical weights (verified - same seed produces same weights)
2. See identical batches in identical order (must use SAME dataloader object)
3. Have zero dropout

Strategy: Create ONE dataloader, run both models on the SAME batches.
"""

import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

warnings.filterwarnings('ignore')

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def main():
    seed = 2021
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    
    # Import models
    from models.Transformer import Model as TSLModel
    from liulian.models.torch.transformer import TransformerAdapter
    from data_provider.data_factory import data_provider
    
    # Args for TSL
    class Args:
        task_name = 'long_term_forecast'
        is_training = 1
        root_path = os.path.join(PROJECT_ROOT, 'dataset/ETT-small/')
        data_path = 'ETTh1.csv'
        data = 'ETTh1'
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
        dropout = 0.0  # ZERO dropout
        embed = 'timeF'
        activation = 'gelu'
        output_attention = False
        batch_size = 32
        num_workers = 0
        augmentation_ratio = 0
        channel_independence = 0
    
    args = Args()
    label_len = 48
    pred_len = 96
    criterion = nn.MSELoss()
    
    # =========================================================================
    # CREATE TSL MODEL
    # =========================================================================
    print("[1] Creating TSL model...")
    set_deterministic(seed)
    tsl_model = TSLModel(args).float().to(device)
    tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
    
    # Save initial weights for verification
    tsl_init_weight = tsl_model.enc_embedding.value_embedding.tokenConv.weight.clone()
    
    # =========================================================================
    # CREATE LIULIAN MODEL (same seed = same weights)
    # =========================================================================
    print("[2] Creating Liulian model...")
    set_deterministic(seed)
    
    model_config = {
        'seq_len': 96, 'pred_len': 96, 'label_len': 48,
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
        'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048,
        'factor': 3, 'dropout': 0.0, 'activation': 'gelu',
        'embed': 'timeF', 'freq': 'h', 'task_name': 'long_term_forecast',
    }
    adapter = TransformerAdapter(model_config)
    liu_model = adapter._model.float().to(device)
    liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
    
    liu_init_weight = liu_model.enc_embedding.value_embedding.tokenConv.weight.clone()
    
    # Verify weights are identical
    weight_match = torch.allclose(tsl_init_weight, liu_init_weight)
    print(f"   Initial weights identical: {weight_match}")
    if not weight_match:
        print("   ERROR: Weights don't match!")
        return 1
    
    # =========================================================================
    # CREATE SINGLE DATALOADER
    # =========================================================================
    print("[3] Creating shared dataloader...")
    set_deterministic(seed)
    _, train_loader = data_provider(args, 'train')
    _, val_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')
    
    # =========================================================================
    # TRAIN BOTH MODELS ON SAME BATCHES
    # =========================================================================
    epochs = 3
    print(f"\n[4] Training both models for {epochs} epochs on SAME batches...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Collect batches for this epoch (to use for both models)
        set_deterministic(seed + epoch)  # Different seed per epoch for shuffle
        _, train_loader = data_provider(args, 'train')
        
        batches = list(train_loader)
        print(f"\nEpoch {epoch+1}: {len(batches)} batches")
        
        # Train TSL
        tsl_model.train()
        tsl_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in batches:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            tsl_optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = tsl_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            target = batch_y[:, -pred_len:, :]
            
            loss = criterion(outputs, target)
            loss.backward()
            tsl_optimizer.step()
            tsl_losses.append(loss.item())
        
        tsl_train_loss = np.mean(tsl_losses)
        
        # Train Liulian on SAME batches
        liu_model.train()
        liu_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in batches:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            liu_optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            target = batch_y[:, -pred_len:, :]
            
            loss = criterion(outputs, target)
            loss.backward()
            liu_optimizer.step()
            liu_losses.append(loss.item())
        
        liu_train_loss = np.mean(liu_losses)
        
        # Compare per-batch losses
        batch_diffs = [abs(t - l) for t, l in zip(tsl_losses, liu_losses)]
        max_batch_diff = max(batch_diffs)
        
        print(f"  TSL train loss:     {tsl_train_loss:.10f}")
        print(f"  Liulian train loss: {liu_train_loss:.10f}")
        print(f"  Max batch diff:     {max_batch_diff:.2e}")
        
        if max_batch_diff > 1e-6:
            print(f"  First 5 batch losses TSL: {tsl_losses[:5]}")
            print(f"  First 5 batch losses Liu: {liu_losses[:5]}")
    
    # =========================================================================
    # COMPARE FINAL WEIGHTS
    # =========================================================================
    print("\n[5] Comparing final weights...")
    tsl_final_weight = tsl_model.enc_embedding.value_embedding.tokenConv.weight
    liu_final_weight = liu_model.enc_embedding.value_embedding.tokenConv.weight
    
    weight_diff = (tsl_final_weight - liu_final_weight).abs().max().item()
    print(f"   Max weight difference: {weight_diff:.2e}")
    
    # =========================================================================
    # TEST ON SAME DATA
    # =========================================================================
    print("\n[6] Evaluating on test set...")
    set_deterministic(seed)
    _, test_loader = data_provider(args, 'test')
    test_batches = list(test_loader)
    
    tsl_model.eval()
    liu_model.eval()
    
    tsl_preds = []
    liu_preds = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_batches:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            tsl_out = tsl_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            liu_out = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            tsl_preds.append(tsl_out[:, -pred_len:, :])
            liu_preds.append(liu_out[:, -pred_len:, :])
    
    tsl_preds = torch.cat(tsl_preds, dim=0)
    liu_preds = torch.cat(liu_preds, dim=0)
    
    pred_diff = (tsl_preds - liu_preds).abs().max().item()
    print(f"   Max prediction difference: {pred_diff:.2e}")
    
    # Final assessment
    print("\n" + "=" * 70)
    if weight_diff < 1e-6 and pred_diff < 1e-6:
        print("✅ IDENTICAL - Both implementations produce identical results!")
    else:
        print("❌ DIFFERENCES DETECTED")
        print(f"   Weight diff: {weight_diff:.2e}")
        print(f"   Pred diff:   {pred_diff:.2e}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
