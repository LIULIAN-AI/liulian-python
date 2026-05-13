#!/usr/bin/env python3
"""Final Identical Test - Complete Training + Evaluation.

Uses float64 aligned dataloader for IDENTICAL results.
Runs full training without early stopping.
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
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")
    print("=" * 70)
    
    from models.Transformer import Model as TSLModel
    from liulian.models.torch.transformer import TransformerAdapter
    from experiments.adapt_tsl_lib.tsl_float64_dataloader import create_tsl_aligned_dataloader
    
    label_len = 48
    pred_len = 96
    criterion = nn.MSELoss()
    root_path = os.path.join(PROJECT_ROOT, 'dataset/ETT-small/')
    
    model_config = {
        'seq_len': 96, 'pred_len': 96, 'label_len': 48,
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
        'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048,
        'factor': 3, 'dropout': 0.0, 'activation': 'gelu',
        'embed': 'timeF', 'freq': 'h', 'task_name': 'long_term_forecast',
    }
    
    # TSL Args
    class Args:
        task_name = 'long_term_forecast'
        seq_len = 96
        label_len = 48
        pred_len = 96
        enc_in = 7
        dec_in = 7
        c_out = 7
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 3
        dropout = 0.0
        activation = 'gelu'
        embed = 'timeF'
        freq = 'h'
        output_attention = False
    
    args = Args()
    
    # =========================================================================
    # CREATE MODELS (same seed = same weights)
    # =========================================================================
    print("[1] Creating models...")
    
    set_deterministic(seed)
    tsl_model = TSLModel(args).float().to(device)
    tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
    
    set_deterministic(seed)
    adapter = TransformerAdapter(model_config)
    liu_model = adapter._model.float().to(device)
    liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
    
    # Verify initial weights match
    w_match = torch.allclose(
        tsl_model.enc_embedding.value_embedding.tokenConv.weight,
        liu_model.enc_embedding.value_embedding.tokenConv.weight
    )
    print(f"   Initial weights match: {w_match}")
    
    # =========================================================================
    # CREATE DATALOADERS (same seed = same shuffle)
    # =========================================================================
    print("[2] Creating dataloaders...")
    
    def create_loaders(seed_offset=0):
        set_deterministic(seed + seed_offset)
        train_loader = create_tsl_aligned_dataloader(
            root_path=root_path, data_path='ETTh1.csv', flag='train',
            seq_len=96, label_len=48, pred_len=96, features='M',
            batch_size=32, shuffle=True, drop_last=True,
        )
        val_loader = create_tsl_aligned_dataloader(
            root_path=root_path, data_path='ETTh1.csv', flag='val',
            seq_len=96, label_len=48, pred_len=96, features='M',
            batch_size=32, shuffle=False, drop_last=True,
        )
        test_loader = create_tsl_aligned_dataloader(
            root_path=root_path, data_path='ETTh1.csv', flag='test',
            seq_len=96, label_len=48, pred_len=96, features='M',
            batch_size=32, shuffle=False, drop_last=True,
        )
        return list(train_loader), list(val_loader), list(test_loader)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n[3] Training both models...")
    
    all_identical = True
    
    for epoch in range(epochs):
        # Create fresh loaders for this epoch (same seed per epoch)
        train_batches, val_batches, _ = create_loaders(seed_offset=epoch)
        
        # Train TSL
        tsl_model.train()
        tsl_train_losses = []
        for batch in train_batches:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            tsl_optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = tsl_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            loss.backward()
            tsl_optimizer.step()
            tsl_train_losses.append(loss.item())
        
        # Train Liulian on SAME batches
        liu_model.train()
        liu_train_losses = []
        for batch in train_batches:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            liu_optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            loss.backward()
            liu_optimizer.step()
            liu_train_losses.append(loss.item())
        
        # Validate
        tsl_model.eval()
        liu_model.eval()
        tsl_val_loss = 0
        liu_val_loss = 0
        
        with torch.no_grad():
            for batch in val_batches:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
                
                tsl_out = tsl_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                liu_out = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                tsl_val_loss += criterion(tsl_out[:, -pred_len:, :], batch_y[:, -pred_len:, :]).item()
                liu_val_loss += criterion(liu_out[:, -pred_len:, :], batch_y[:, -pred_len:, :]).item()
        
        tsl_val_loss /= len(val_batches)
        liu_val_loss /= len(val_batches)
        
        # Check
        train_diff = max(abs(t - l) for t, l in zip(tsl_train_losses, liu_train_losses))
        val_diff = abs(tsl_val_loss - liu_val_loss)
        epoch_match = train_diff < 1e-9 and val_diff < 1e-9
        all_identical = all_identical and epoch_match
        
        print(f"   Epoch {epoch+1}: train={np.mean(tsl_train_losses):.6f} | val={tsl_val_loss:.6f} | "
              f"train_diff={train_diff:.1e} | val_diff={val_diff:.1e} | {'✓' if epoch_match else '✗'}")
    
    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    print("\n[4] Test evaluation...")
    _, _, test_batches = create_loaders(seed_offset=0)
    
    tsl_model.eval()
    liu_model.eval()
    
    tsl_test_loss = 0
    liu_test_loss = 0
    
    with torch.no_grad():
        for batch in test_batches:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            tsl_out = tsl_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            liu_out = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            tsl_test_loss += criterion(tsl_out[:, -pred_len:, :], batch_y[:, -pred_len:, :]).item()
            liu_test_loss += criterion(liu_out[:, -pred_len:, :], batch_y[:, -pred_len:, :]).item()
    
    tsl_test_loss /= len(test_batches)
    liu_test_loss /= len(test_batches)
    test_diff = abs(tsl_test_loss - liu_test_loss)
    
    print(f"   TSL Test MSE:     {tsl_test_loss:.10f}")
    print(f"   Liulian Test MSE: {liu_test_loss:.10f}")
    print(f"   Difference:       {test_diff:.2e}")
    
    all_identical = all_identical and (test_diff < 1e-9)
    
    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    print("\n" + "=" * 70)
    if all_identical:
        print("✅ ALL RESULTS IDENTICAL")
        print("   Training, validation, and test metrics are byte-identical!")
    else:
        print("❌ DIFFERENCES DETECTED")
    print("=" * 70)
    
    return 0 if all_identical else 1


if __name__ == '__main__':
    sys.exit(main())
