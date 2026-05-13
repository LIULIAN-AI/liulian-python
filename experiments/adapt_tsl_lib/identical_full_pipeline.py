#!/usr/bin/env python3
"""Full Pipeline Identical Comparison.

This script tests that when using:
1. Same seed
2. TSL model created FIRST (consumes RNG)
3. TSL dataloader created AFTER (sees post-model RNG state)
4. Liulian model created FIRST (consumes same RNG)
5. Liulian dataloader created AFTER (sees same post-model RNG state)

Both should produce IDENTICAL results.
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    
    from models.Transformer import Model as TSLModel
    from liulian.models.torch.transformer import TransformerAdapter
    from data_provider.data_factory import data_provider
    from experiments.adapt_tsl_lib.tsl_float64_dataloader import create_tsl_aligned_dataloader
    
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
        dropout = 0.0
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
    # TSL: seed -> model -> dataloader
    # =========================================================================
    print("[TSL] seed -> model -> dataloader")
    set_deterministic(seed)
    tsl_model = TSLModel(args).float().to(device)
    _, tsl_train_loader = data_provider(args, 'train')
    tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
    
    tsl_batch = next(iter(tsl_train_loader))
    print(f"   First batch sum: {tsl_batch[0].sum().item():.10f}")
    
    # =========================================================================
    # LIULIAN: seed -> model -> dataloader (using our float64 loader)
    # =========================================================================
    print("[Liulian] seed -> model -> dataloader")
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
    
    liu_train_loader = create_tsl_aligned_dataloader(
        root_path=os.path.join(PROJECT_ROOT, 'dataset/ETT-small/'),
        data_path='ETTh1.csv',
        flag='train',
        seq_len=96,
        label_len=48,
        pred_len=96,
        features='M',
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
    
    liu_batch = next(iter(liu_train_loader))
    print(f"   First batch sum: {liu_batch[0].sum().item():.10f}")
    
    # =========================================================================
    # COMPARE BATCHES
    # =========================================================================
    print("\n[Comparison]")
    batch_match = torch.allclose(tsl_batch[0], liu_batch[0])
    print(f"   First batch identical: {batch_match}")
    
    if not batch_match:
        print("   ERROR: Batches don't match - RNG state differs!")
        print(f"   TSL[0,0,:3]:     {tsl_batch[0][0,0,:3]}")
        print(f"   Liulian[0,0,:3]: {liu_batch[0][0,0,:3]}")
        return 1
    
    # =========================================================================
    # VERIFY MODEL WEIGHTS
    # =========================================================================
    tsl_w = tsl_model.enc_embedding.value_embedding.tokenConv.weight
    liu_w = liu_model.enc_embedding.value_embedding.tokenConv.weight
    weight_match = torch.allclose(tsl_w, liu_w)
    print(f"   Model weights identical: {weight_match}")
    
    if not weight_match:
        print("   ERROR: Weights don't match!")
        return 1
    
    # =========================================================================
    # TRAIN ONE EPOCH AND COMPARE
    # =========================================================================
    print("\n[Training 1 epoch]")
    
    # Convert full loaders to lists
    set_deterministic(seed)
    _, tsl_train_loader = data_provider(args, 'train')
    tsl_batches = list(tsl_train_loader)
    
    set_deterministic(seed)
    liu_train_loader = create_tsl_aligned_dataloader(
        root_path=os.path.join(PROJECT_ROOT, 'dataset/ETT-small/'),
        data_path='ETTh1.csv',
        flag='train',
        seq_len=96,
        label_len=48,
        pred_len=96,
        features='M',
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    liu_batches = list(liu_train_loader)
    
    print(f"   TSL batches: {len(tsl_batches)}")
    print(f"   Liu batches: {len(liu_batches)}")
    
    # Check all batches match
    all_batches_match = True
    for i, (tb, lb) in enumerate(zip(tsl_batches, liu_batches)):
        if not torch.allclose(tb[0], lb[0]):
            print(f"   Batch {i} MISMATCH!")
            all_batches_match = False
            break
    
    print(f"   All batches identical: {all_batches_match}")
    
    if all_batches_match:
        # Reset models and train
        set_deterministic(seed)
        tsl_model = TSLModel(args).float().to(device)
        tsl_optimizer = optim.Adam(tsl_model.parameters(), lr=0.0001)
        
        set_deterministic(seed)
        adapter = TransformerAdapter(model_config)
        liu_model = adapter._model.float().to(device)
        liu_optimizer = optim.Adam(liu_model.parameters(), lr=0.0001)
        
        # Train both on same batches
        tsl_losses = []
        liu_losses = []
        
        for i, batch in enumerate(tsl_batches):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # TSL
            tsl_model.train()
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
            
            # Liulian (same batch)
            liu_model.train()
            liu_optimizer.zero_grad()
            outputs = liu_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            loss = criterion(outputs, target)
            loss.backward()
            liu_optimizer.step()
            liu_losses.append(loss.item())
        
        # Compare losses
        loss_diffs = [abs(t - l) for t, l in zip(tsl_losses, liu_losses)]
        max_loss_diff = max(loss_diffs)
        
        print(f"\n   TSL mean loss:     {np.mean(tsl_losses):.10f}")
        print(f"   Liulian mean loss: {np.mean(liu_losses):.10f}")
        print(f"   Max batch loss diff: {max_loss_diff:.2e}")
        
        # Compare final weights
        tsl_w_final = tsl_model.enc_embedding.value_embedding.tokenConv.weight
        liu_w_final = liu_model.enc_embedding.value_embedding.tokenConv.weight
        weight_diff = (tsl_w_final - liu_w_final).abs().max().item()
        print(f"   Final weight diff: {weight_diff:.2e}")
        
        print("\n" + "=" * 70)
        if max_loss_diff < 1e-9 and weight_diff < 1e-9:
            print("✅ IDENTICAL - Full pipeline produces identical results!")
        else:
            print("❌ DIFFERENCES DETECTED")
        print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
