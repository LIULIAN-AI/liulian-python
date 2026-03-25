import torch
import random
import numpy as np
from argparse import Namespace
import sys

from liulian.models.torch.etsformer import Model as LL_ETSformer

sys.path.append('refer_projects/Time-Series-Library')
from models.ETSformer import Model as TSL_ETSformer

def set_seed():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

config = Namespace(
    task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=96,
    e_layers=2, d_layers=2, d_model=512, enc_in=7, c_out=7, d_ff=2048,
    dropout=0.1, activation='gelu', top_k=5, n_heads=8, embed='timeF', freq='h', num_class=3
)

set_seed()
m_ll = LL_ETSformer(config)

set_seed()
m_tsl = TSL_ETSformer(config)

diff = 0.0
for (n1, p1), (n2, p2) in zip(m_ll.named_parameters(), m_tsl.named_parameters()):
    diff += (p1 - p2).abs().sum().item()

print(f"Total parameter difference: {diff}")
