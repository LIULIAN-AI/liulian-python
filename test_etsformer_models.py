import torch
from argparse import Namespace
import sys

# load Liulian Model
from liulian.models.torch.etsformer import Model as LL_ETSformer

# load TSL Model
sys.path.append('refer_projects/Time-Series-Library')
from models.ETSformer import Model as TSL_ETSformer

config = Namespace(
    task_name='long_term_forecast',
    seq_len=96,
    label_len=48,
    pred_len=96,
    e_layers=2,
    d_layers=2,
    d_model=512,
    enc_in=7,
    c_out=7,
    d_ff=2048,
    dropout=0.1,
    activation='gelu',
    top_k=5,
    n_heads=8,
    embed='timeF',
    freq='h'
)

m_ll = LL_ETSformer(config)
m_tsl = TSL_ETSformer(config)

m_tsl.load_state_dict(m_ll.state_dict())

m_ll.eval()
m_tsl.eval()

x_enc = torch.randn(32, 96, 7)
x_mark_enc = torch.randn(32, 96, 4)
x_dec = torch.randn(32, 96 + 48, 7)
x_mark_dec = torch.randn(32, 96 + 48, 4)

with torch.no_grad():
    y_ll = m_ll(x_enc, x_mark_enc, x_dec, x_mark_dec)
    y_tsl = m_tsl(x_enc, x_mark_enc, x_dec, x_mark_dec)

print(f"Max diff between LL and TSL: {(y_ll - y_tsl).abs().max().item()}")

