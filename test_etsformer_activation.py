from liulian.pipeline import get_model, parse_config
from argparse import Namespace
cfg = parse_config(['experiments/etth1/etsformer_config.yaml'])
m = get_model(cfg, None, None, None)
print(m.config.activation)
print(m.model.encoder.layers[0].activation)
