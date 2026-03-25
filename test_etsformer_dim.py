import yaml
from liulian.pipeline import Pipeline
from pprint import pprint

pl = Pipeline()
cfg = pl.get_config('experiments/etth1/etsformer_config.yaml')
print(f"c_out: {cfg.get('c_out')}")
m = pl.get_model(cfg, None, None, None)
print(m)
