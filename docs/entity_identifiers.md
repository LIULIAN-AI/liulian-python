# Entity Identifiers

Entity identifiers allow models to distinguish between different entities (sensors, stations, clients) in multi-entity time series datasets.

## Overview

Many real-world time-series datasets contain multiple independent entities measuring the same quantity:
- **Traffic**: 862 road occupancy sensors
- **Electricity**: 321 household meters
- **PEMS**: 170–883 traffic flow sensors
- **Swiss-river**: river monitoring stations with coordinates

Entity identifiers inject per-entity information into the model, enabling it to learn entity-specific biases, spatial relationships, and distributional differences.

## When to Use Entity Identifiers

### Beneficial (homogeneous entities)
Entity identifiers help when features **are** entities — interchangeable channels measuring the same physical quantity:

| Dataset | Entities? | Why |
|---------|:---------:|-----|
| Traffic | ✅ | 862 sensors measuring road occupancy % |
| Electricity | ✅ | 321 clients measuring kWh consumption |
| PEMS03/04/07/08 | ✅ | Traffic flow sensors with adjacency graph |
| Swiss-river | ✅ | Stations with coordinates and river topology |
| Exchange Rate | ⚠️ | 8 countries — borderline (few, semi-heterogeneous) |

### Not Beneficial (heterogeneous features)
Entity identifiers **do not** help when features are heterogeneous — they measure different physical quantities and interact semantically:

| Dataset | Entities? | Why |
|---------|:---------:|-----|
| ETT (h1/h2/m1/m2) | ❌ | Columns are load (MW), temperature (°C) — different units |
| Weather | ❌ | Pressure (mbar), temperature (°C), humidity (%) |
| ILI | ❌ | Weighted ILI %, age groups, totals |
| M4 | ❌ | Univariate — no entity concept |

## Supported Modes

| Mode | Key | Description |
|------|-----|-------------|
| None | `'none'` | No entity features (default) |
| Embedding | `'embedding'` | Learned `nn.Embedding` per entity; projected back to original dimension |
| One-hot | `'onehot'` | Binary one-hot vector concatenated to input features |
| Numeric ID | `'numeric_id'` | Raw integer ID as an additional feature |
| Sinusoidal | `'sinusoidal'` | Positional-encoding-style features from entity ID |
| Coordinates | `'coordinates'` | Geographic lat/lon (Swiss-river only) |
| Descriptors | `'descriptors'` | Custom descriptor vector per entity |

## Usage

### With TSL Models (via EntityAwareMixin)

All 11 Time-Series-Library model adapters support entity identifiers through the `EntityAwareMixin`:

```python
from liulian.models.torch import DLinearAdapter

config = {
    'enc_in': 862,
    'seq_len': 96,
    'pred_len': 96,
    'identifier_mode': 'embedding',   # Enable entity embedding
    'num_embeddings': 862,            # Number of entities
    'embedding_size': 16,             # Embedding dimension
    'entity_id_col': 0,              # Column index in x_mark for entity IDs
    # ... other model config
}

adapter = DLinearAdapter(config)
# The model automatically wraps in EntityWrapper for 'embedding' mode
```

For transparent modes (`'onehot'`, `'numeric_id'`, `'sinusoidal'`, `'coordinates'`), the entity features are concatenated to `x_enc` by the data layer. Set `enc_in` to include the extra dimensions:

```python
config = {
    'enc_in': 862 + 862,  # original features + one-hot entity features
    'identifier_mode': 'onehot',
    # ...
}
```

### With Custom Models (LSTMAdapter, TransformerEncoderAdapter)

These models have built-in entity support with all 7 modes:

```python
from liulian.models.torch import LSTMAdapter

config = {
    'enc_in': 2,       # features per station (water_temp, air_temp)
    'c_out': 1,        # predict water temperature
    'seq_len': 30,
    'pred_len': 20,
    'identifier_mode': 'embedding',
    'num_embeddings': 50,
    'embedding_size': 10,
    # ...
}
adapter = LSTMAdapter(config)
```

## How It Works

### Embedding Mode (Recommended)

The `EntityWrapper` wraps the inner model and:
1. Extracts entity IDs from `x_mark_enc[:, :, entity_id_col]`
2. Looks up embeddings via `nn.Embedding(num_embeddings, embedding_size)`
3. Concatenates embeddings to `x_enc` → shape `(B, T, enc_in + emb_size)`
4. Projects back to `enc_in` via a learned `nn.Linear` layer
5. The inner model sees the original dimensions — no architecture changes needed

For encoder-decoder models, the decoder input `x_dec` is also augmented.

### Transparent Modes

For `'onehot'`, `'numeric_id'`, `'sinusoidal'`, and `'coordinates'`, the data layer (`TimeSeriesDataset.make_entity_features()`) appends entity features directly to the input tensor. The model's `enc_in` parameter must account for the additional dimensions.

## Benchmark Experiments

The entity identifier ablation (Experiment E.2.2) tests all 15 models × 10 entity datasets × 5 modes. Run:

```bash
# Generate configs
python tools/generate_configs.py --group entity

# Run experiments
python tools/run_benchmark.py --config-dir experiments/configs/entity/ --seeds 1,2,3

# Aggregate results
python tools/aggregate_results.py --results-dir experiments/results/
```
