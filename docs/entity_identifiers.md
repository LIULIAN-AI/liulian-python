# Entity Identifiers

Entity identifiers allow models to distinguish between different entities (sensors, stations, clients) in multi-entity time series datasets.

## Overview

Many real-world time-series datasets contain multiple independent entities measuring the same quantity:
- **Traffic**: 862 road occupancy sensors
- **Electricity**: 321 household meters
- **PEMS**: 170–883 traffic flow sensors
- **Solar-Energy**: 137 PV plant power output sensors
- **Swiss-river**: river monitoring stations with coordinates

Entity identifiers inject per-entity information into the model, enabling it to learn entity-specific biases, spatial relationships, and distributional differences.

## When to Use Entity Identifiers

### Beneficial (homogeneous entities)

Entity identifiers help when features **are** entities — interchangeable channels measuring the same physical quantity.

#### TSLib / Standard Forecasting Benchmarks

These datasets are from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib) and are used by most long-term forecasting papers including PatchTST, iTransformer, TimesNet, DLinear, Autoformer, FEDformer, Informer, etc.

| Dataset | #Entities | Freq | Why beneficial | Source & References |
|---------|----------:|------|----------------|---------------------|
| **Traffic** | 862 | Hourly | Road occupancy % sensors on San Francisco Bay area freeways; all channels measure the same quantity | [Caltrans PEMS](http://pems.dot.ca.gov), introduced by [LSTNet (Lai et al., SIGIR 2018)](https://arxiv.org/abs/1703.07015). Used by Autoformer, FEDformer, PatchTST, iTransformer, TimesNet, DLinear, TimeMixer, etc. |
| **Electricity** (ECL) | 321 | Hourly | Household electricity consumption in kWh; all channels are interchangeable clients | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), introduced by [LSTNet (Lai et al., SIGIR 2018)](https://arxiv.org/abs/1703.07015). Used by Autoformer, Informer, PatchTST, iTransformer, Time-LLM, OneFitsAll, CALF, etc. |
| **Solar-Energy** | 137 | 10-min | Solar power output from 137 PV plants in Alabama; all channels measure the same quantity (kW) | [NREL Solar Data](http://www.nrel.gov/grid/solar-power-data.html), introduced by [LSTNet (Lai et al., SIGIR 2018)](https://arxiv.org/abs/1703.07015). Used by AutoTimes, iTransformer, Crossformer, etc. |
| **Exchange Rate** | 8 | Daily | Daily exchange rates of 8 countries; borderline — few entities, semi-heterogeneous | [Lai et al., SIGIR 2018](https://arxiv.org/abs/1703.07015). Used by Autoformer, Informer, PatchTST, Time-LLM, OneFitsAll, CALF, etc. |

#### PEMS Traffic Sensor Datasets

From the California DOT Performance Measurement System.  Used extensively in spatial-temporal GNN papers.

| Dataset | #Sensors | Features | Freq | Why beneficial | Source & References |
|---------|----------|----------|------|----------------|---------------------|
| **PEMS03** | 358 | Flow | 5-min | Traffic flow sensors with adjacency graph | [Song et al., STSGCN (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |
| **PEMS04** | 307 | Flow, Occupy, Speed | 5-min | Traffic sensors with 3 features each; channels are interchangeable sensors | [Song et al., STSGCN (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |
| **PEMS07** | 883 | Flow | 5-min | Large-scale traffic flow network | [Song et al., STSGCN (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |
| **PEMS08** | 170 | Flow, Occupy, Speed | 5-min | Smaller traffic sensor network | [Song et al., STSGCN (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |

#### Spatial-Temporal Graph Forecasting Datasets

These are from the GNN / spatial-temporal forecasting literature (DCRNN, STGCN, Graph WaveNet, AGCRN, MTGNN, etc.):

| Dataset | #Entities | Freq | Why beneficial | Source & References |
|---------|----------:|------|----------------|---------------------|
| **METR-LA** | 207 | 5-min | Loop detector speed sensors in Los Angeles highways; all channels measure traffic speed (mph) | [Li et al., DCRNN (ICLR 2018)](https://arxiv.org/abs/1707.01926). Used by STGCN, Graph WaveNet, AGCRN, MTGNN, StemGNN, etc. |
| **PEMS-BAY** | 325 | 5-min | Traffic speed sensors in San Francisco Bay Area; same quantity across all sensors | [Li et al., DCRNN (ICLR 2018)](https://arxiv.org/abs/1707.01926). Used by STGCN, Graph WaveNet, AGCRN, MTGNN, etc. |
| **NYC Taxi** | 75 zones | 30-min | Taxi trip demand per zone; homogeneous count data across pickup zones | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Used by [STResNet (Zhang et al., AAAI 2017)](https://arxiv.org/abs/1610.00081), AGCRN, etc. |
| **NYC Bike** | 250+ stations | Hourly | Bike-sharing trip counts per station; homogeneous demand count | [Citi Bike](https://citibikenyc.com/system-data). Used by STGCN, STResNet, AGCRN, etc. |
| **Beijing Air Quality** | 12 stations | Hourly | PM2.5 concentration across monitoring stations; same pollutant measurement | [UCI AQ Dataset](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data+set). Used by [Yi et al. (KDD 2016)](https://dl.acm.org/doi/10.1145/2939672.2939679), MTGNN, etc. |

#### Domain-Specific Datasets

| Dataset | #Entities | Freq | Why beneficial | Source & References |
|---------|----------:|------|----------------|---------------------|
| **Swiss-river** | 10–50 stations | Daily | River monitoring stations with coordinates and graph topology; per-station water temperature | [Swiss FOEN](https://www.bafu.admin.ch/bafu/en/home.html). Used in this project. |
| **NN5** | 111 ATMs | Daily | Cash withdrawal time series from 111 ATMs in England; homogeneous demand data | [NN5 Competition (Crone, 2008)](http://www.neural-forecasting-competition.com/NN5/). Used by [Tan et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.16964). |
| **London Smart Meters** | 5,567 meters | 30-min | Household energy consumption; all channels are interchangeable meters measuring kWh | [UK Power Networks / Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london). |
| **Kaggle Store Sales** | 54 stores×33 families | Daily | Store-level product sales; same metric (units sold) across stores | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). |
| **Kaggle Web Traffic** | 145K pages | Daily | Daily page views for Wikipedia articles; each channel is one page | [Kaggle](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting). |
| **FRED-MD** | ~130 series | Monthly | Macroeconomic indicators; each channel is one economic variable — borderline heterogeneous | [McCracken & Ng (2016)](https://doi.org/10.20955/r.2016.31-56). Used by [Tan et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.16964). |

> **Note on the "Are Language Models Actually Useful for Time Series Forecasting?" paper**
> ([Tan et al., NeurIPS 2024 Spotlight](https://arxiv.org/abs/2406.16964),
> [code](https://github.com/BennyTMT/LLMsForTimeSeries),
> [OpenReview](https://openreview.net/forum?id=DV15UbHCY1)):
>
> This paper benchmarks on 8 core datasets: **ETTh1, ETTh2, ETTm1, ETTm2,
> Weather, Electricity, Traffic, ILI**, plus 5 extended datasets from the
> rebuttal: **Exchange Rate, Covid Deaths, NYC Taxi, NN5, FRED-MD**.
> Of these, **Electricity** (321 clients), **Traffic** (862 sensors), and
> **NN5** (111 ATMs) are homogeneous-entity datasets that benefit from entity
> identifiers.  The rest (ETT, Weather, ILI, Covid Deaths) have heterogeneous
> features and do not benefit.

> **Note on AutoTimes** ([Liu et al., NeurIPS 2024](https://arxiv.org/abs/2402.02370),
> [code](https://github.com/thuml/AutoTimes)):
>
> AutoTimes benchmarks on **ECL** (321 clients), **ETTh1** (7 features),
> **Solar-Energy** (137 PV plants), **Traffic** (862 sensors), and
> **Weather** (21 features).  Entity identifiers are beneficial for
> **ECL**, **Solar-Energy**, and **Traffic** (homogeneous entities);
> they are not beneficial for **ETTh1** and **Weather** (heterogeneous features).

### Not Beneficial (heterogeneous features)

Entity identifiers **do not** help when features are heterogeneous — they measure different physical quantities and interact semantically:

| Dataset | Entities? | #Channels | Why not beneficial | Source & References |
|---------|:---------:|----------:|--------------------|----|
| **ETT** (h1/h2/m1/m2) | ❌ | 7 | Columns are load (MW) and temperature (°C) from one transformer — different physical units | [Zhou et al., Informer (AAAI 2021)](https://arxiv.org/abs/2012.07436). Used by virtually all TSF papers. |
| **Weather** | ❌ | 21 | Pressure (mbar), temperature (°C), humidity (%) — semantically distinct meteorological variables | [Wetterstation (Max Planck Biogeochemistry)](https://www.bgc-jena.mpg.de/wetter/). Used by Autoformer, PatchTST, iTransformer, etc. |
| **ILI** (Illness) | ❌ | 7 | Weighted ILI %, unweighted %, age group breakdowns — different statistical measures | [CDC ILINet](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html). Used by Autoformer, TimesNet, Time-LLM, CALF, etc. |
| **M4** | ❌ | 1 each | Univariate competition dataset — 100K individual series with no multi-entity structure | [M4 Competition (Makridakis et al., 2020)](https://doi.org/10.1016/j.ijforecast.2019.04.014). |
| **Covid Deaths** | ⚠️ | varies | Regional death counts; could be treated as entities but series are short and policy-driven | Used by [Tan et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.16964). |

## Supported Modes

| Mode | Key | Description |
|------|-----|-------------|
| None | `'none'` | No entity features (default) |
| Embedding | `'embedding'` | Learned `nn.Embedding` per entity; projected back to original dimension |
| One-hot | `'onehot'` | Binary one-hot vector concatenated to input features |
| Numeric ID | `'numeric_id'` | Raw integer ID as an additional feature |
| Sinusoidal | `'sinusoidal'` | Positional-encoding-style features from entity ID |
| Random | `'random'` | Deterministic random vector baseline per entity |
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

For transparent modes (`'onehot'`, `'numeric_id'`, `'sinusoidal'`, `'random'`, `'coordinates'`), the entity features are concatenated to `x_enc` by the data layer. Set `enc_in` to include the extra dimensions:

```python
config = {
    'enc_in': 862 + 862,  # original features + one-hot entity features
    'identifier_mode': 'onehot',
    # ...
}
```

### With Custom Models (LSTMAdapter, TransformerEncoderAdapter)

These models have built-in entity support with all listed modes:

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

For `'onehot'`, `'numeric_id'`, `'sinusoidal'`, `'random'`, and `'coordinates'`, the data layer (`TimeSeriesDataset.make_entity_features()`) appends entity features directly to the input tensor. The model's `enc_in` parameter must account for the additional dimensions.

> **Current caveat (important):** In this repository's default matrix setup, transparent modes are currently applied only to `swiss-river-1990 + lstm` (per-entity split). For `traffic`, `electricity`, and most other multi-channel CSV/PEMS setups, transparent modes are not enabled in the matrix runner because those paths do not currently construct per-entity transparent features.

### Why include sinusoidal and random baselines?

- **Sinusoidal** is a non-learnable structured baseline: it injects a smooth, index-aware entity code without extra trainable parameters.
- **Random** is a non-semantic control baseline: if random IDs perform close to learned embeddings, gains may come from extra capacity only; if embeddings clearly win, gains are likely entity-structure-aware.

## Matrix Experiment Integration (`experiments/entity_identifier`)

The matrix runner (`experiments/entity_identifier/run.py`) now supports:

- Global modes: `none`, `embedding`
- Additional competitor modes (applicable subset): `onehot`, `coordinates`, `sinusoidal`, `random`

By default, the runner auto-filters inapplicable dataset-model-mode combinations.
For random/sinusoidal transparent baselines, you can tune:

- `random_identifier_dim` (default `16`)
- `random_identifier_seed` (default `2026`)
- `sinusoidal_dim` (default `16`)

Examples:

```bash
# Local dry run with extra identifier competitors
python experiments/entity_identifier/run.py \
  --phase dry \
  --modes none embedding onehot coordinates sinusoidal random

# Slurm: recommended one job per experiment (better GPU parallelism + fault isolation)
python experiments/entity_identifier/submit_slurm.py \
  --dispatch-mode per-experiment \
  --run-tag eid_competitors \
  --modes none embedding onehot coordinates sinusoidal random
```

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
