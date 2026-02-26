# Datasets

Catalog of all supported datasets with shapes, features, and loading instructions.

## Standard Benchmarks

### ETT (Electricity Transformer Temperature)

| Variant | File | Rows | Channels | Frequency | Target |
|---------|------|------|----------|-----------|--------|
| ETTh1 | `ETTh1.csv` | 17,420 | 7 | Hourly | OT |
| ETTh2 | `ETTh2.csv` | 17,420 | 7 | Hourly | OT |
| ETTm1 | `ETTm1.csv` | 69,680 | 7 | 15-min | OT |
| ETTm2 | `ETTm2.csv` | 69,680 | 7 | 15-min | OT |

Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (heterogeneous: load + temperature).

```python
from liulian.data.data_factory import create_dataloader
loader = create_dataloader(data_name='ETTh1', root_path='dataset/ETT-small',
                           data_path='ETTh1.csv', flag='train',
                           size=(96, 48, 96), batch_size=32)
```

### Weather

| Rows | Channels | Frequency | Entity |
|------|----------|-----------|--------|
| 52,696 | 21 | 10-min | No (heterogeneous) |

Features: pressure (mbar), temperature (°C), humidity (%), wind speed (m/s), radiation (W/m²), etc.

```python
loader = create_dataloader(data_name='weather', root_path='dataset/weather',
                           data_path='weather.csv', flag='train',
                           size=(96, 48, 96), batch_size=32)
```

### Electricity

| Rows | Channels | Frequency | Entity |
|------|----------|-----------|--------|
| 26,304 | 321 | Hourly | Yes (clients) |

321 household electricity meters. Entity identifiers recommended.

```python
loader = create_dataloader(data_name='electricity', root_path='dataset/electricity',
                           data_path='electricity.csv', flag='train',
                           size=(96, 48, 96), batch_size=32)
```

### Traffic

| Rows | Channels | Frequency | Entity |
|------|----------|-----------|--------|
| 17,544 | 862 | Hourly | Yes (sensors) |

862 San Francisco Bay Area road occupancy sensors.

```python
loader = create_dataloader(data_name='traffic', root_path='dataset/traffic',
                           data_path='traffic.csv', flag='train',
                           size=(96, 48, 96), batch_size=32)
```

### Exchange Rate

| Rows | Channels | Frequency | Entity |
|------|----------|-----------|--------|
| ~7,588 | 8 | Daily | Borderline (countries) |

8 national currency exchange rates vs USD.

```python
loader = create_dataloader(data_name='exchange_rate', root_path='dataset/exchange_rate',
                           data_path='exchange_rate.csv', flag='train',
                           size=(96, 48, 96), batch_size=32)
```

### ILI (Influenza-Like Illness)

| Rows | Channels | Frequency | Entity |
|------|----------|-----------|--------|
| ~966 | 7 | Weekly | No (heterogeneous) |

CDC influenza surveillance data. Use shorter horizons (24, 36, 48, 60) due to weekly granularity.

```python
loader = create_dataloader(data_name='illness', root_path='dataset/illness',
                           data_path='national_illness.csv', flag='train',
                           size=(36, 18, 24), batch_size=32)
```

## PEMS Traffic Datasets

| Dataset | Sensors | Features/Sensor | Time Steps | Adjacency |
|---------|---------|-----------------|------------|:---------:|
| PEMS03 | 358 | 1 (flow) | 26,208 | Yes |
| PEMS04 | 307 | 3 (flow, occupancy, speed) | 16,992 | Yes |
| PEMS07 | 883 | 1 (flow) | 28,224 | Yes |
| PEMS08 | 170 | 3 (flow, occupancy, speed) | 17,856 | Yes |

Loaded from `.npz` files. Entity identifiers strongly recommended.

```python
from liulian.data.pems_dataset import PEMSDataset
ds = PEMSDataset(root_path='dataset/PEMS', data_path='PEMS03.npz',
                 flag='train', size=(96, 48, 12))
```

Or via the data factory:
```python
loader = create_dataloader(data_name='PEMS03', root_path='dataset/PEMS',
                           data_path='PEMS03.npz', flag='train',
                           size=(96, 48, 12), batch_size=32)
```

## Swiss-river Datasets

| Variant | Stations | Features/Station | Time Steps | Period | Topology |
|---------|----------|-----------------|------------|--------|:--------:|
| Swiss-1990 | ~64 | 2 (water_temp, air_temp) | ~11,000 | 1990–2020 | Yes |
| Swiss-2010 | ~64 | 2 (water_temp, air_temp) | ~4,000 | 2010–2020 | Yes |
| Swiss-Zurich | ~15 | 2 (water_temp, air_temp) | ~3,000 | Zurich region | Yes |

Swiss river water temperature monitoring network. Full entity identifier support with coordinates and river graph topology.

```python
from liulian.data.ts.timeseriesdataset import TimeSeriesDataset
ds = TimeSeriesDataset.from_csv('dataset/swiss_river/2010.csv', ...)
```

## M4 Competition Datasets

| Frequency | Series Count | Forecast Horizon |
|-----------|-------------|-----------------|
| Yearly | 23,000 | 6 |
| Quarterly | 24,000 | 8 |
| Monthly | 48,000 | 18 |
| Weekly | 359 | 13 |
| Daily | 4,227 | 14 |
| Hourly | 414 | 48 |

Univariate series. Uses SMAPE/MASE/OWA metrics.

```python
loader = create_dataloader(data_name='m4', root_path='dataset/m4',
                           seasonal_patterns='Monthly', flag='train',
                           batch_size=16)
```

## Data Factory Registry

All datasets are registered in `liulian/data/data_factory.py`:

```python
from liulian.data.data_factory import DATASET_REGISTRY
print(list(DATASET_REGISTRY.keys()))
# ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity',
#  'traffic', 'exchange_rate', 'illness', 'solar', 'custom', 'm4',
#  'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
```
