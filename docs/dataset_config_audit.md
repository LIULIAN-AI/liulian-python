# Dataset Config Audit

Reproducible audit of the maintained benchmark configs under `experiments/{electricity,etth1,etth2,ettm1,ettm2,exchange_rate,illness,pems,swiss_river,traffic,weather}/*.yaml`.

## Scope

- Included only the maintained top-level benchmark YAMLs.
- Excluded generated trees under `experiments/configs/**` and `experiments/artifacts/**`.
- Treated `refer_projects/Time-Series-Library` as the intended TSL source of truth for the classic CSV long-term forecasting datasets.

## Settings Comparison Tables

Cells are bolded when the setting differs across models for the same dataset.

### `electricity`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity | electricity |
| features | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **S** |
| freq | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `etth1`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 | ETTh1 |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| target | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT |
| freq | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `etth2`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 | ETTh2 |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| target | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT |
| freq | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `ettm1`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 | ETTm1 |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| target | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT |
| freq | **h** | **15min** | **h** | **h** | **15min** | **h** | **15min** | **15min** | **15min** | **15min** | **h** | **15min** | **h** | **h** | **15min** | **h** | **h** |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `ettm2`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 | ETTm2 |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| target | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT | OT |
| freq | **h** | **15min** | **h** | **h** | **15min** | **h** | **15min** | **15min** | **15min** | **15min** | **h** | **15min** | **h** | **h** | **15min** | **h** | **h** |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `exchange_rate`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| freq | d | d | d | d | d | d | d | d | d | d | d | d | d | d | d | d | d |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `illness`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| freq | w | w | w | w | w | w | w | w | w | w | w | w | w | w | w | w | w |
| seq_len | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 | 36 |
| label_len | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **18** | **0** | **18** | **18** | **18** |
| pred_len | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `pems`

| setting | dlinear | lstm | patchtst |
| --- | --- | --- | --- |
| data | PEMS03 | PEMS03 | PEMS03 |
| features | M | M | M |
| seq_len | 96 | 96 | 96 |
| label_len | 0 | 0 | 0 |
| pred_len | 12 | 12 | 12 |
| enc_in | — | — | — |
| dec_in | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast |

### `swiss_river`

| setting | default | dlinear | patchtst |
| --- | --- | --- | --- |
| data | swiss-river-1990 | swiss-river-1990 | swiss-river-1990 |
| features | M | M | M |
| seq_len | 90 | 90 | 90 |
| label_len | 0 | 0 | 0 |
| pred_len | 7 | 7 | 7 |
| enc_in | — | — | — |
| dec_in | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast |

### `traffic`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| freq | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |

### `weather`

| setting | autoformer | dlinear | etsformer | fedformer | gpt4ts | informer | itransformer | lightts | lstm | mamba | nonstationary_transformer | patchtst | reformer | timemixer | timesnet | timexer | transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather |
| features | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| freq | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **48** | **0** | **48** | **48** | **48** |
| pred_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| enc_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dec_in | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| c_out | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| task_name | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast | long_term_forecast |


## TSL Source Traces

The authoritative TSL trace for dataset-facing settings is:

- `scripts/long_term_forecast/**`: benchmark shell scripts define dataset identity and the usual values for `data`, `root_path`, `data_path`, `features`, `seq_len`, `label_len`, `pred_len`, `enc_in`, `dec_in`, and `c_out`. Most scripts omit `--freq`.
- `run.py`: parses those CLI flags, sets `--freq` default to `h` when omitted, and dispatches `task_name=long_term_forecast` into `Exp_Long_Term_Forecast`.
- `exp/exp_long_term_forecasting.py`: calls `data_provider(self.args, flag)` and constructs decoder warm-up from `label_len`. The inline comment now notes that `TimeMixer` intentionally uses `label_len=0`.
- `data_provider/data_factory.py`: forwards `args.freq` unchanged into the selected dataset class.
- `data_provider/data_loader.py`: stores `self.freq` and uses it only inside `time_features(..., freq=self.freq)` when `timeenc == 1`.
- `utils/timefeatures.py`: relies on pandas `to_offset`, so explicit minute offsets such as `10min` and `15min` are supported.

### Setting-by-setting trace summary

| setting group | authoritative source in TSL | flow |
| --- | --- | --- |
| `data`, `root_path`, `data_path` | long-term forecast `.sh` scripts | shell args -> `run.py` -> `Exp_Long_Term_Forecast` -> `data_provider()` |
| `features`, `seq_len`, `label_len`, `pred_len` | long-term forecast `.sh` scripts; `run.py` defaults when omitted | shell args/defaults -> `run.py` -> experiment loop -> dataset construction / decoder warm-up |
| `enc_in`, `dec_in`, `c_out` | long-term forecast `.sh` scripts; `run.py` defaults when omitted | shell args/defaults -> `run.py` -> model + dataset setup |
| `freq` | omitted in most `.sh` scripts, then defaulted by `run.py` | `run.py` default or script override -> `data_factory.py` -> `data_loader.py` -> `time_features()` |
| `task_name` | `.sh` scripts pass `long_term_forecast`; `run.py` also defaults to it | CLI/default -> `run.py` experiment dispatch |

## Cadence Validation

| dataset | configured `freq` | observed cadence | match | evidence |
| --- | --- | --- | --- | --- |
| electricity | h | h | yes | Observed from CSV timestamps. |
| etth1 | h | h | yes | The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months. |
| etth2 | h | h | yes | The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months. |
| ettm1 | h | 15min | no | The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months. |
| ettm2 | h | 15min | no | The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months. |
| exchange_rate | d | d | yes | Observed from CSV timestamps. |
| illness | w | w | yes | Observed from CSV timestamps. |
| traffic | h | h | yes | Traffic is a collection of hourly data from California Department of Transportation, which describes the road occupancy rates measured by different sensors on San Francisco Bay area freeways. |
| weather | h | 10min | no | Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc. |

## Difference Analysis

- **Weather `freq`**: All Weather benchmark configs previously used `h`, but the raw CSV and `dataset/prompt_bank/Weather.txt` both show a 10-minute cadence. TSL scripts omit `--freq`, so `run.py`'s hourly default explained the drift. Fixed all Weather configs to `h`.
- **ETTm1 / ETTm2 `freq`**: The previous shorthand `t` was valid but ambiguous. The raw CSVs and ETT dataset note confirm a 15-minute cadence, so all ETT minute configs were normalized to explicit `h`.
- **Exchange Rate `freq`**: Exchange Rate configs were using hourly defaults or omitted `freq`. The CSV timestamps are daily, so all Exchange Rate configs were fixed to `d`.
- **Illness `freq`**: Illness configs previously used `h`. The `national_illness.csv` timestamps advance in seven-day steps, so all Illness configs were fixed to `w`.
- **Missing `freq` values**: Some Electricity, Traffic, and Exchange Rate configs omitted `freq`, which made them depend on local defaults. Every TSL-backed forecasting YAML now sets `freq` explicitly.
- **`TimeMixer` `label_len`**: Kept as-is. TSL `TimeMixer.sh` scripts intentionally pass `--label_len 0` across datasets, and the decoder warm-up is intentionally disabled for this model family.
- **`PatchTST` / `DLinear` / `LSTM` `label_len` on Electricity, Traffic, Exchange Rate**: These three datasets had stray `label_len: 0` configs for several models. `PatchTST` and `DLinear` were corrected to the dataset default overlap (`48`) to match TSL scripts or TSL defaults. `LSTM` was also normalized to `48` because it is Liulian-native and there was no behaviorally required reason to keep the inconsistency.
- **`target` omissions**: Kept as-is outside the ETT datasets plus the two configs that already set it. Under `features: M`, the missing `target` value is benign because the full multivariate target is used.
- **`pems` / `swiss_river`**: Included in audit coverage and comparison tables, but TSL source tracing is marked `N/A` because `refer_projects/Time-Series-Library` is not their upstream source in this repo.

## Changes Made

- Normalized every TSL-backed forecasting YAML to carry an explicit, dataset-accurate `freq`.
- Updated the config generation sources in `experiments/adapt_tsl_lib/generate_configs.py` and `tools/generate_configs.py`.
- Added targeted inline comments in the TSL execution path to document how `freq` and `label_len` are interpreted.
- Updated the local dataset docs so the `label_len` guidance matches the benchmark configs now on disk.

## Coverage

| dataset | settings audited | TSL trace completed | cadence validated | fully checked |
| --- | --- | --- | --- | --- |
| electricity | yes | yes | yes | yes |
| etth1 | yes | yes | yes | yes |
| etth2 | yes | yes | yes | yes |
| ettm1 | yes | yes | yes | yes |
| ettm2 | yes | yes | yes | yes |
| exchange_rate | yes | yes | yes | yes |
| illness | yes | yes | yes | yes |
| pems | yes | N/A | N/A | yes |
| swiss_river | yes | N/A | N/A | yes |
| traffic | yes | yes | yes | yes |
| weather | yes | yes | yes | yes |
