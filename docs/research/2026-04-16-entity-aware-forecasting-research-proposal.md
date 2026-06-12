# Entity-Aware Time Series Forecasting: A Comprehensive Research Proposal

_Date: 2026-04-16_
_Principal Investigator: Linlin Jia (jajupmochi@gmail.com)_
_Framework: LIULIAN (liulian-python)_

---

## Executive Summary

This proposal outlines a multi-period research program investigating **when, why, and how entity identifiers improve multi-entity time series forecasting**. Starting from a systematic ablation study across identifier types, architectures, and datasets, the program progressively expands into adaptive identifier selection, spatio-temporal graph integration, LLM-enhanced entity modeling, cross-domain transfer, and foundation model adaptation. Each period produces a publishable paper, with the first deliverable within one month.

---

## 1. Problem Statement and Motivation

Multi-entity time series forecasting — predicting future values for many homogeneous sensors, stations, or clients simultaneously — is fundamental to transportation, energy, hydrology, healthcare, and IoT. A critical design choice is **how to represent entity identity**: should a shared model learn that "sensor 42 behaves differently from sensor 17"?

Despite numerous approaches (learned embeddings, one-hot encoding, geographic coordinates, graph-based representations), **no systematic study compares these identifier types across architectures and datasets**. The field lacks:

1. A principled answer to "when do entity identifiers help?"
2. A unified framework for injecting identifiers across architecture families
3. Guidelines for selecting identifier types based on dataset properties
4. Understanding of how identifiers interact with modern paradigms (foundation models, LLMs, graph neural networks)

This research program addresses all four gaps.

---

## 2. Literature Review and Related Work

### 2.1 Channel-Independence vs Channel-Mixing in Time Series

The debate over whether multivariate time series models should treat channels independently or allow cross-channel interaction is central to our work.

- **PatchTST** (Nie et al., ICLR 2023): Demonstrated that channel-independent (CI) processing often outperforms channel-mixing (CM), challenging the assumption that cross-variate information always helps [1].
- **iTransformer** (Liu et al., ICLR 2024): Inverted the Transformer paradigm by treating each variate as a token, achieving strong results on Traffic and Electricity through implicit entity representation learning [2].
- **CARD** (Wang et al., ICLR 2024): Channel-Aligned Robust Blend Transformer adds channel-specific tokens (effectively learnable entity embeddings) to a Transformer backbone, showing gains over pure CI approaches [3].
- **CrossFormer** (Zhang & Yan, ICLR 2023): Uses cross-dimension attention with learnable dimension/segment embeddings, a form of entity identifier within the attention mechanism [4].
- **TSMixer** (Chen et al., NeurIPS 2023): MLP-based architecture demonstrating that simple cross-variate mixing can capture inter-entity relationships effectively [5].

**Gap**: These works each test one embedding strategy; none systematically compares identifier types across architectures.

### 2.2 Explicit Entity/Node Embeddings

- **STID** (Shao et al., CIKM 2022): Showed that a simple MLP + spatial identity embedding matches or beats complex spatio-temporal GNNs. Directly validates the "learnable entity embedding" approach [6].
- **TiDE** (Das et al., TMLR 2023): MLP-based encoder-decoder with per-variate static covariates (attribute embeddings), directly comparable to coordinate-based and learnable embedding approaches [7].
- **TimeXer** (Wang et al., NeurIPS 2024): Introduces exogenous variable tokens alongside endogenous patch tokens; the exogenous token design is analogous to entity embeddings for auxiliary context [8].
- **UniTS** (Gao et al., NeurIPS 2024): Unified model across multiple datasets/domains with dataset-specific prompt tokens — entity/domain identifiers at a higher level [9].

### 2.3 Spatio-Temporal Graph Neural Networks

Entity identifiers are closely related to node embeddings in spatial-temporal GNNs:

- **DCRNN** (Li et al., ICLR 2018): Diffusion convolutional recurrent NN modeling traffic as a directed graph; nodes are entities with spatial relationships [10].
- **Graph WaveNet** (Wu et al., IJCAI 2019): Learns adaptive adjacency matrices, effectively constructing entity relationships from data [11].
- **STEP** (Shao et al., ICML 2022 workshop): Pre-trains spatial-temporal embeddings, enabling zero-shot prediction for unseen nodes through transfer of entity representations [12].
- **STNorm** (Deng et al., AAAI 2021): Spatial and temporal normalization for traffic forecasting; entity-specific normalization as an alternative to explicit identifiers [13].
- **MegaCRN** (Jiang et al., AAAI 2023): Meta-graph convolutional recurrent network learning node-specific patterns through meta-learning [14].
- **PDFormer** (Jiang et al., AAAI 2023): Propagation delay-aware dynamic long-range transformer, incorporating spatial propagation patterns as entity relationships [15].
- **STAEformer** (Liu et al., CIKM 2023): Spatio-temporal adaptive embedding transformer, learning joint spatial-temporal embeddings per node [16].

**Gap**: GNN literature treats entity embeddings as node features but doesn't compare embedding strategies or study when simpler identifiers (one-hot, random) suffice.

### 2.4 LLM-Based Time Series Forecasting

Large language models bring new opportunities for entity-aware forecasting:

- **GPT4TS / One Fits All** (Zhou et al., NeurIPS 2023): Freezes GPT-2 backbone and adapts for time series tasks; entity context could be injected as text prompts [17].
- **Time-LLM** (Jin et al., ICLR 2024): Reprograms time series as text prototypes for LLM reasoning; entity descriptions could serve as natural language context [18].
- **TEST** (Sun et al., ICLR 2024): Text-prototype-aligned embedding for time series; aligns time series representations with textual descriptions of entities [19].
- **UniTime** (Liu et al., WWW 2024): Unified model across domains using domain/dataset-specific text instructions as conditioning [20].
- **CALF** (Liu et al., NeurIPS 2024): Cross-modal fine-tuning aligning time series with text representations [21].
- **S2IP-LLM** (Pan et al., NeurIPS 2024): Semantic space-informed prompt learning for temporal forecasting with LLMs [22].
- **LLMTime** (Gruver et al., NeurIPS 2023): Directly uses LLMs for zero-shot time series forecasting by encoding numbers as text [23].

**Gap**: No work investigates entity identifiers as text prompts ("this is sensor 42 on highway I-95, measuring traffic flow") vs learned embeddings vs positional encodings.

### 2.5 Foundation Models for Time Series

- **TimesFM** (Das et al., ICML 2024): Google's foundation model for time series; pre-trained on large corpus [24].
- **Chronos** (Ansari et al., ICML 2024): Amazon's tokenized time series foundation model [25].
- **Moirai** (Woo et al., ICML 2024): Salesforce's unified forecasting transformer with any-variate attention [26].
- **Timer** (Liu et al., ICML 2024): Generative pre-trained transformer for time series; single-series token paradigm [27].
- **Lag-Llama** (Rasul et al., NeurIPS 2023 workshop): Foundation model using lags as features [28].

**Gap**: How entity identifiers serve as adaptation/prompting mechanisms for frozen foundation models is unexplored.

### 2.6 LLM Agents for Time Series

- **AutoML for time series** (various): Automated model selection and hyperparameter tuning.
- **Data-Copilot** (Zhang et al., 2023): LLM agent for automated data analysis pipelines.
- **InsightPilot** (Ma et al., 2023): Automated data exploration using LLM agents.

**Gap**: No LLM agent system specifically targets entity-aware forecasting pipeline design — e.g., automatically deciding identifier type, constructing entity graphs from metadata, or selecting architectures based on dataset properties.

### 2.7 Entity Relationship Construction

- **Adaptive adjacency learning** in Graph WaveNet [11], MTGNN (Wu et al., KDD 2020) [29]: Learn entity relationships from data.
- **LLM-based graph construction**: Emerging work using LLMs to infer entity relationships from descriptions or metadata.
- **Knowledge-enhanced time series**: Injecting domain knowledge (entity hierarchies, spatial proximity) into forecasting models.

### 2.8 Cross-Domain Applications

- **Medical/ICU forecasting**: Patient-specific models for clinical time series (MIMIC-III/IV benchmarks); patient embeddings for ICU mortality prediction.
- **IoT sensor networks**: Smart building energy forecasting with sensor-specific adaptation; air quality prediction across monitoring stations.
- **Financial markets**: Entity-aware forecasting for stocks/instruments with sector embeddings.
- **Environmental monitoring**: River flow, weather station networks, soil moisture sensor arrays.

### 2.9 Robustness and Interpretability

- **Distribution shift**: How entity embeddings perform under concept drift when sensor behavior changes over time.
- **Missing entities**: Zero-shot prediction for new/unseen entities at inference time.
- **Embedding interpretability**: Visualization and analysis of learned entity representations (clustering, correlation with metadata).

---

## 3. Current Implementation Status

### 3.1 LIULIAN Framework

The LIULIAN framework provides a complete entity-aware forecasting pipeline:

- **Entity identifier modes**: `none`, `embedding` (learned nn.Embedding), `onehot`, `coordinates`, `sinusoidal`, `random`, `numeric_id`, `descriptors`
- **Architecture support**: LSTM, PatchTST, DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, GPT4TS, and more
- **Entity injection mechanisms**:
  - `EntityWrapper`: Per-entity embedding for per_entity split mode
  - `ChannelEntityWrapper`: Per-channel learned embedding for multi_channel mode
  - `ChannelTransparentWrapper`: Per-channel pre-computed features (onehot, sinusoidal, random, coordinates) for multi_channel mode
  - PatchTST native `add_after_patch`: Embedding injection in patch-token space
- **Experiment orchestration**: Matrix runner with 48 experiment combinations, HPO via Ray Tune, Slurm cluster dispatch, comparison reporting

### 3.2 Current Experiment Matrix

| Component | Values |
|---|---|
| Datasets | swiss-river-1990, traffic, electricity |
| Models | LSTM, PatchTST, DLinear |
| Identifier modes | none, embedding, onehot, sinusoidal, random (+coordinates for Swiss) |
| Seeds | Configurable (default: 1) |
| HPO | Ray Tune with ASHA, 10 samples default |
| Metrics | MSE, RMSE, MAE, NSE |

---

## 4. Research Program: Multi-Period Plan

### Period 1: Systematic Entity Identifier Ablation Study (Month 1)

**Paper Title**: "When and How Do Entity Identifiers Help? A Systematic Study of Identifier Types Across Architectures for Multi-Entity Time Series Forecasting"

**Target Venue**: NeurIPS 2026 Datasets & Benchmarks Track / AAAI 2027 / ECML-PKDD 2026

#### 4.1 Objectives
1. First systematic comparison of 5+ identifier types across 3+ architecture families on 5+ datasets
2. Empirical analysis of when identifiers help (dataset characteristics, entity correlation, entity count)
3. Practical guidelines for identifier selection
4. Open-source benchmark framework

#### 4.2 Experiment Design

**Phase 1A: Core ablation (Week 1-2)**

| Dimension | Values |
|---|---|
| Datasets | swiss-river-1990, traffic, electricity, PEMS04, PEMS08 |
| Models | LSTM, PatchTST, DLinear |
| Identifier modes | none, embedding, onehot, sinusoidal, random, coordinates (Swiss only) |
| Seeds | 3 seeds per combination |
| Prediction lengths | 96, 192, 336, 720 (standard TSL protocol) |
| Metrics | MSE, MAE (primary), RMSE, NSE (secondary) |

Total experiments: ~5 datasets x 3 models x 5 modes x 3 seeds x 4 pred_lens = ~900 runs

**Phase 1B: Extended architectures (Week 2-3)**

Add models: iTransformer, TimeMixer, TimeXer
Add negative-control datasets: Weather, ETTh1 (heterogeneous channels where IDs should NOT help)

**Phase 1C: Analysis and writing (Week 3-4)**

- Statistical significance tests (paired Wilcoxon signed-rank)
- Dataset characterization: inter-entity correlation matrix, entity count, data regime analysis
- Ablation insights: random vs learned (isolates distinction vs representation), coordinates vs embedding (domain knowledge vs data-driven)
- Visualization: t-SNE/UMAP of learned embeddings colored by geographic location, embedding similarity vs spatial distance

#### 4.3 Key Contributions
- C1: First comprehensive benchmark of entity identifier types across architectures
- C2: Empirical guidelines: "Use embeddings when N > 50 and entities are correlated; random baselines work when entities just need to be distinguished"
- C3: Open-source LIULIAN entity identifier framework and benchmark
- C4: Novel Swiss River hydrological dataset as a real-world evaluation domain

#### 4.4 Expected Insights
- Learned embeddings should dominate on large homogeneous datasets (traffic, electricity)
- Random baselines should perform surprisingly well (entity distinction matters more than representation quality)
- One-hot should degrade with large N (862 channels for traffic = very sparse)
- Sinusoidal should work well for ordered entities but poorly for unordered ones
- Coordinates should help specifically for spatially-correlated data (Swiss River)
- Negative controls (Weather, ETT) should show no benefit — channels are heterogeneous variables, not entities

#### 4.5 Timeline
| Week | Activities |
|---|---|
| 1 | Run core 3x3x5 matrix with 3 seeds on cluster; collect baseline results |
| 2 | Add extended models (iTransformer, TimeMixer, TimeXer); add PEMS/negative datasets |
| 3 | Statistical analysis, visualization, ablation analysis |
| 4 | Paper writing, figure preparation, camera-ready |

---

### Period 2: Adaptive Entity Identifier Selection (Month 2-3)

**Paper Title**: "AdaptID: Adaptive Entity Identifier Selection for Multi-Entity Time Series Forecasting via Meta-Learning"

**Target Venue**: ICML 2027 / NeurIPS 2027 main track

#### 4.6 Objectives
1. Learn to automatically select the best identifier type per dataset/entity
2. Develop a meta-learning module that gates over multiple identifier representations
3. Show that adaptive selection outperforms any single identifier type

#### 4.7 Technical Approach

**Multi-View Entity Representation**:
```
x_entity = Gate([emb(id), onehot(id), sinusoidal(id), random(id), coords(id)])
```

Where `Gate` is a learned attention mechanism over identifier views:
- **Input**: Dataset statistics (N_entities, temporal correlation matrix, spatial proximity if available)
- **Output**: Soft weights over identifier types per entity
- **Training**: End-to-end with forecasting objective + auxiliary identifier discrimination loss

**Architecture**:
1. Identifier Encoder: Produces K representations per entity (one per identifier type)
2. Dataset Profiler: Extracts dataset-level statistics (entity count, inter-entity MI, temporal autocorrelation)
3. Gating Network: Attention over K identifier views conditioned on dataset profile
4. Forecasting Backbone: Any architecture (LSTM, Transformer, etc.)

#### 4.8 Key Contributions
- C1: First adaptive identifier selection framework for time series
- C2: Meta-learning formulation enabling cross-dataset transfer of selection strategy
- C3: Theoretical analysis: connection between entity correlation structure and optimal identifier type
- C4: Identifier fusion consistently outperforms best single identifier

---

### Period 3: Entity Identifiers Meet Spatio-Temporal Graphs (Month 3-5)

**Paper Title**: "Beyond Node Embeddings: A Unified Framework for Entity Identification in Spatio-Temporal Forecasting"

**Target Venue**: KDD 2027 / AAAI 2027

#### 4.9 Objectives
1. Unify entity identifier approaches with graph neural network node embeddings
2. Study when explicit graph structure helps vs when learned identifiers suffice
3. Develop hybrid approaches: identifiers + adaptive graph construction

#### 4.10 Technical Approach

**Graph-Enhanced Entity Identification**:
- Start with entity identifiers (embedding, sinusoidal, etc.)
- Learn adaptive adjacency matrix from entity representations (cf. Graph WaveNet [11])
- Compare: predefined graph (if available) vs learned graph vs no graph + identifiers only
- Key insight: Entity identifiers can be seen as a special case of GNN node features without edges

**Entity Relationship Construction**:
- From data: Learn pairwise entity similarity from temporal patterns
- From metadata: Use geographic coordinates, network topology, domain knowledge
- From LLMs: Generate entity relationship descriptions and convert to graph structure (novel)

**Zero-Shot Entity Prediction (STEP-inspired)**:
- Pre-train entity embeddings on source entities
- Transfer to unseen entities via:
  - Interpolation from nearby known entities (spatial)
  - LLM-generated descriptions mapped to embedding space
  - Few-shot adaptation from limited samples

#### 4.11 Key Contributions
- C1: Unified framework connecting entity identifiers and GNN node embeddings
- C2: LLM-based entity graph construction from metadata
- C3: Zero-shot entity prediction via embedding transfer
- C4: Comprehensive comparison: identifiers-only vs GNN vs hybrid on standard benchmarks

---

### Period 4: LLM-Enhanced Entity-Aware Forecasting (Month 5-8)

**Paper Title**: "EntityLLM: Large Language Model-Enhanced Entity Representation for Universal Time Series Forecasting"

**Target Venue**: ICLR 2028 / NeurIPS 2027

#### 4.12 Objectives
1. Use LLMs to generate rich entity descriptions from metadata
2. Align entity text descriptions with learned embeddings
3. Enable zero-shot entity adaptation via natural language specification
4. Build LLM agent for automated entity-aware forecasting pipeline design

#### 4.13 Technical Approach

**Text-Enhanced Entity Identifiers**:
```
Entity description: "Sensor 42 is a traffic flow sensor on highway I-95, 
                     located at coordinates (40.7, -74.0), installed in 2015, 
                     measuring vehicle count in a suburban area."
→ LLM encoder → entity_text_emb → align with entity_learned_emb
```

**LLM Agent for Forecasting Pipeline**:
- Agent analyzes dataset metadata and recommends:
  - Identifier type (based on entity count, metadata availability, correlation structure)
  - Architecture choice (based on dataset size, prediction horizon, computational budget)
  - Graph structure (if spatial metadata available)
- Agent can be iterative: run initial experiments, analyze results, refine choices

**Multi-Modal Entity Representation**:
- Combine: time series patterns + text descriptions + spatial coordinates + images (satellite for hydrology)
- Attention-based fusion across modalities
- Each modality provides complementary entity information

#### 4.14 Key Contributions
- C1: First LLM-enhanced entity representation for time series
- C2: Natural language entity specification for zero-shot adaptation
- C3: LLM agent for automated entity-aware forecasting
- C4: Multi-modal entity fusion framework

---

### Period 5: Cross-Domain Entity-Aware Foundation Model (Month 8-12)

**Paper Title**: "UniEntity: A Universal Entity-Aware Foundation Model for Multi-Domain Time Series Forecasting"

**Target Venue**: ICML 2028 / Nature Machine Intelligence

#### 4.15 Objectives
1. Build a foundation model that natively supports entity identifiers across domains
2. Pre-train on diverse multi-entity datasets from multiple domains
3. Enable few-shot adaptation to new domains via entity-aware prompting
4. Demonstrate cross-domain entity transfer (e.g., traffic → hydrology)

#### 4.16 Technical Approach

**Domain-Agnostic Entity Framework**:
- Entity type ontology: sensor, patient, financial instrument, weather station, etc.
- Shared entity embedding space across domains
- Domain-specific entity adapters (LoRA-style)

**Cross-Domain Applications**:

| Domain | Entity Type | Entity Metadata | Key Challenge |
|---|---|---|---|
| Transportation | Traffic sensors | Location, road type, speed limit | Spatial correlation, periodic patterns |
| Energy | Power meters | Customer type, location, capacity | Diverse consumption patterns |
| Hydrology | River gauging stations | Coordinates, catchment area, elevation | Network topology, extreme events |
| Healthcare | ICU patients | Demographics, diagnoses, medications | High variability, missing data |
| IoT | Building sensors | Floor, room, sensor type | Heterogeneous sensor types |
| Finance | Stocks/instruments | Sector, market cap, country | Non-stationary, regime changes |
| Environment | Air quality monitors | Location, proximity to roads | Spatial dispersion patterns |

**Perioperative Patient Clustering** (specific application):
- Treat each patient as an entity in perioperative time series
- Entity features: demographics, surgical procedure type, comorbidities
- Task: Predict post-operative complications using patient-aware models
- Entity clustering reveals patient phenotypes with distinct outcome trajectories

#### 4.17 Key Contributions
- C1: First entity-aware foundation model for time series
- C2: Cross-domain entity transfer learning
- C3: Domain-specific entity adapter design
- C4: Application to perioperative patient forecasting

---

## 5. Missing Key Competitors to Include

The following architectures must be added to the LIULIAN framework for comprehensive comparison:

| Architecture | Why Critical | Effort |
|---|---|---|
| **iTransformer** [2] | Implicit entity tokens via variate-as-token; most direct competitor | Medium (adapter exists in TSL) |
| **STID** [6] | Simple MLP + spatial ID; proves identifiers alone can match GNNs | Low (simple architecture) |
| **CARD** [3] | Channel-specific tokens in Transformer | Medium |
| **TiDE** [7] | Static covariates per variate (= entity features) | Medium |
| **Graph WaveNet** [11] | Adaptive graph + node embeddings; GNN baseline | High |
| **DCRNN** [10] | Predefined graph + node features; classic GNN baseline | High |

---

## 6. Innovation Highlights

### 6.1 Immediate Innovations (Period 1)
- **Random embedding baseline**: Simple but crucial ablation — isolates "entity distinction" from "entity representation quality"
- **ChannelTransparentWrapper**: Novel architecture for injecting pre-computed features in multi-channel mode
- **Negative controls**: First study showing identifiers should NOT help on heterogeneous-channel datasets

### 6.2 Medium-Term Innovations (Period 2-3)
- **AdaptID meta-learning**: First adaptive identifier selection via gating attention
- **LLM-based graph construction**: Use LLMs to infer entity relationships from descriptions
- **Zero-shot entity prediction**: Transfer entity embeddings to unseen entities

### 6.3 Long-Term Innovations (Period 4-5)
- **Text-enhanced entity identifiers**: Natural language entity specification
- **LLM forecasting agent**: Automated pipeline design
- **Cross-domain entity foundation model**: Universal entity representation

---

## 7. Evaluation Protocol

### 7.1 Datasets

**Primary (homogeneous multi-entity)**:
| Dataset | Entities | Length | Domain | Source |
|---|---|---|---|---|
| Traffic | 862 | 17,544 | Road sensors | TSL benchmark |
| Electricity | 321 | 26,304 | Power clients | TSL benchmark |
| Swiss-River-1990 | ~50 | ~10,000 | Gauging stations | Novel (LIULIAN) |
| PEMS03/04/07/08 | 170-883 | 16,992-26,208 | Traffic sensors | STGNN benchmark |

**Negative controls (heterogeneous channels)**:
| Dataset | Channels | Why negative | Source |
|---|---|---|---|
| Weather | 21 | Channels are different physical variables | TSL benchmark |
| ETTh1/ETTm1 | 7 | Channels are heterogeneous measurements | TSL benchmark |

**Extended domains (Periods 3-5)**:
| Dataset | Domain | Source |
|---|---|---|
| METR-LA, PEMS-BAY | Spatial traffic | STGNN benchmark |
| MIMIC-III/IV | ICU patients | PhysioNet |
| Air quality | Environmental | EPA/WHO |
| Solar Energy | PV installations | NREL |

### 7.2 Metrics
- **Primary**: MSE, MAE (standard TSL protocol)
- **Secondary**: RMSE, MAPE, NSE (hydrology-specific)
- **Statistical**: Mean +/- std over 3+ seeds, Wilcoxon signed-rank test

### 7.3 Baselines
- No identifier (pure shared model)
- Each identifier type independently
- iTransformer (implicit entity tokens)
- STID (MLP + spatial ID)
- Graph WaveNet (learned graph + node embeddings)

---

## 8. Resource Requirements

| Resource | Period 1 | Period 2-3 | Period 4-5 |
|---|---|---|---|
| GPU hours | ~500 (cluster) | ~2000 | ~5000 |
| Storage | 50 GB | 200 GB | 1 TB |
| Human effort | 1 month (AI-assisted) | 3 months | 4 months |
| Compute | RTX 4090 cluster | A100 cluster | A100/H100 cluster |

---

## 9. Risk Analysis and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Entity identifiers show minimal benefit on most datasets | Medium | High | Frame as negative result (valuable); focus on "when" not "whether" |
| Random baselines match learned embeddings | Medium | Medium | Important finding — entity distinction >> representation quality |
| Scalability issues with large entity counts (862 for Traffic) | Low | Medium | One-hot degrades; embedding scales well; study this as a finding |
| Negative controls show unexpected benefit | Low | Low | Investigate and report; may reveal interesting channel interactions |
| LLM entity descriptions don't improve over numerical embeddings | Medium | Medium | Focus on zero-shot/transfer scenarios where text is uniquely valuable |

---

## 10. References

[1] Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.

[2] Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024.

[3] Wang, X., et al. "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting." ICLR 2024.

[4] Zhang, Y. & Yan, J. "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting." ICLR 2023.

[5] Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." NeurIPS 2023.

[6] Shao, Z., et al. "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting." CIKM 2022.

[7] Das, A., et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder." TMLR 2023.

[8] Wang, Y., et al. "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables." NeurIPS 2024.

[9] Gao, S., et al. "UniTS: Building a Unified Time Series Model." NeurIPS 2024.

[10] Li, Y., et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018.

[11] Wu, Z., et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019.

[12] Shao, Z., et al. "Pre-training Enhanced Spatial-Temporal Graph Neural Network for Multivariate Time Series Forecasting." KDD 2022.

[13] Deng, J., et al. "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting." KDD 2021.

[14] Jiang, R., et al. "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting." AAAI 2023.

[15] Jiang, J., et al. "PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction." AAAI 2023.

[16] Liu, H., et al. "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting." CIKM 2023.

[17] Zhou, T., et al. "One Fits All: Power General Time Series Analysis by Pretrained LM." NeurIPS 2023.

[18] Jin, M., et al. "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." ICLR 2024.

[19] Sun, C., et al. "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series." ICLR 2024.

[20] Liu, X., et al. "UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting." WWW 2024.

[21] Liu, C., et al. "CALF: Aligning LLMs for Time Series Forecasting via Cross-Modal Fine-Tuning." NeurIPS 2024.

[22] Pan, S., et al. "S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting." NeurIPS 2024.

[23] Gruver, N., et al. "Large Language Models Are Zero-Shot Time Series Forecasters." NeurIPS 2023.

[24] Das, A., et al. "A Decoder-Only Foundation Model for Time-Series Forecasting." ICML 2024.

[25] Ansari, A., et al. "Chronos: Learning the Language of Time Series." ICML 2024.

[26] Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024.

[27] Liu, Y., et al. "Timer: Generative Pre-trained Transformers Are Large Time Series Models." ICML 2024.

[28] Rasul, K., et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." NeurIPS 2023 Workshop.

[29] Wu, Z., et al. "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." KDD 2020.

[30] Shao, Z., et al. "Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting." VLDB 2022.

[31] Ji, J., et al. "STDEN: Towards Physics-Guided Neural Networks for Traffic Flow Prediction." AAAI 2022.

[32] Lan, S., et al. "DStagNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting." ICML 2022.

[33] Choi, J., et al. "Graph Neural Controlled Differential Equations for Traffic Forecasting." AAAI 2022.

[34] Chen, Y., et al. "Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting." IEEE TKDE 2023.

[35] Cini, A., et al. "Scalable Spatiotemporal Graph Neural Networks." AAAI 2023.

[36] Wen, Q., et al. "Transformers in Time Series: A Survey." IJCAI 2023.

[37] Liang, T., et al. "Foundation Models for Time Series Analysis: A Tutorial and Survey." KDD 2024 Tutorial.

[38] Jin, M., et al. "Position Paper: What Can Large Language Models Tell Us about Time Series Analysis." ICML 2024.

[39] Harutyunyan, H., et al. "Multitask Learning and Benchmarking with Clinical Time Series Data." Scientific Data 2019. (MIMIC benchmark)

[40] Tang, S., et al. "Spatiotemporal Multi-Graph Convolution Network for Ride-Hailing Demand Forecasting." AAAI 2020.

---

## 11. Detailed Period 1 Execution Plan (1 Month)

### Week 1: Infrastructure and Core Experiments

**Day 1-2: Framework readiness**
- [ ] Verify all 48 matrix jobs pass dry-run
- [ ] Add iTransformer, TimeMixer, TimeXer to experiment matrix
- [ ] Add PEMS04, PEMS08 datasets to matrix
- [ ] Add Weather, ETTh1 as negative control datasets
- [ ] Increase default seeds to 3 (2024, 2025, 2026)
- [ ] Set up prediction length sweep (96, 192, 336, 720) as config option

**Day 3-4: Cluster submission**
- [ ] Submit Phase 1A core matrix to cluster (swiss, traffic, electricity x LSTM, PatchTST, DLinear x 5 modes x 3 seeds x 4 pred_lens)
- [ ] Monitor first batch; fix any pipeline issues
- [ ] Submit extended models (iTransformer, TimeMixer, TimeXer)

**Day 5-7: Extended datasets**
- [ ] Submit PEMS04, PEMS08 experiments
- [ ] Submit negative controls (Weather, ETTh1 x all models x none+embedding only)
- [ ] Collect first batch results; validate sanity

### Week 2: Full Experimental Campaign

**Day 8-10: Result collection and validation**
- [ ] Verify all Phase 1A results are complete
- [ ] Run comparison reports; check for anomalies
- [ ] Re-run any failed experiments
- [ ] Begin extended model experiments on cluster

**Day 11-14: Additional analyses**
- [ ] Compute inter-entity correlation matrices for each dataset
- [ ] Run t-SNE/UMAP visualization of learned embeddings
- [ ] Compute embedding similarity vs spatial distance (Swiss River)
- [ ] Generate comprehensive comparison tables

### Week 3: Analysis and Figures

**Day 15-17: Statistical analysis**
- [ ] Compute mean +/- std across seeds for all experiments
- [ ] Run Wilcoxon signed-rank tests: embedding vs none, each mode vs none
- [ ] Create performance improvement heatmaps (dataset x model x mode)
- [ ] Analyze effect of entity count on identifier benefit

**Day 18-21: Visualization and insights**
- [ ] Create main results table (all datasets x models x modes)
- [ ] Create figure: performance vs entity count
- [ ] Create figure: embedding t-SNE colored by geography
- [ ] Create figure: identifier type recommendation flowchart
- [ ] Draft key findings and practical guidelines

### Week 4: Paper Writing

**Day 22-24: Paper structure and draft**
- [ ] Write introduction and motivation (positioning vs PatchTST CI debate)
- [ ] Write related work (comprehensive, from Section 2)
- [ ] Write methodology (LIULIAN framework, identifier types, injection mechanisms)
- [ ] Write experimental setup

**Day 25-27: Results and analysis**
- [ ] Write main results section with tables and figures
- [ ] Write analysis section (when do identifiers help? why?)
- [ ] Write ablation studies (random vs learned, scaling with N, negative controls)
- [ ] Write practical guidelines section

**Day 28-30: Polish and submit**
- [ ] Write abstract and conclusion
- [ ] Finalize all figures and tables
- [ ] Internal review and revision
- [ ] Prepare supplementary material (full tables, code, configs)
- [ ] Submit to arXiv; prepare venue-specific submission

---

## 12. Summary of Research Contributions Across All Periods

| Period | Contribution | Novelty Level | Impact |
|---|---|---|---|
| 1 | Systematic identifier type benchmark | **High** — first comprehensive study | Benchmark paper; establishes baselines for the field |
| 2 | Adaptive identifier selection (AdaptID) | **Very High** — novel meta-learning module | Method paper; practical tool for practitioners |
| 3 | Unified identifier + GNN framework | **High** — bridges two research communities | Framework paper; connects identifier and GNN worlds |
| 4 | LLM-enhanced entity representation | **Very High** — new paradigm | Vision paper; defines new research direction |
| 5 | Cross-domain entity foundation model | **Extremely High** — ambitious scope | If successful, very high citation potential |

---

## Appendix A: LIULIAN Entity Identifier Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Entity Identifier Layer       │
                    │                                       │
                    │  none ──── pass-through                │
                    │  embedding ── nn.Embedding → project   │
                    │  onehot ──── I_N matrix → project      │
                    │  sinusoidal ─ PE(idx) → project        │
                    │  random ──── hash(seed:name) → project │
                    │  coordinates ─ (lat,lon) → project     │
                    └──────────┬────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼─────────┐
    │  EntityWrapper  │  │ Channel    │  │ Channel        │
    │  (per_entity)   │  │ Entity     │  │ Transparent    │
    │                 │  │ Wrapper    │  │ Wrapper        │
    │  emb → concat   │  │ (learned)  │  │ (pre-computed) │
    │  → project      │  │            │  │                │
    │  → inner model  │  │ emb per ch │  │ feat per ch    │
    └─────────────────┘  │ → concat   │  │ → concat       │
                         │ → project  │  │ → project      │
                         │ → inner    │  │ → inner        │
                         └────────────┘  └────────────────┘
                                ↓
                    ┌──────────────────────┐
                    │   Forecasting Model   │
                    │  (LSTM / PatchTST /   │
                    │   DLinear / iTransf.) │
                    └──────────────────────┘
```

## Appendix B: Code Artifacts

| File | Description |
|---|---|
| `experiments/entity_identifier/run.py` | Matrix experiment runner |
| `experiments/entity_identifier/matrix.py` | Matrix definition and job expansion |
| `experiments/entity_identifier/compare.py` | Comparison report generator |
| `experiments/entity_identifier/submit_slurm.py` | Slurm cluster dispatcher |
| `liulian/models/torch/entity_mixin.py` | EntityWrapper, ChannelEntityWrapper, ChannelTransparentWrapper |
| `liulian/data/ts/timeseriesdataset.py` | make_entity_features and transparent mode data layer |
| `experiments/run.py` | Unified experiment entry point |
| `tests/runtime/test_entity_identifier_pipeline.py` | Pipeline tests (15 tests) |
