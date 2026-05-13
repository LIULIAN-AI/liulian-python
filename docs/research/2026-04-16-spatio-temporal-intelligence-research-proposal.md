# LIULIAN: From Entity-Aware Forecasting to Universal Spatio-Temporal Intelligence — A Comprehensive Research Program

_Date: 2026-04-16 (revised)_
_Principal Investigator: Linlin Jia (jajupmochi@gmail.com)_
_Framework: LIULIAN — Liquid Intelligence and Unified Logic for Interactive Adaptive Networks_

---

## Executive Summary

This proposal outlines a **multi-year, multi-period research program** that starts from a focused study on entity identifiers in time series forecasting and progressively expands into a **universal spatio-temporal intelligence platform**. The program spans seven research periods across three phases:

- **Phase A (Months 1–5):** Entity-aware forecasting — systematic ablation, adaptive selection, and spatial graph integration. Three papers.
- **Phase B (Months 5–12):** Spatio-temporal foundation models — LLM-enhanced ST modeling, cross-domain ST foundation models, and Mamba/SSM architectures for ST data. Three papers.
- **Phase C (Months 8–18):** The LIULIAN platform itself — an open-source, agent-powered, BI-integrated spatio-temporal intelligence platform with vertical domain plugins. One system paper + domain application papers.

Each period produces a publishable paper. The platform itself is a separate, high-impact contribution targeting top-tier systems venues. Cross-domain applications (hydrology, healthcare, disaster management, finance, IoT, materials science) serve as validation and generate additional domain-specific publications.

**Total planned outputs:** 7+ core papers, 1 platform/system paper, 3+ domain application papers.

---

## 1. Problem Statement and Motivation

### 1.1 The Spatio-Temporal Data Challenge

Spatio-temporal data — measurements distributed across both space and time — is the backbone of modern sensing, infrastructure monitoring, and scientific discovery. Transportation networks generate millions of sensor readings per hour; hospital ICUs produce continuous patient vitals across beds; river monitoring stations track water quality across watersheds; satellite imagery captures land-use change across continents.

Despite the ubiquity of spatio-temporal data, the field suffers from **fragmentation** along multiple axes:

1. **Methodological fragmentation**: Time series models (PatchTST, DLinear, iTransformer) ignore spatial structure; graph neural networks (DCRNN, Graph WaveNet, MTGNN) focus on fixed topologies; LLM-based methods (Time-LLM, UrbanGPT) lack systematic entity/spatial awareness.

2. **Domain fragmentation**: Traffic forecasting, air quality prediction, hydrological modeling, clinical time series, and financial analysis each have their own datasets, metrics, baselines, and conventions — with minimal cross-pollination.

3. **Infrastructure fragmentation**: Researchers must cobble together TSLib [1] for temporal models, PyG [2] for graph models, GluonTS [3] for probabilistic forecasting, and custom scripts for domain-specific preprocessing. No single platform supports the full spectrum from raw spatio-temporal data to scientific insight.

4. **Entity representation fragmentation**: A fundamental design choice — how to represent entity identity in multi-entity systems — has never been systematically studied. Different works use learned embeddings [6], one-hot encoding, geographic coordinates, graph node features [10,11], or text descriptions [18] without principled comparison.

### 1.2 Vision

This research program addresses all four forms of fragmentation through a unified approach:

- **Methodological unification**: A framework that seamlessly combines entity identifiers, graph structure, temporal models, and LLM-enhanced representations.
- **Domain unification**: Cross-domain evaluation on 15+ datasets spanning transportation, energy, hydrology, healthcare, IoT, and finance.
- **Infrastructure unification**: The LIULIAN platform — an open-source, modular, agent-powered spatio-temporal intelligence system with BI capabilities and vertical domain plugins.
- **Entity representation unification**: The first systematic study of entity representation strategies, progressing from ablation to adaptive selection to foundation-model-level entity awareness.

---

## 2. Literature Review and Related Work

### 2.1 Channel-Independence vs Channel-Mixing in Time Series

The debate over whether multivariate time series models should treat channels independently or allow cross-channel interaction is foundational.

- **PatchTST** (Nie et al., ICLR 2023) [1]: Channel-independent (CI) patching often outperforms channel-mixing, challenging the assumption that cross-variate information always helps.
- **iTransformer** (Liu et al., ICLR 2024) [4]: Inverts the Transformer paradigm — variates as tokens — achieving strong results through implicit entity representation.
- **CARD** (Wang et al., ICLR 2024) [5]: Channel-Aligned Robust Blend Transformer adds channel-specific tokens (learnable entity embeddings).
- **CrossFormer** (Zhang & Yan, ICLR 2023) [6]: Cross-dimension attention with learnable dimension/segment embeddings.
- **TSMixer** (Chen et al., NeurIPS 2023) [7]: MLP-based cross-variate mixing captures inter-entity relationships.
- **Client** (Gao et al., KDD 2024) [8]: Integrates cross-variable linear components with transformer attention to balance CI with cross-variate sharing.

**Gap**: Each tests one embedding strategy; none systematically compares identifier types across architectures.

### 2.2 Explicit Entity/Node Embeddings

- **STID** (Shao et al., CIKM 2022) [9]: Simple MLP + spatial identity embedding matches complex GNNs — directly validates learnable entity embeddings. **Key insight: spatial indistinguishability, not architecture complexity, is the bottleneck.**
- **TiDE** (Das et al., TMLR 2023) [10]: MLP encoder-decoder with per-variate static covariates (attribute embeddings).
- **TimeXer** (Wang et al., NeurIPS 2024) [11]: Exogenous variable tokens alongside endogenous patch tokens.
- **UniTS** (Gao et al., NeurIPS 2024) [12]: Unified multi-task model with dataset-specific prompt tokens — entity/domain identifiers at a higher level.
- **TRA** (Lin et al., AAAI 2023) [13]: Temporal Routing Adaptor for entity-aware stock forecasting — entity-specific routing to select temporal patterns per stock.

### 2.3 Spatio-Temporal Graph Neural Networks

#### 2.3.1 Classic and Foundational ST-GNNs

- **DCRNN** (Li et al., ICLR 2018) [14]: Diffusion convolutional recurrent NN on directed traffic graphs.
- **Graph WaveNet** (Wu et al., IJCAI 2019) [15]: Learns adaptive adjacency matrices from data.
- **MTGNN** (Wu et al., KDD 2020) [16]: Connecting the dots — multivariate TS forecasting with learned graph structures.
- **STEP** (Shao et al., KDD 2022) [17]: Pre-trains spatial-temporal embeddings, enables zero-shot prediction for unseen nodes.
- **STNorm** (Deng et al., KDD 2021) [18]: Entity-specific normalization as alternative to explicit identifiers.
- **MegaCRN** (Jiang et al., AAAI 2023) [19]: Meta-graph convolutional recurrent network with a Meta-Node Bank.
- **PDFormer** (Jiang et al., AAAI 2023) [20]: Propagation delay-aware dynamic long-range transformer.
- **STAEformer** (Liu et al., CIKM 2023) [21]: Spatio-temporal adaptive embedding transformer, joint spatial-temporal embeddings per node.

#### 2.3.2 Adaptive Graph Learning (Beyond Fixed Topologies)

- **TESTAM** (Lee et al., ICLR 2024) [22]: MoE with three experts (no graph, static graph, dynamic graph) — reformulates routing as classification. **Novelty check: closest to our AdaptID concept, but gates over graph structures, not identifier types.**
- **STID** [9]: Proves identifiers alone can replace graphs entirely. **Critical baseline for our work.**

#### 2.3.3 Scalable ST-GNNs

- **SGP** (Cini et al., AAAI 2023) [23]: Randomized recurrent + adjacency powers enables node-wise parallel training.
- **BigST** (Han et al., VLDB 2024) [24]: Linear-complexity spatial convolution, scales to 100K+ nodes.
- **EasyST** (Tang et al., arXiv 2024) [25]: Knowledge distillation from teacher STGNN into student MLP.

#### 2.3.4 Spatio-Temporal Transformers

- **AirFormer** (Liang et al., AAAI 2023) [26]: Decoupled ST learning with deterministic + stochastic stages for 1,085 air quality stations.
- **EarthFormer** (Gao et al., NeurIPS 2022) [27]: Cuboid Attention for 3D spatio-temporal Earth observation data.

#### 2.3.5 Mamba/SSM for Spatio-Temporal Data

- **SpoT-Mamba** (Choi et al., IJCAI 2024 Workshop) [28]: Selective state space models on ST graphs via node-specific walk sequences.
- **STG-Mamba** (Li et al., arXiv 2024) [29]: First exploration of Mamba for full ST graph learning with ST-S3M module.

**Gap**: GNN literature treats entity embeddings as node features but never compares embedding strategies or studies when simpler identifiers suffice. No work bridges the entity-identifier literature with the GNN node-embedding literature.

### 2.4 Pre-Training and Foundation Models for Spatio-Temporal Data

- **GPT-ST** (Li et al., NeurIPS 2023) [30]: Generative pre-training for ST-GNNs with adaptive masking.
- **UniST** (Yuan et al., KDD 2024) [31]: Universal ST model with spatio-temporal knowledge-guided prompts, zero-shot generalization across 20+ urban scenarios.
- **OpenCity** (Li et al., ACM TIST 2024) [32]: Combines Transformers + GNNs, pre-trains on heterogeneous traffic data for zero-shot transfer.

**Gap**: All ST foundation models are traffic/urban-centric. None supports entity-aware conditioning across domains (hydrology, healthcare, IoT, finance).

### 2.5 Time Series Foundation Models

- **TimesFM** (Das et al., ICML 2024) [33]: Google's decoder-only model pre-trained on 100B time points.
- **Chronos** (Ansari et al., ICML 2024) [34]: Amazon's tokenized TS model using T5 architecture.
- **Moirai** (Woo et al., ICML 2024) [35]: Salesforce's any-variate attention with mixture distributions.
- **Timer** (Liu et al., ICML 2024) [36]: GPT-style generative pre-training for TS in S3 format.
- **MOMENT** (Goswami et al., ICML 2024) [37]: Masked pre-training on the Time Series Pile.
- **Lag-Llama** (Rasul et al., ICML 2024) [38]: LLaMA-inspired with lag features for probabilistic forecasting.
- **TTM** (Ekambaram et al., NeurIPS 2024) [39]: Tiny Time Mixers — <1M params, competitive zero-shot.
- **Time-MoE** (Shi et al., arXiv 2024) [40]: Billion-scale TS foundation model with sparse MoE.
- **TimeGPT-1** (Garza & Mergenthaler-Canseco, arXiv 2023) [41]: First production API TS foundation model.

**Gap**: All TS foundation models treat series as anonymous channels. None has explicit entity-identity conditioning. **This is the central gap our program fills.**

### 2.6 LLM-Based Time Series Forecasting

- **GPT4TS / One Fits All** (Zhou et al., NeurIPS 2023) [42]: Frozen GPT-2 with minimal adaptation.
- **Time-LLM** (Jin et al., ICLR 2024) [43]: Reprograms frozen LLMs via input transformations.
- **TEST** (Sun et al., ICLR 2024) [44]: Text-prototype-aligned embeddings for TS.
- **UniTime** (Liu et al., WWW 2024) [45]: Domain-specific text instructions as conditioning.
- **CALF** (Liu et al., NeurIPS 2024) [46]: Cross-modal fine-tuning aligning TS with text.
- **S2IP-LLM** (Pan et al., ICML 2024) [47]: Semantic space-informed prompt learning.
- **LLMTime** (Gruver et al., NeurIPS 2023) [48]: Zero-shot TS forecasting by encoding numbers as text.
- **TEMPO** (Cao et al., ICLR 2024) [49]: Trend/seasonal/residual decomposition with soft prompts per component.
- **AutoTimes** (Liu et al., NeurIPS 2024) [50]: Autoregressive in-context learning with LLMs.
- **"Are Language Models Actually Useful for Time Series Forecasting?"** (Tan et al., NeurIPS 2024) [51]: **Critical paper** — shows LLM backbone can be replaced by simple attention or random embeddings. Questions whether LLM pre-training truly transfers.

**Gap**: No work investigates entity identifiers as text prompts vs learned embeddings vs positional encodings. No work combines LLM entity descriptions with spatio-temporal graph structure.

### 2.7 Spatio-Temporal LLMs

- **UrbanGPT** (Li et al., KDD 2024) [52]: ST dependency encoder + instruction-tuned LLM for urban computing with zero-shot cross-city transfer.
- **ST-LLM** (Cao et al., arXiv 2024) [53]: Tokenized ST sequences fed directly to LLMs with spatial positional embeddings.
- **TrafficGPT** (Zhang et al., Transport Policy 2024) [54]: ChatGPT + traffic foundation models for conversational traffic analysis.

**Gap**: ST-LLMs focus on traffic/urban. None handles arbitrary spatio-temporal domains or integrates entity metadata as natural language.

### 2.8 LLM Agents for Data Analysis and Scientific Discovery

- **Data-Copilot** (Zhang et al., arXiv 2023) [55]: Autonomous data querying, processing, visualization workflows.
- **InsightPilot** (Ma et al., VLDB 2024) [56]: Iterative data exploration via LLM-generated analysis actions.
- **The AI Scientist** (Lu et al., arXiv 2024) [57]: End-to-end research automation — idea generation through paper writing.
- **DS-Agent** (Guo et al., ICML 2024) [58]: Case-based reasoning for Kaggle-style ML problems.
- **MLAgentBench** (Huang et al., ICML 2024) [59]: Benchmark for LLM agents on ML experimentation.
- **AutoML-GPT** (Zhang et al., arXiv 2023) [60]: GPT-4 for automatic model selection and pipeline composition.
- **MatPlotAgent** (Li et al., ACL 2024) [61]: Agent for publication-quality scientific visualization.
- **ResearchAgent** (Baek et al., ACL 2024) [62]: Iterative research hypothesis generation with review-refine loop.
- **ChemCrow** (Bran et al., Nature Machine Intelligence 2024) [63]: LLM agent with 18 chemistry tools for synthesis planning.
- **Coscientist** (Boiko et al., Nature 2023) [64]: LLM-driven autonomous chemistry experiments with robotic labs.

**Gap**: No LLM agent system targets spatio-temporal forecasting pipeline design — automatically deciding entity representation, graph construction, architecture selection, and HPO based on dataset properties. **This is a novel contribution we propose.**

### 2.9 AI-Powered Business Intelligence and Natural Language Interfaces

- **LIDA** (Dibia, Microsoft Research, ACL 2023) [65]: Multi-stage LLM pipeline for grammar-agnostic visualization.
- **DIN-SQL** (Pourreza & Rafiei, NeurIPS 2023) [66]: Decomposed text-to-SQL with self-correction.
- **DAIL-SQL** (Gao et al., VLDB 2024) [67]: Efficient few-shot text-to-SQL.
- **NL4DV** (Narechania et al., IEEE VIS 2023) [68]: Natural language to Vega-Lite visualization specs.
- **Chat2VIS** (Maddigan & Susnjak, IEEE Access 2023) [69]: LLM-generated Python visualization code from natural language.
- **TableGPT** (Li et al., arXiv 2023) [70]: LLM fine-tuned for holistic table understanding.
- **Rath** (Yu et al., Kanaries, 2022–2024) [71]: Open-source automated data exploration and visualization.
- **Sheet2Report** (Chen et al., arXiv 2024) [72]: LLM pipeline for professional analyst-style reports from spreadsheet data.

**Gap**: No BI platform is integrated with spatio-temporal ML. Current BI tools lack graph-aware analytics, spatial visualization, and domain-specific ST model integration.

### 2.10 ML/AI Platforms for Time Series and Graphs

| Platform | Spatial | Missing Data | Probabilistic | Multi-Task | Plugin Arch | Entity-Aware | LLM/Agent | BI |
|---|---|---|---|---|---|---|---|---|
| **TSLib** [73] | No | No | No | Yes (5) | No | No | No | No |
| **BasicTS** [74] | Yes | No | No | Forecasting | No | No | No | No |
| **tsl** [75] | Yes | Yes | No | Forecast+Impute | Partial | No | No | No |
| **GluonTS** [3] | No | No | Yes | Forecasting | Yes | No | No | No |
| **NeuralForecast** [76] | Limited | No | Yes | Forecasting | No | No | No | No |
| **Darts** [77] | No | No | Yes | Forecasting | No | No | No | No |
| **PyPOTS** [78] | No | Yes | No | Yes (4+) | Yes | No | No | No |
| **LibCity** [79] | Yes | No | No | Urban tasks | Partial | Implicit | No | No |
| **PyG** [2] | Yes | No | No | General GNN | Yes | Node features | No | No |
| **LIULIAN (ours)** | **Yes** | **Planned** | **Planned** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

**Gap**: No existing platform combines (a) entity-aware modules, (b) graph+TS unified modeling, (c) LLM agent integration, (d) BI features, and (e) vertical domain plugins. **LIULIAN fills this gap.**

### 2.11 Cross-Domain Applications

#### Healthcare / Clinical Time Series
- **STraTS** (Tipirneni & Reddy, MLHC 2022) [80]: Sparse attention on irregularly-sampled clinical TS with self-supervised pretraining on MIMIC-III.
- **MedGTX** (Kim et al., CHIL 2023) [81]: Graph-Transformer for patient outcome prediction combining patient relationship graphs with temporal EHR.
- **MISTS** (Li et al., AAAI 2024) [82]: Multi-input multi-output irregular sampled TS for ICU forecasting.
- **MedTsLLM** (Jiang et al., arXiv 2024) [83]: Clinical TS + clinical notes via LLM multimodal fusion.
- **Perioperative GNN** (Gao et al., npj Digital Medicine 2024) [84]: Patient graphs for post-operative complication prediction.

#### Environmental / Hydrology
- **AirFormer** [26]: Nationwide air quality prediction across 1,085 stations.
- **GAGNN** (various, ACM TKDD 2024) [85]: Group-aware GNN for nationwide PM2.5 forecasting.
- **FloodCast** (Xu et al., arXiv 2023) [86]: River network topology GNN for flood prediction.
- **ST water quality GNN** (various, Water Research 2025) [87]: Multi-scale ST-GNN for dissolved oxygen and nitrogen.
- **Water distribution GNN** (Zanfei et al., Water Research 2023) [88]: GNN for pipe network leak detection.
- **Multi-scale water demand STGNN** (various, Water Research 2025) [89]: Multi-scale temporal+spatial for water infrastructure.

#### Finance
- **FinGPT** (Yang et al., NeurIPS 2023 Workshop) [90]: Open-source financial LLMs with TS sentiment.
- **TRA** [13]: Temporal Routing Adaptor for entity-aware stock forecasting.
- **MASTER** (Li et al., AAAI 2024) [91]: Market-guided Stock Transformer for portfolio-level forecasting.

#### Disaster Management / Earth Science
- **EarthFormer** [27]: Cuboid Attention for precipitation, temperature, sea surface forecasting.
- **Prithvi** (Jakubik et al., IBM/NASA, arXiv 2023) [92]: Foundation model for geospatial AI.
- **GeoChat** (Kuckreja et al., CVPR 2024) [93]: Vision-language model for remote sensing analysis.
- **SatCLIP / GeoCLIP** (Klemmer et al., NeurIPS 2023 Workshop / AAAI 2024) [94]: Location-aware embeddings aligning satellite imagery with geographic metadata.
- **HeatGNN** (various, arXiv 2024) [95]: Epidemiology-informed GNN for pandemic forecasting.

#### IoT / Smart Buildings
- **Brick Schema + ML** (Balaji et al., BuildSys 2022) [96]: GNN-based cross-building transfer learning.
- **CityLearn** (Vazquez-Canteli et al., Applied Energy 2023) [97]: RL with building energy digital twins.

### 2.12 Contrastive and Self-Supervised Learning for ST Data

- **TS2Vec** (Yue et al., AAAI 2022) [98]: Hierarchical contrastive learning for universal TS representations.
- **TF-C** (Zhang et al., NeurIPS 2022) [99]: Time-frequency consistency contrastive pre-training.
- **SimMTM** (Dong et al., NeurIPS 2023) [100]: Masked TS modeling with neighbor aggregation.
- **AutoST** (Zhang et al., WWW 2023) [101]: Automated ST graph contrastive learning.
- **STAG** (Zhang et al., ICML 2023) [102]: Adversarial contrastive adaptation for ST graph learning.
- **CaST** (Xia et al., NeurIPS 2023) [103]: Causal lens for ST graph forecasting with back-door and front-door adjustment.
- **STONE** (Wang et al., KDD 2024) [104]: Handles both spatial and temporal distribution shifts.

### 2.13 Distribution Shift and Robustness in ST Models

- **ADATIME** (Ragab et al., ACM TKDD 2023) [105]: Benchmark of 11 domain adaptation methods on TS.
- **RAINCOAT** (He et al., ICML 2023) [106]: Frequency-based feature alignment for TS domain adaptation.

### 2.14 Explainability in Spatio-Temporal Models

- **STExplainer** (Zhang et al., KDD 2023) [107]: Post-hoc explanations for ST-GNN via mutual information perturbation.
- **TimeSHAP** (Bento et al., KDD 2021) [108]: SHAP for sequential models — feature importance over time steps.
- **TFT** (Lim et al., IJF 2021) [109]: Built-in interpretability via variable selection networks.

### 2.15 Plugin/Adapter Architectures in ML Frameworks

- **LoRA** (Hu et al., ICLR 2022) [110]: Parameter-efficient adapter layers for frozen models — foundational for our domain adapter design.
- **Ludwig** (Molino et al., JMLR 2023) [111]: Declarative YAML + encoder-combiner-decoder plugin architecture.
- **MMEngine** (OpenMMLab, 2022) [112]: Registry + hook + runner architecture pattern.
- **Hydra** (Yadan, Facebook Research, 2020) [113]: Composable YAML configs with plugin architecture.

### 2.16 Dynamic and Temporal Graph Networks

- **TGN** (Rossi et al., ICML 2020 Workshop) [114]: Memory + graph attention + temporal encoding on continuous-time dynamic graphs.
- **DyGFormer** (Yu et al., NeurIPS 2023) [115]: Transformer for continuous-time dynamic graphs with neighbor co-occurrence encoding.
- **ROLAND** (You et al., KDD 2022) [116]: Incremental GNN for dynamic social networks.

### 2.17 Multi-Modal Spatio-Temporal Data

- **Time-MMD** (Liu et al., NeurIPS 2024 D&B) [117]: Large-scale benchmark pairing TS with text metadata across 9 domains.
- **ChatTS** (Xie et al., arXiv 2024) [118]: Synthetic TS-text pairs for LLM temporal reasoning.
- **UniST** [31]: Multi-modal prompts (spatial, temporal, textual) for urban ST prediction.
- **TorchGeo** (Stewart et al., SIGSPATIAL 2022) [119]: Bridges GIS data formats with DL training pipelines.

### 2.18 Benchmarking and Fair Evaluation

- **TSLib** (Wu et al., NeurIPS 2023 D&B) [73]: Fair re-implementation revealed many SOTA results were evaluation artifacts.
- **BasicTS** (Shao et al., NeurIPS 2023 D&B) [74]: Simpler models often match complex GNNs under fair evaluation.
- **TFB** (Qiu et al., ICML 2024) [120]: 8,068 univariate + 25 multivariate datasets challenge claimed SOTA.
- **TOTEM** (Talukder et al., NeurIPS 2024) [121]: Discrete tokenization via VQ-VAE creating shared vocabulary across domains.

### 2.19 Human-in-the-Loop ML and Report Generation

- **HITL-TSF** (Li et al., Salesforce, arXiv 2024) [122]: Framework for injecting domain knowledge into neural forecasting.
- **Calliope** (Shi et al., IEEE TVCG 2023) [123]: Automatic visual data stories from spreadsheets.
- **DataTales** (Sultan et al., CHI 2024) [124]: LLM-generated narrative articles from datasets.

### 2.20 Digital Twins and Simulation + ML

- **TwinGen** (Thelen et al., Nature Machine Intelligence 2023) [125]: Physics-informed neural networks for data-efficient digital twins.
- **Transolver** (Wu et al., ICML 2024 Spotlight) [126]: Fast transformer PDE solver on general geometries.
- **PARCv2** (Nguyen et al., ICML 2024) [127]: Physics-aware recurrent convolutions for spatiotemporal dynamics.

---

## 3. Current Implementation Status

### 3.1 LIULIAN Framework Architecture

LIULIAN (Liquid Intelligence and Unified Logic for Interactive Adaptive Networks) provides a modular, task-driven spatio-temporal intelligence pipeline:

```
                    ┌───────────────────────────────────────┐
                    │        LIULIAN Platform Stack          │
                    │                                       │
                    │  ┌─────────────────────────────────┐  │
                    │  │   LLM Agent Layer (planned)      │  │
                    │  │   • Pipeline design agent        │  │
                    │  │   • Data ingestion agent         │  │
                    │  │   • Report generation agent      │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   BI & Visualization Layer       │  │
                    │  │   • Dashboards (planned)         │  │
                    │  │   • NL query interface (planned) │  │
                    │  │   • Automated reports (planned)  │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   Task Layer                     │  │
                    │  │   • Forecasting (implemented)    │  │
                    │  │   • Classification (planned)     │  │
                    │  │   • Anomaly detection (planned)  │  │
                    │  │   • Imputation (planned)         │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   Model Layer                    │  │
                    │  │   • Temporal: LSTM, PatchTST,    │  │
                    │  │     DLinear, iTransformer, ...   │  │
                    │  │   • Entity-aware wrappers        │  │
                    │  │   • Spatial: GNN modules         │  │
                    │  │   • LLM-based: TimeLLM, GPT4TS  │  │
                    │  │   • Foundation: Chronos, Moirai  │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   Data Layer                     │  │
                    │  │   • YAML manifests + provenance  │  │
                    │  │   • Graph topology support       │  │
                    │  │   • Multi-modal data loaders     │  │
                    │  │   • Entity metadata management   │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   Runtime Layer                  │  │
                    │  │   • State machine experiment     │  │
                    │  │   • Ray Tune HPO                 │  │
                    │  │   • Slurm cluster dispatch       │  │
                    │  │   • MLflow/W&B logging           │  │
                    │  └─────────────────────────────────┘  │
                    │                                       │
                    │  ┌─────────────────────────────────┐  │
                    │  │   Plugin Layer (vertical domains)│  │
                    │  │   • Smart Earth / Hydrology      │  │
                    │  │   • Healthcare                   │  │
                    │  │   • Finance                      │  │
                    │  │   • Smart Engineering            │  │
                    │  │   • Computational Chemistry      │  │
                    │  └─────────────────────────────────┘  │
                    └───────────────────────────────────────┘
```

### 3.2 Implemented Entity Identifier System

- **Modes**: `none`, `embedding`, `onehot`, `coordinates`, `sinusoidal`, `random`, `numeric_id`, `descriptors`
- **Wrappers**: `EntityWrapper` (per-entity), `ChannelEntityWrapper` (learned per-channel), `ChannelTransparentWrapper` (pre-computed per-channel)
- **Experiment orchestration**: 48-job matrix runner, Ray Tune HPO, Slurm dispatch, comparison reporting

### 3.3 Implemented Spatial Modules

- Entity embeddings with configurable injection points
- 1-hop neighbor aggregation
- GNN modules (integrated with ICPR 2026 submission on Swiss river water temperature prediction)

---

## 4. Research Program: Multi-Period Plan

### Phase A: Entity-Aware Forecasting (Months 1–5, 3 Papers)

---

### Period 1: Systematic Entity Identifier Ablation Study (Month 1)

**Paper Title**: "When and How Do Entity Identifiers Help? A Systematic Study Across Architectures for Multi-Entity Time Series Forecasting"

**Target Venue**: NeurIPS 2026 D&B / AAAI 2027

#### Objectives
1. First systematic comparison of 6+ identifier types across 6+ architecture families on 7+ datasets
2. Empirical analysis of when identifiers help (dataset characteristics, entity count, correlation structure)
3. Practical guidelines for identifier selection
4. Open-source benchmark framework (LIULIAN)

#### Experiment Design

**Phase 1A — Core ablation (Week 1–2)**:

| Dimension | Values |
|---|---|
| Datasets | swiss-river-1990, traffic, electricity, PEMS04, PEMS08 |
| Models | LSTM, PatchTST, DLinear |
| Identifier modes | none, embedding, onehot, sinusoidal, random, coordinates (Swiss only) |
| Seeds | 3 per combination |
| Prediction lengths | 96, 192, 336, 720 |
| Metrics | MSE, MAE (primary); RMSE, NSE (secondary) |

~900 runs total.

**Phase 1B — Extended architectures (Week 2–3)**:
Add: iTransformer, TimeMixer, TimeXer.
Add negative controls: Weather, ETTh1 (heterogeneous channels — identifiers should NOT help).

**Phase 1C — Analysis and writing (Week 3–4)**:
- Wilcoxon signed-rank tests
- t-SNE/UMAP of learned embeddings colored by geography
- Embedding similarity vs spatial distance analysis
- Dataset characterization: inter-entity MI, entity count regime analysis

#### Expected Key Findings
- Learned embeddings dominate on large homogeneous datasets (traffic N=862, electricity N=321)
- Random baselines perform surprisingly well → **entity distinction >> representation quality**
- One-hot degrades with large N (very sparse at 862 channels)
- Negative controls (Weather, ETT) show no benefit → channels are heterogeneous variables, not entities

#### Novelty Verification
- **Systematic identifier comparison across architectures**: NOT published. STID [9] only tests spatial identity embeddings. CARD [5] only tests channel tokens. iTransformer [4] only tests variate-as-token. No work systematically compares none/embedding/onehot/sinusoidal/random/coordinates across LSTM/Transformer/Linear families. **NOVEL.**
- **Random embedding baseline**: Used in NLP (random word embeddings) but never as a controlled ablation for entity identifiers in TS. **NOVEL for TS.**
- **ChannelTransparentWrapper**: No prior work injects pre-computed features in multi-channel mode for TSLib models. **NOVEL architecture.**
- **Negative control datasets**: Previous works don't distinguish homogeneous-entity vs heterogeneous-feature datasets for identifier analysis. **NOVEL framing.**

#### Contributions
- C1: First comprehensive entity identifier benchmark across architectures
- C2: Empirical guidelines for identifier selection based on dataset properties
- C3: Open-source LIULIAN entity identifier framework
- C4: Novel Swiss River hydrological dataset as evaluation domain

---

### Period 2: Adaptive Entity Identifier Selection via Meta-Learning (Month 2–3)

**Paper Title**: "AdaptID: Adaptive Entity Identifier Selection for Multi-Entity Time Series Forecasting via Meta-Learning"

**Target Venue**: ICML 2027 / NeurIPS 2027

#### Technical Approach

**Multi-View Entity Representation with Gating**:
```
x_entity = Gate([emb(id), onehot(id), sinusoidal(id), random(id), coords(id)])
```

- **Gate**: Attention mechanism over identifier views, conditioned on dataset statistics (N_entities, inter-entity MI, temporal autocorrelation)
- **Training**: End-to-end with forecasting objective + auxiliary identifier discrimination loss
- **Architecture**: Identifier Encoder → Dataset Profiler → Gating Network → Forecasting Backbone

#### Novelty Verification
- **TESTAM** [22] uses MoE to gate over graph structure types (none, static, dynamic), NOT identifier types. Different input and granularity.
- **TRA** [13] does entity-specific routing for temporal patterns in stocks, NOT identifier type selection.
- **CARD** [5] uses fixed channel tokens, no adaptive selection.
- **Gating over multiple identifier representations conditioned on dataset statistics**: **NOT PUBLISHED. NOVEL.**

#### Contributions
- C1: First adaptive identifier selection framework for time series
- C2: Meta-learning formulation enabling cross-dataset transfer of selection strategy
- C3: Theoretical analysis connecting entity correlation structure ↔ optimal identifier type
- C4: Identifier fusion consistently outperforms best single identifier

---

### Period 3: Unified Entity Identifiers + Spatio-Temporal Graphs (Month 3–5)

**Paper Title**: "Beyond Node Embeddings: Bridging Entity Identifiers and Graph Neural Networks for Spatio-Temporal Forecasting"

**Target Venue**: KDD 2027 / AAAI 2027

#### Technical Approach

**Key Insight**: Entity identifiers (Period 1–2) are a special case of GNN node features without edges. This work unifies them:

1. **Identifier-to-Graph**: Learn adaptive adjacency from entity representations (cf. Graph WaveNet [15], MTGNN [16])
2. **Graph-to-Identifier**: Extract entity-level features from graph structure as identifier
3. **Hybrid**: Identifiers + adaptive graph + message passing

**Entity Relationship Construction**:
- From data: Pairwise entity similarity from temporal patterns
- From metadata: Geographic coordinates, network topology
- From LLMs: Generate entity relationship descriptions → graph structure (**novel**)

**Zero-Shot Entity Prediction** (STEP-inspired [17]):
- Pre-train entity embeddings on source entities
- Transfer via spatial interpolation, LLM-generated description mapping, or few-shot adaptation

#### Novelty Verification
- **STID** [9] shows identifiers can replace graphs, but doesn't study the combination.
- **STEP** [17] does pre-training for zero-shot nodes, but within same domain only.
- **LLM-based entity graph construction from metadata descriptions**: **NOT PUBLISHED. NOVEL.** (Closest: LLM-based knowledge graphs in NLP, but not applied to ST sensor networks.)
- **Unified framework connecting entity identifiers and GNN node embeddings**: **NOT PUBLISHED as a systematic study. NOVEL.**

---

### Phase B: Spatio-Temporal Foundation Models (Months 5–12, 3 Papers)

---

### Period 4: LLM-Enhanced Spatio-Temporal Intelligence (Month 5–8)

**Paper Title**: "EntityLLM: Large Language Model-Enhanced Entity Representation for Universal Spatio-Temporal Forecasting"

**Target Venue**: ICLR 2028 / NeurIPS 2027

#### Technical Approach

**Text-Enhanced Entity Identifiers**:
```
Entity: "Station Aare-Bern: river gauging station on the Aare river,
         coordinates (46.95, 7.45), elevation 502m, catchment area 3000 km²,
         upstream of Lake Thun, measures water temperature hourly."
→ LLM encoder → entity_text_emb → align with entity_learned_emb
```

**Multi-Modal Entity Representation**:
- Combine: TS patterns + text descriptions + spatial coordinates + satellite imagery (for hydrology)
- Attention-based cross-modal fusion
- Each modality provides complementary entity information

**LLM Agent for ST Forecasting Pipeline**:
- Analyzes dataset metadata → recommends identifier type, architecture, graph structure
- Iterative: run → analyze → refine
- Natural language entity specification for zero-shot adaptation

#### Novelty Verification
- **Time-LLM** [43]: Reprograms LLMs for TS but without entity-specific text.
- **TEST** [44]: Aligns TS with text prototypes but not entity metadata descriptions.
- **UrbanGPT** [52]: Uses LLMs for ST but doesn't use entity metadata as text prompts.
- **Text descriptions of sensor entities as LLM prompts for identifier injection**: **NOT PUBLISHED. NOVEL.**
- **LLM agent for automated ST pipeline design**: **NOT PUBLISHED.** TrafficGPT [54] is closest but limited to traffic + ChatGPT wrapper, not pipeline design automation. DS-Agent [58] and AI Scientist [57] are domain-general. **NOVEL for ST domain.**
- **Multi-modal entity fusion (TS + text + coordinates + satellite)**: **NOT PUBLISHED in this combination. NOVEL.**

---

### Period 5: Spatio-Temporal Foundation Model with Entity Awareness (Month 7–10)

**Paper Title**: "UniEntity: An Entity-Aware Foundation Model for Cross-Domain Spatio-Temporal Forecasting"

**Target Venue**: ICML 2028 / Nature Machine Intelligence

#### Technical Approach

**Domain-Agnostic Entity Framework**:
- Entity type ontology: sensor, patient, financial instrument, weather station, river gauge, ...
- Shared entity embedding space across domains
- Domain-specific entity adapters (LoRA-style [110])

**Pre-Training Strategy**:
- LOTSA-style [35] multi-domain ST corpus: traffic + electricity + hydrology + air quality + clinical
- Entity-conditioned masked modeling: predict masked patches conditioned on entity identity
- Spatial-aware attention: combine temporal self-attention with learned spatial attention

**Cross-Domain Entity Transfer**:
- Demonstrate: traffic sensor embeddings transfer to river gauging station embeddings (both are spatial sensor networks with periodic patterns)
- Patient-as-entity: ICU time series with patient demographics as entity metadata

| Domain | Entity Type | Entity Metadata | Key Challenge |
|---|---|---|---|
| Transportation | Traffic sensors | Location, road type, speed limit | Periodic patterns, spatial correlation |
| Energy | Power meters | Customer type, location, capacity | Diverse consumption, peak demand |
| Hydrology | River gauging stations | Coordinates, catchment area, elevation | Network topology, extreme events |
| Healthcare | ICU patients | Demographics, diagnoses, medications | High variability, missing data, irregular sampling |
| IoT | Building sensors | Floor, room, sensor type | Heterogeneous types, edge deployment |
| Finance | Stocks/instruments | Sector, market cap, country | Non-stationary, regime changes |
| Environment | Air quality monitors | Location, proximity to roads/industry | Spatial dispersion, meteorological coupling |

#### Novelty Verification
- **Moirai** [35]: Any-variate but no entity conditioning or spatial awareness.
- **UniST** [31]: ST foundation model but traffic/urban-only.
- **OpenCity** [32]: ST pre-training but traffic-only.
- **Entity-conditioned cross-domain ST foundation model**: **NOT PUBLISHED. NOVEL.** Closest: UniTS [12] uses dataset-specific prompt tokens but not entity identity conditioning.
- **LoRA-style domain adapters for ST foundation model**: **NOT PUBLISHED in ST context. NOVEL.**

---

### Period 6: Mamba/SSM Architectures for Efficient Spatio-Temporal Learning (Month 9–12)

**Paper Title**: "ST-Mamba: Efficient Spatio-Temporal Forecasting with Selective State Space Models and Entity-Aware Conditioning"

**Target Venue**: ICLR 2028 / ICML 2028

#### Technical Approach

Building on SpoT-Mamba [28] and STG-Mamba [29], develop a full ST-Mamba architecture with:

1. **Spatial Mamba**: Selective scan along spatial neighborhoods (graph-guided walk sequences)
2. **Temporal Mamba**: Selective scan along time dimension with variable-length context
3. **Entity conditioning**: Inject entity identifiers into Mamba's selection mechanism (∆, B, C matrices)
4. **Linear complexity**: O(N·T) vs O(N²·T²) for full attention — critical for scaling to 1000+ entity networks

#### Novelty Verification
- **SpoT-Mamba** [28]: Uses Mamba for ST but in a workshop paper, limited scope.
- **STG-Mamba** [29]: ArXiv preprint, first exploration but no entity conditioning.
- **Entity-conditioned Mamba with full spatial+temporal scanning**: **NOT PUBLISHED. NOVEL.**
- **Systematic comparison of Mamba vs Transformer vs GNN for ST forecasting**: **NOT PUBLISHED as a systematic study. NOVEL.**

---

### Phase C: LIULIAN Platform and Domain Applications (Months 8–18)

---

### Period 7: LIULIAN Platform System Paper (Month 8–14)

**Paper Title**: "LIULIAN: An Open-Source Agent-Powered Platform for Spatio-Temporal Intelligence with Vertical Domain Plugins"

**Target Venue**: VLDB 2028 / KDD 2028 (Demo) / JMLR MLOSS / NeurIPS 2027 D&B

#### The Platform as Contribution

LIULIAN is not just a model library — it is a **full-stack spatio-temporal intelligence platform** combining:

1. **Modular ML Pipeline**:
   - Task-driven design separating task semantics from model logic
   - ExecutableModel ABC with lightweight adapter pattern
   - State machine experiment orchestration (train → eval → infer → analyze → visualize)
   - YAML manifests with semantic schemas, topology, integrity hashing

2. **Entity-Aware and Spatial Modules** (from Phases A–B):
   - Entity identifier injection across all model families
   - Adaptive graph construction
   - GNN spatial modules
   - Cross-domain entity transfer

3. **LLM Agent Layer**:
   - **Pipeline Design Agent**: Analyzes dataset metadata, recommends models/identifiers/graphs
   - **Data Ingestion Agent**: Natural language-based adaptive data mapping ("one-click graph ingestion")
   - **Report Generation Agent**: Automated analysis reports from experiment results
   - **Human-in-the-loop interaction**: Domain experts describe problems in NL, system recommends solutions

4. **BI and Visualization Layer**:
   - Interactive dashboards (Power BI integration + custom extensions)
   - Natural language query interface for spatio-temporal data
   - Automated report generation with charts and insights
   - Graph explainability visualizations

5. **Vertical Domain Plugins**:

| Domain | Plugin | Status | Key Features |
|---|---|---|---|
| **Smart Earth / Hydrology** | `plugins/hydrology/` | Partially implemented | Swiss river database, station coordinates, watershed topology, GIS visualization |
| **Healthcare** | `plugins/healthcare/` | Planned | MIMIC-III/IV integration, patient-as-entity, perioperative prediction |
| **Finance** | `plugins/finance/` | Planned (N-Banker collaboration) | Financial TS analysis, entity-aware stock prediction, AI agent chatbot |
| **Smart Engineering** | `plugins/engineering/` | Planned (DeepSchemata) | Topology extraction from drawings, automated analysis |
| **Computational Chemistry** | `plugins/chemistry/` | Planned | Molecular property prediction, reaction analysis |
| **Disaster Management** | `plugins/disaster/` | Planned (Renmin Univ. collaboration) | Multi-modal (satellite + population), evacuation routing |
| **Digital Humanities** | `plugins/humanities/` | Planned | Document structure analysis, handwriting recognition |

6. **Deployment and MLOps**:
   - Docker containerization
   - Cloud deployment (AWS planned)
   - CI/CD via GitHub Actions
   - Ray Tune + Slurm integration
   - Experiment tracking (MLflow/W&B)

#### Novelty Verification (Platform)
- **TSLib** [73], **BasicTS** [74], **tsl** [75]: Model libraries only — no agents, no BI, no plugins, no entity awareness.
- **LibCity** [79]: Urban computing only — no cross-domain, no agents, no BI.
- **GluonTS** [3], **NeuralForecast** [76]: Temporal only — no spatial, no graphs, no entities.
- **Ludwig** [111]: Declarative ML but not ST-specific.
- **Full-stack ST platform with entity-aware models + LLM agents + BI + vertical domain plugins**: **NOT PUBLISHED. NOVEL.**

---

### Domain Application Papers (Months 10–18, 3+ Papers)

These papers apply LIULIAN to specific domains, each producing a domain contribution:

#### D1: Swiss River Water Temperature Prediction (Month 10–12)
- Extends ICPR 2026 submission with entity identifier ablation + graph modules
- Domain-specific evaluation with NSE metric and hydrological baselines
- **Target**: Environmental Data Science / Water Resources Research

#### D2: Perioperative Patient Outcome Prediction (Month 12–15)
- Collaboration with China Pharmaceutical University and Inselspital Bern
- Patient-as-entity with demographics, surgical type, comorbidities as entity metadata
- Entity clustering reveals patient phenotypes with distinct outcome trajectories
- **Target**: npj Digital Medicine / CHIL / ML4H

#### D3: Disaster Impact Analysis from Multi-Modal ST Data (Month 14–18)
- Collaboration with Renmin University of China
- Multi-modal: satellite imagery + population distribution + time series
- Tasks: population distribution prediction, damage assessment, evacuation planning
- **Target**: Nature Communications / KDD Applied Data Science

---

## 5. Innovation Highlights and Novelty Summary

### 5.1 Near-Term Innovations (Phase A, Periods 1–3)

| Innovation | Novelty Status | Closest Prior Work | What's New |
|---|---|---|---|
| Systematic identifier ablation across architectures | **Novel** | STID [9] (one type only) | First multi-type, multi-architecture comparison |
| Random embedding baseline for entity distinction | **Novel for TS** | Random word embeddings in NLP | Isolates "distinction" from "representation quality" |
| ChannelTransparentWrapper | **Novel architecture** | None | Pre-computed features in multi-channel TSLib models |
| Negative control datasets for identifier study | **Novel framing** | None | First to explicitly test where IDs should NOT help |
| AdaptID meta-learning over identifier types | **Novel** | TESTAM [22] (graph types) | Gating over identifier types, not graph structures |
| LLM-based entity graph construction | **Novel** | LLM knowledge graphs in NLP | Applied to ST sensor networks from metadata |
| Unified identifier + GNN framework | **Novel** | STID [9], Graph WaveNet [15] separately | Systematic bridge between two research communities |

### 5.2 Medium-Term Innovations (Phase B, Periods 4–6)

| Innovation | Novelty Status | Closest Prior Work | What's New |
|---|---|---|---|
| Text-enhanced entity identifiers via LLM | **Novel** | TEST [44] (text prototypes) | Entity-specific metadata descriptions as identifiers |
| LLM agent for ST pipeline design | **Novel** | TrafficGPT [54], AI Scientist [57] | Domain-specific ST pipeline automation |
| Multi-modal entity fusion (TS+text+coords+satellite) | **Novel** | Time-MMD [117] (TS+text only) | Full multi-modal fusion for entity representation |
| Entity-aware cross-domain ST foundation model | **Novel** | UniST [31] (urban only), Moirai [35] (no entities) | Entity conditioning + cross-domain pre-training |
| Entity-conditioned Mamba for ST | **Novel** | SpoT-Mamba [28] (no entity conditioning) | Entity identifiers in SSM selection mechanism |

### 5.3 Long-Term Innovations (Phase C, Period 7+)

| Innovation | Novelty Status | Closest Prior Work | What's New |
|---|---|---|---|
| Full-stack ST intelligence platform | **Novel** | TSLib [73], LibCity [79] (model libraries only) | Agent + BI + entity-aware + plugin architecture |
| LLM-driven adaptive data ingestion for ST data | **Novel** | Data-Copilot [55] (tabular only) | NL-based ST data mapping with graph topology |
| Vertical domain plugins with domain-specific GUIs | **Novel for ST** | Hugging Face spaces (NLP) | Domain-specific GUIs for hydrology, healthcare, etc. |
| Perioperative patient clustering via entity-aware ST models | **Novel application** | Perioperative GNN [84] (single task) | Entity identifier framework applied to clinical TS |

---

## 6. Evaluation Protocol

### 6.1 Datasets

**Primary — Homogeneous multi-entity (Phases A–B)**:

| Dataset | Entities | Length | Domain | Source |
|---|---|---|---|---|
| Traffic | 862 | 17,544 | Road sensors | TSLib / Caltrans PEMS |
| Electricity | 321 | 26,304 | Power clients | TSLib / UCI |
| Swiss-River-1990 | ~50 | ~10,000 | Gauging stations | LIULIAN (novel) |
| PEMS03/04/07/08 | 170–883 | 16,992–26,208 | Traffic sensors | STGNN benchmark |
| Solar-Energy | 137 | ~52,000 | PV plants | NREL |
| METR-LA | 207 | ~34,272 | Traffic speed | DCRNN benchmark |
| PEMS-BAY | 325 | ~52,116 | Traffic speed | DCRNN benchmark |

**Negative controls — Heterogeneous channels**:

| Dataset | Channels | Why Negative | Source |
|---|---|---|---|
| Weather | 21 | Different physical variables | TSLib / Max Planck |
| ETTh1/ETTm1 | 7 | Heterogeneous measurements | TSLib |
| ILI | 7 | Different statistical measures | CDC |

**Extended domains (Phases B–C)**:

| Dataset | Domain | Entity Type | Source |
|---|---|---|---|
| MIMIC-III/IV | ICU patients | Patient | PhysioNet |
| Beijing Air Quality | Environmental | Monitoring station | UCI |
| NYC Taxi/Bike | Urban mobility | Zone/station | NYC TLC / Citi Bike |
| London Smart Meters | Energy | Household | UK Power Networks |
| Perioperative (private) | Healthcare | Patient | China Pharm. Univ. / Inselspital |

### 6.2 Metrics

- **Primary**: MSE, MAE (standard TSLib protocol)
- **Secondary**: RMSE, MAPE, NSE (hydrology-specific)
- **Statistical**: Mean ± std over 3+ seeds; Wilcoxon signed-rank test; Nemenyi CD diagrams
- **Probabilistic** (when applicable): CRPS, calibration, interval coverage

### 6.3 Baselines

| Category | Methods |
|---|---|
| No identifier | Shared model, all architectures |
| Each identifier type | embedding, onehot, sinusoidal, random, coordinates |
| Implicit entity | iTransformer [4] (variate-as-token) |
| Explicit entity (simple) | STID [9] (MLP + spatial ID) |
| Channel tokens | CARD [5] |
| Graph-based | Graph WaveNet [15], MTGNN [16], DCRNN [14] |
| Foundation models | Chronos [34], Moirai [35], TimesFM [33] |
| LLM-based | Time-LLM [43], GPT4TS [42] |

---

## 7. Resource Requirements

| Resource | Phase A (Mo 1–5) | Phase B (Mo 5–12) | Phase C (Mo 8–18) |
|---|---|---|---|
| GPU hours | ~1,500 | ~5,000 | ~3,000 |
| Storage | 100 GB | 500 GB | 1 TB |
| Human effort | 5 months (AI-assisted) | 7 months | 10 months |
| Compute | RTX 4090 cluster | A100 cluster | A100/H100 cluster |
| LLM API costs | Minimal | ~$500/month | ~$1,000/month |

---

## 8. Risk Analysis and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Entity identifiers show minimal benefit on most datasets | Medium | High | Frame as negative result; focus on "when" not "whether" |
| Random baselines match learned embeddings | Medium | Medium | Important finding — entity distinction >> representation quality |
| LLM entity descriptions don't improve over numerical embeddings | Medium | Medium | Focus on zero-shot/transfer scenarios where text is uniquely valuable |
| ST foundation model can't generalize across domains | Medium | High | Start with similar domains (traffic → hydrology); use domain adapters |
| Platform development takes longer than expected | High | Medium | Prioritize research papers; platform is incremental |
| Collaborator datasets not available on time | Medium | Medium | Use public datasets as fallback; domain papers shift timeline |
| Tan et al. [51] criticism applies to our LLM-based methods | Medium | Medium | Design controlled ablations; focus on entity-specific contributions |

---

## 9. Collaboration Network

| Partner | Domain | Status | Role |
|---|---|---|---|
| University of Bern (PRG) | Hydrology, graph ML | Active | Swiss river data, graph matching |
| BFH (Bern Univ. Applied Sciences) | Hydrology, mudslide detection | Active | Domain expertise, private data |
| China Pharmaceutical University | Healthcare (perioperative) | Under connection | Clinical datasets, domain expertise |
| Inselspital Bern | Healthcare (perioperative) | Building connection | Clinical data, validation |
| Renmin University of China | Disaster management | Built connection | Multi-modal earthquake datasets |
| Digital Financial Services Research Center | Finance (N-Banker) | Strong connection | Financial AI agent system |
| HES-SO Fribourg | Document analysis, engineering | Previous institution | Historical docs, engineering drawings |
| Chinese Academy of Sciences | Materials science | Built connection | Molten pool dynamics data |

---

## 10. Detailed Period 1 Execution Plan (1 Month)

### Week 1: Infrastructure and Core Experiments
- [ ] Verify all 48 matrix jobs pass dry-run
- [ ] Add iTransformer, TimeMixer, TimeXer to experiment matrix
- [ ] Add PEMS04, PEMS08, and negative control datasets
- [ ] Increase default seeds to 3 (2024, 2025, 2026)
- [ ] Set up prediction length sweep (96, 192, 336, 720)
- [ ] Submit Phase 1A core matrix to cluster
- [ ] Monitor and fix pipeline issues

### Week 2: Full Experimental Campaign
- [ ] Verify Phase 1A results; re-run failures
- [ ] Submit extended model experiments
- [ ] Submit negative controls (Weather, ETTh1 × all models × none+embedding)
- [ ] Compute inter-entity correlation matrices
- [ ] Run t-SNE/UMAP visualization of learned embeddings

### Week 3: Analysis and Figures
- [ ] Compute mean ± std across seeds
- [ ] Wilcoxon signed-rank tests: each mode vs none
- [ ] Performance improvement heatmaps (dataset × model × mode)
- [ ] Effect of entity count on identifier benefit
- [ ] Identifier type recommendation flowchart

### Week 4: Paper Writing
- [ ] Introduction positioning vs PatchTST CI debate
- [ ] Related work (comprehensive, from Section 2)
- [ ] Methodology (LIULIAN framework, identifier types, injection mechanisms)
- [ ] Results with tables, figures, ablation studies
- [ ] Practical guidelines section
- [ ] Abstract, conclusion, supplementary
- [ ] Submit to arXiv; prepare venue-specific submission

---

## 11. Summary: Research Contributions Across All Periods

| Period | Title | Type | Novelty | Target Venue | Timeline |
|---|---|---|---|---|---|
| **1** | Entity identifier systematic ablation | Benchmark | **High** | NeurIPS D&B / AAAI | Month 1 |
| **2** | AdaptID: Adaptive identifier selection | Method | **Very High** | ICML / NeurIPS | Month 2–3 |
| **3** | Unified identifiers + GNN framework | Framework | **High** | KDD / AAAI | Month 3–5 |
| **4** | EntityLLM: LLM-enhanced entity representation | Vision+Method | **Very High** | ICLR / NeurIPS | Month 5–8 |
| **5** | UniEntity: Entity-aware ST foundation model | Foundation | **Extremely High** | ICML / Nature MI | Month 7–10 |
| **6** | ST-Mamba with entity conditioning | Method | **High** | ICLR / ICML | Month 9–12 |
| **7** | LIULIAN platform system paper | System | **Very High** | VLDB / KDD Demo | Month 8–14 |
| **D1** | Swiss river water temperature | Domain | Medium-High | Env. Data Science | Month 10–12 |
| **D2** | Perioperative patient prediction | Domain | High | npj Digital Med. | Month 12–15 |
| **D3** | Disaster multi-modal ST analysis | Domain | High | Nature Comms. / KDD | Month 14–18 |

---

## 12. Long-Term Vision (Year 2+)

Beyond the core 18-month program, LIULIAN aims to become the **standard open-source platform for spatio-temporal intelligence**, analogous to what Hugging Face Transformers is for NLP:

1. **Community-driven model hub**: Researchers contribute ST models via the adapter pattern
2. **Benchmark-as-a-service**: Standardized evaluation across domains (addressing the TSLib [73] / TFB [120] critique)
3. **Agent marketplace**: Domain-specific LLM agents for different verticals
4. **Innosuisse Innovation Project**: Funding application for industry deployment
5. **Educational tools**: Interactive tutorials for spatio-temporal ML (MkDocs-based, already partially implemented)

---

## References

[1] Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
[2] Fey, M. & Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." ICLR 2019 Workshop.
[3] Alexandrov, A., et al. "GluonTS: Probabilistic and Neural Time Series Modeling in Python." JMLR 2020.
[4] Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024.
[5] Wang, X., et al. "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting." ICLR 2024.
[6] Zhang, Y. & Yan, J. "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting." ICLR 2023.
[7] Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." NeurIPS 2023.
[8] Gao, S., et al. "Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting." KDD 2024.
[9] Shao, Z., et al. "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting." CIKM 2022.
[10] Das, A., et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder." TMLR 2023.
[11] Wang, Y., et al. "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables." NeurIPS 2024.
[12] Gao, S., et al. "UniTS: Building a Unified Time Series Model." NeurIPS 2024.
[13] Lin, H., et al. "TRA: Temporal Routing Adaptor for Entity-Aware Stock Forecasting." AAAI 2023.
[14] Li, Y., et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018.
[15] Wu, Z., et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019.
[16] Wu, Z., et al. "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." KDD 2020.
[17] Shao, Z., et al. "Pre-training Enhanced Spatial-Temporal Graph Neural Network for Multivariate Time Series Forecasting." KDD 2022.
[18] Deng, J., et al. "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting." KDD 2021.
[19] Jiang, R., et al. "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting." AAAI 2023.
[20] Jiang, J., et al. "PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction." AAAI 2023.
[21] Liu, H., et al. "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting." CIKM 2023.
[22] Lee, H., et al. "TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts." ICLR 2024.
[23] Cini, A., et al. "Scalable Spatiotemporal Graph Neural Networks." AAAI 2023.
[24] Han, J., et al. "BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks." VLDB 2024.
[25] Tang, J., et al. "EasyST: A Simple Framework for Spatio-Temporal Prediction." arXiv 2024.
[26] Liang, Y., et al. "AirFormer: Predicting Nationwide Air Quality in China with Transformers." AAAI 2023.
[27] Gao, Z., et al. "EarthFormer: Exploring Space-Time Transformers for Earth System Forecasting." NeurIPS 2022.
[28] Choi, J., et al. "SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces." IJCAI 2024 Workshop.
[29] Li, L., et al. "STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model." arXiv 2024.
[30] Li, Z., et al. "GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks." NeurIPS 2023.
[31] Yuan, Y., et al. "UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction." KDD 2024.
[32] Li, Z., et al. "OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction." ACM TIST 2024.
[33] Das, A., et al. "A Decoder-Only Foundation Model for Time-Series Forecasting." ICML 2024.
[34] Ansari, A., et al. "Chronos: Learning the Language of Time Series." ICML 2024.
[35] Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024.
[36] Liu, Y., et al. "Timer: Generative Pre-trained Transformers Are Large Time Series Models." ICML 2024.
[37] Goswami, M., et al. "MOMENT: A Family of Open Time-Series Foundation Models." ICML 2024.
[38] Rasul, K., et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." ICML 2024.
[39] Ekambaram, V., et al. "TTM: Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting." NeurIPS 2024.
[40] Shi, X., et al. "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." arXiv 2024.
[41] Garza, A. & Mergenthaler-Canseco, M. "TimeGPT-1." arXiv 2023.
[42] Zhou, T., et al. "One Fits All: Power General Time Series Analysis by Pretrained LM." NeurIPS 2023.
[43] Jin, M., et al. "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." ICLR 2024.
[44] Sun, C., et al. "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series." ICLR 2024.
[45] Liu, X., et al. "UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting." WWW 2024.
[46] Liu, C., et al. "CALF: Aligning LLMs for Time Series Forecasting via Cross-Modal Fine-Tuning." NeurIPS 2024.
[47] Pan, S., et al. "S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting." ICML 2024.
[48] Gruver, N., et al. "Large Language Models Are Zero-Shot Time Series Forecasters." NeurIPS 2023.
[49] Cao, D., et al. "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting." ICLR 2024.
[50] Liu, Y., et al. "AutoTimes: Autoregressive Time Series Forecasting via Large Language Models." NeurIPS 2024.
[51] Tan, M., et al. "Are Language Models Actually Useful for Time Series Forecasting?" NeurIPS 2024.
[52] Li, Z., et al. "UrbanGPT: Spatio-Temporal Large Language Models." KDD 2024.
[53] Cao, D., et al. "ST-LLM: Large Language Models Are Effective Temporal Learners for Spatio-Temporal Forecasting." arXiv 2024.
[54] Zhang, S., et al. "TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models." Transport Policy 2024.
[55] Zhang, W., et al. "Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow." arXiv 2023.
[56] Ma, P., et al. "InsightPilot: An LLM-Empowered Automated Data Exploration System." VLDB 2024.
[57] Lu, C., et al. "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." arXiv 2024.
[58] Guo, S., et al. "DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning." ICML 2024.
[59] Huang, Q., et al. "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation." ICML 2024.
[60] Zhang, S., et al. "AutoML-GPT: Automatic Machine Learning with GPT." arXiv 2023.
[61] Li, Z., et al. "MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization." ACL 2024.
[62] Baek, J., et al. "ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models." ACL 2024.
[63] Bran, A.M., et al. "ChemCrow: Augmenting Large Language Models with Chemistry Tools." Nature Machine Intelligence 2024.
[64] Boiko, D.A., et al. "An Autonomous Agent for Scientific Discovery." Nature 2023.
[65] Dibia, V. "LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models." ACL 2023.
[66] Pourreza, M. & Rafiei, D. "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction." NeurIPS 2023.
[67] Gao, D., et al. "DAIL-SQL: Efficient and Effective Few-Shot Text-to-SQL." VLDB 2024.
[68] Narechania, A., et al. "NL4DV: A Toolkit for Generating Analytic Specifications from Natural Language." IEEE VIS 2023.
[69] Maddigan, P. & Susnjak, T. "Chat2VIS: Generating Data Visualizations via Natural Language." IEEE Access 2023.
[70] Li, Z., et al. "TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT." arXiv 2023.
[71] Yu, K., et al. "Rath: Augmented Analytics with Automated Insight Discovery." GitHub / Kanaries 2022–2024.
[72] Chen, Y., et al. "Sheet2Report: Automated Report Generation from Spreadsheets using LLMs." arXiv 2024.
[73] Wu, H., et al. "TimesNet / TSLib: A Unified Library for Time Series Analysis." NeurIPS 2023 D&B.
[74] Shao, Z., et al. "BasicTS: An Open-Source Fair Benchmark for Time Series Prediction." NeurIPS 2023 D&B.
[75] Cini, A., et al. "Torch Spatiotemporal (tsl)." JMLR MLOSS 2024.
[76] Olivares, K., et al. "NeuralForecast: User-Friendly State-of-the-Art Neural Forecasting Models." arXiv 2022.
[77] Herzen, J., et al. "Darts: User-Friendly Modern Machine Learning for Time Series." JMLR 2022.
[78] Du, W. "PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series." ICLR 2024.
[79] Wang, J., et al. "LibCity: An Open Library for Traffic Flow Prediction." SIGSPATIAL 2021.
[80] Tipirneni, S. & Reddy, C.K. "STraTS: Self-Supervised Sparse Transformer for Clinical Time Series." MLHC 2022.
[81] Kim, Y., et al. "MedGTX: Multimodal Graph-Transformer for Patient Outcome Prediction." CHIL 2023.
[82] Li, C., et al. "MISTS: Multi-Input Multi-Output Irregular Sampled Time Series Forecasting for ICU." AAAI 2024.
[83] Jiang, Z., et al. "MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis." arXiv 2024.
[84] Gao, Y., et al. "Perioperative Outcome Prediction with Graph Neural Networks." npj Digital Medicine 2024.
[85] Various. "GAGNN: Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting." ACM TKDD 2024.
[86] Xu, Z., et al. "FloodCast: Flood Forecasting with Spatio-Temporal Graph Neural Networks." arXiv 2023.
[87] Various. "Spatio-Temporal Feature Graph Neural Network for River Water Quality." Water Research 2025.
[88] Zanfei, A., et al. "Spatio-Temporal Graph Neural Networks for Water Distribution Network Monitoring." Water Research 2023.
[89] Various. "Multi-scale Spatio-Temporal Graph Neural Network for Water Demand Forecasting." Water Research 2025.
[90] Yang, H., et al. "FinGPT: Open-Source Financial Large Language Models." NeurIPS 2023 Workshop.
[91] Li, Y., et al. "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI 2024.
[92] Jakubik, J., et al. "Prithvi: A Foundation Model for Geospatial AI." arXiv 2023.
[93] Kuckreja, K., et al. "GeoChat: Grounded Large Language Model for Remote Sensing." CVPR 2024.
[94] Klemmer, K., et al. "SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery." NeurIPS 2023 Workshop.
[95] Various. "HeatGNN: Heterogeneous Epidemic-Aware Transmission Graph Neural Network." arXiv 2024.
[96] Balaji, B., et al. "Brick Schema + ML for Smart Building Analytics." BuildSys 2022.
[97] Vazquez-Canteli, J., et al. "CityLearn: A Gym Environment for Building Energy Coordination." Applied Energy 2023.
[98] Yue, Z., et al. "TS2Vec: Towards Universal Representation of Time Series." AAAI 2022.
[99] Zhang, X., et al. "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency." NeurIPS 2022.
[100] Dong, J., et al. "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling." NeurIPS 2023.
[101] Zhang, Q., et al. "AutoST: Automated Spatio-Temporal Graph Contrastive Learning." WWW 2023.
[102] Zhang, Q., et al. "STAG: Spatial-Temporal Graph Learning with Adversarial Contrastive Adaptation." ICML 2023.
[103] Xia, Y., et al. "CaST: Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment." NeurIPS 2023.
[104] Wang, B., et al. "STONE: A Spatio-temporal OOD Learning Framework." KDD 2024.
[105] Ragab, M., et al. "ADATIME: A Benchmarking Suite for Domain Adaptation on Time Series Data." ACM TKDD 2023.
[106] He, X., et al. "RAINCOAT: An Adaptive Framework for Robust Domain Adaptation in Time Series." ICML 2023.
[107] Zhang, J., et al. "STExplainer: A Framework for Explaining Spatio-Temporal Graph Neural Networks." KDD 2023.
[108] Bento, J., et al. "TimeSHAP: Explaining Recurrent Models through Sequence Perturbations." KDD 2021.
[109] Lim, B., et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." IJF 2021.
[110] Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
[111] Molino, P., et al. "Ludwig: A Type-Based Declarative Deep Learning Framework." JMLR 2023.
[112] Contributors, OpenMMLab. "MMEngine: OpenMMLab Foundational Library for Training Deep Learning Models." 2022.
[113] Yadan, O. "Hydra: A Framework for Elegantly Configuring Complex Applications." Facebook Research 2020.
[114] Rossi, E., et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." ICML 2020 Workshop.
[115] Yu, L., et al. "DyGFormer: Towards Better Dynamic Graph Learning." NeurIPS 2023.
[116] You, J., et al. "ROLAND: Graph Learning Framework for Dynamic Social Networks." KDD 2022.
[117] Liu, H., et al. "Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis." NeurIPS 2024 D&B.
[118] Xie, Z., et al. "ChatTS: Aligning Time Series with LLMs via Synthetic Data." arXiv 2024.
[119] Stewart, A., et al. "TorchGeo: Deep Learning with Geospatial Data." SIGSPATIAL 2022.
[120] Qiu, X., et al. "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods." ICML 2024.
[121] Talukder, S., et al. "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis." NeurIPS 2024.
[122] Li, Y., et al. "HITL-TSF: Human-in-the-Loop Time Series Forecasting." arXiv 2024.
[123] Shi, D., et al. "Calliope: Automatic Visual Data Stories from Spreadsheets." IEEE TVCG 2023.
[124] Sultan, M., et al. "DataTales: Investigating the Use of Large Language Models for Authoring Data-Driven Articles." CHI 2024.
[125] Thelen, A., et al. "TwinGen: Generative Digital Twins with Physics-Informed Neural Networks." Nature Machine Intelligence 2023.
[126] Wu, H., et al. "Transolver: A Fast Transformer Solver for PDEs on General Geometries." ICML 2024.
[127] Jiang, R., et al. "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting (MegaCRN)." AAAI 2023.

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
                    │  text ─────── LLM(description) → proj  │
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
              ┌──────────────────────────────────┐
              │       Spatial Module (optional)    │
              │  • 1-hop neighbor aggregation      │
              │  • GNN message passing             │
              │  • Adaptive graph learning          │
              │  • Mamba spatial scan               │
              └──────────┬───────────────────────┘
                         ↓
              ┌──────────────────────────────────┐
              │        Forecasting Model           │
              │  LSTM / PatchTST / DLinear /       │
              │  iTransformer / TimeMixer / Mamba / │
              │  Time-LLM / Chronos / Moirai       │
              └────────────────────────────────────┘
```

## Appendix B: Code Artifacts

| File | Description |
|---|---|
| `experiments/entity_identifier/run.py` | Matrix experiment runner |
| `experiments/entity_identifier/matrix.py` | Matrix definition and job expansion |
| `experiments/entity_identifier/compare.py` | Comparison report generator |
| `experiments/entity_identifier/submit_slurm.py` | Slurm cluster dispatcher |
| `liulian/models/torch/entity_mixin.py` | EntityWrapper, ChannelEntityWrapper, ChannelTransparentWrapper |
| `liulian/data/ts/timeseriesdataset.py` | Entity features and transparent mode data layer |
| `experiments/run.py` | Unified experiment entry point |
| `tests/runtime/test_entity_identifier_pipeline.py` | Pipeline tests (15 tests) |

## Appendix C: Novelty Verification Summary

| # | Proposed Idea | Already Published? | Closest Work | What's New |
|---|---|---|---|---|
| 1 | Systematic multi-type identifier ablation across architectures | **No** | STID [9] | Multi-type, multi-arch systematic study |
| 2 | Random embedding baseline for TS entity identification | **No (for TS)** | NLP random baselines | First controlled ablation in TS domain |
| 3 | ChannelTransparentWrapper architecture | **No** | — | Pre-computed features in multi-channel TSLib |
| 4 | Negative control datasets for identifier analysis | **No** | — | Explicit where-IDs-shouldn't-help testing |
| 5 | AdaptID: gating over identifier types | **No** | TESTAM [22] (graph types) | Gates over identifier types, not graph structures |
| 6 | LLM-based entity graph construction from metadata | **No** | LLM knowledge graphs (NLP) | Applied to ST sensor network topology |
| 7 | Unified identifier + GNN framework | **No** | STID [9], GWN [15] separately | Systematic bridge between communities |
| 8 | Entity text descriptions as LLM prompts | **No** | TEST [44] (prototypes) | Entity-specific metadata, not generic prototypes |
| 9 | LLM agent for ST pipeline design | **No** | TrafficGPT [54], AI Scientist [57] | ST-domain-specific pipeline automation |
| 10 | Multi-modal entity fusion (TS+text+coords+satellite) | **No** | Time-MMD [117] (TS+text) | Full 4-modal fusion for entity representation |
| 11 | Entity-aware cross-domain ST foundation model | **No** | UniST [31] (urban), Moirai [35] (no entity) | Entity conditioning + cross-domain |
| 12 | Entity-conditioned Mamba for ST | **No** | SpoT-Mamba [28] (no entity cond.) | Entity IDs in SSM selection mechanism |
| 13 | Full-stack ST platform (agent+BI+entity+plugins) | **No** | TSLib [73], LibCity [79] | Complete platform with all components |
| 14 | Perioperative patient clustering via entity-aware ST | **No (this framing)** | Perioperative GNN [84] | Entity identifier framework for clinical TS |
| 15 | Domain-specific vertical plugins for ST platform | **No** | HuggingFace Spaces (NLP only) | ST-specific domain plugins |
