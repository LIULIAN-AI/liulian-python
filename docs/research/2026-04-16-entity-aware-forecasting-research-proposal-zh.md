# 实体感知时间序列预测：综合研究提案

_日期：2026-04-16_
_首席研究员：贾林林 (jajupmochi@gmail.com)_
_框架：LIULIAN (liulian-python)_

---

## 摘要

本提案概述了一项多阶段研究计划，旨在系统性地研究**实体标识符在多实体时间序列预测中何时、为何以及如何发挥作用**。从对标识符类型、模型架构和数据集的系统消融研究出发，本计划逐步扩展至自适应标识符选择、时空图集成、大语言模型增强的实体建模、跨域迁移以及基础模型适配。每个研究阶段产出一篇可发表的论文，第一阶段可在一个月内完成。

---

## 1. 问题陈述与研究动机

多实体时间序列预测——同时对多个同质传感器、站点或客户的未来值进行预测——是交通、能源、水文、医疗和物联网等领域的基础问题。一个关键的设计决策是**如何表示实体身份**：共享模型是否应该学习到"传感器42与传感器17的行为不同"？

尽管已有众多方法（学习嵌入、独热编码、地理坐标、基于图的表示），但**目前尚无系统性研究跨架构和数据集比较这些标识符类型**。该领域缺乏：

1. 对"实体标识符何时有效"的原则性回答
2. 跨架构家族注入标识符的统一框架
3. 基于数据集属性选择标识符类型的指导方针
4. 对标识符如何与现代范式（基础模型、大语言模型、图神经网络）交互的理解

本研究计划致力于填补以上四个空白。

---

## 2. 文献综述与相关工作

### 2.1 通道独立与通道混合在时间序列中的争论

多变量时间序列模型是否应独立处理各通道还是允许跨通道交互的争论是本研究的核心。

- **PatchTST**（Nie等，ICLR 2023）：证明了通道独立（CI）处理通常优于通道混合（CM），挑战了跨变量信息总是有益的假设 [1]。
- **iTransformer**（Liu等，ICLR 2024）：反转了Transformer范式，将每个变量作为token处理，通过隐式实体表示学习在Traffic和Electricity数据集上取得了优异结果 [2]。
- **CARD**（Wang等，ICLR 2024）：通道对齐鲁棒混合Transformer，在Transformer骨干网络中添加通道特定token（相当于可学习的实体嵌入），相比纯CI方法显示出提升 [3]。
- **CrossFormer**（Zhang & Yan，ICLR 2023）：使用跨维度注意力机制与可学习的维度/段嵌入，是注意力机制中实体标识符的一种形式 [4]。
- **TSMixer**（Chen等，NeurIPS 2023）：基于MLP的架构，证明了简单的跨变量混合即可有效捕获实体间关系 [5]。

**研究空白**：这些工作各自测试了一种嵌入策略；没有一项系统性地跨架构比较不同标识符类型。

### 2.2 显式实体/节点嵌入

- **STID**（Shao等，CIKM 2022）：证明了简单的MLP加空间身份嵌入即可匹配或超越复杂的时空GNN。直接验证了"可学习实体嵌入"的方法 [6]。
- **TiDE**（Das等，TMLR 2023）：基于MLP的编码器-解码器，使用逐变量的静态协变量（属性嵌入），可直接与基于坐标和可学习嵌入的方法比较 [7]。
- **TimeXer**（Wang等，NeurIPS 2024）：在内生补丁token旁引入外生变量token；外生token设计类似于辅助上下文中的实体嵌入 [8]。
- **UniTS**（Gao等，NeurIPS 2024）：跨多个数据集/领域的统一模型，使用数据集特定的提示token——更高层次上的实体/领域标识符 [9]。

### 2.3 时空图神经网络

实体标识符与时空GNN中的节点嵌入密切相关：

- **DCRNN**（Li等，ICLR 2018）：扩散卷积循环神经网络，将交通建模为有向图；节点即具有空间关系的实体 [10]。
- **Graph WaveNet**（Wu等，IJCAI 2019）：学习自适应邻接矩阵，有效地从数据中构建实体关系 [11]。
- **STEP**（Shao等，ICML 2022 workshop）：预训练时空嵌入，通过实体表示的迁移实现对未见节点的零样本预测 [12]。
- **STNorm**（Deng等，AAAI 2021）：交通预测中的空间和时间归一化；实体特定归一化作为显式标识符的替代 [13]。
- **MegaCRN**（Jiang等，AAAI 2023）：元图卷积循环网络，通过元学习学习节点特定模式 [14]。
- **PDFormer**（Jiang等，AAAI 2023）：传播延迟感知的动态长程Transformer，将空间传播模式作为实体关系 [15]。
- **STAEformer**（Liu等，CIKM 2023）：时空自适应嵌入Transformer，学习每个节点的联合时空嵌入 [16]。

**研究空白**：GNN文献将实体嵌入视为节点特征，但未比较不同嵌入策略，也未研究何时更简单的标识符（独热编码、随机编码）即可满足需求。

### 2.4 基于大语言模型的时间序列预测

大语言模型为实体感知预测带来了新机遇：

- **GPT4TS / One Fits All**（Zhou等，NeurIPS 2023）：冻结GPT-2骨干网络并适配时间序列任务；实体上下文可以文本提示形式注入 [17]。
- **Time-LLM**（Jin等，ICLR 2024）：将时间序列重编程为文本原型供LLM推理；实体描述可作为自然语言上下文 [18]。
- **TEST**（Sun等，ICLR 2024）：文本原型对齐嵌入用于时间序列；将时间序列表示与实体的文本描述对齐 [19]。
- **UniTime**（Liu等，WWW 2024）：使用领域/数据集特定的文本指令作为条件的跨域统一模型 [20]。
- **CALF**（Liu等，NeurIPS 2024）：跨模态微调，对齐时间序列与文本表示 [21]。
- **S2IP-LLM**（Pan等，NeurIPS 2024）：基于语义空间的提示学习，用于LLM时间预测 [22]。
- **LLMTime**（Gruver等，NeurIPS 2023）：通过将数字编码为文本，直接使用LLM进行零样本时间序列预测 [23]。

**研究空白**：尚无工作研究将实体标识符作为文本提示（"这是I-95号高速公路上的传感器42，测量交通流量"）与学习嵌入和位置编码进行对比。

### 2.5 时间序列基础模型

- **TimesFM**（Das等，ICML 2024）：Google的时间序列基础模型；在大规模语料上预训练 [24]。
- **Chronos**（Ansari等，ICML 2024）：Amazon的token化时间序列基础模型 [25]。
- **Moirai**（Woo等，ICML 2024）：Salesforce的统一预测Transformer，支持任意变量注意力 [26]。
- **Timer**（Liu等，ICML 2024）：面向时间序列的生成式预训练Transformer；单序列token范式 [27]。
- **Lag-Llama**（Rasul等，NeurIPS 2023 workshop）：使用滞后值作为特征的基础模型 [28]。

**研究空白**：实体标识符如何作为冻结基础模型的适配/提示机制尚未被探索。

### 2.6 面向时间序列的LLM智能体

- **时间序列AutoML**（多项工作）：自动化模型选择和超参数调优。
- **Data-Copilot**（Zhang等，2023）：用于自动化数据分析流程的LLM智能体。
- **InsightPilot**（Ma等，2023）：使用LLM智能体进行自动化数据探索。

**研究空白**：尚无LLM智能体系统专门针对实体感知预测流程设计——例如自动决定标识符类型、从元数据构建实体图、或根据数据集属性选择架构。

### 2.7 实体关系构建

- **自适应邻接学习**：Graph WaveNet [11]、MTGNN（Wu等，KDD 2020）[29]：从数据中学习实体关系。
- **基于LLM的图构建**：新兴工作利用LLM从描述或元数据推断实体关系。
- **知识增强时间序列**：将领域知识（实体层级、空间邻近性）注入预测模型。

### 2.8 跨域应用

- **医疗/ICU预测**：临床时间序列的患者特定模型（MIMIC-III/IV基准）；ICU死亡率预测的患者嵌入。
- **物联网传感器网络**：智能建筑能耗预测中的传感器特定适配；跨监测站的空气质量预测。
- **金融市场**：使用行业嵌入的股票/金融工具实体感知预测。
- **环境监测**：河流流量、气象站网络、土壤湿度传感器阵列。

### 2.9 鲁棒性与可解释性

- **分布偏移**：当传感器行为随时间变化时，实体嵌入在概念漂移下的表现。
- **缺失实体**：推理时对新的/未见实体的零样本预测。
- **嵌入可解释性**：学习得到的实体表示的可视化和分析（聚类、与元数据的关联）。

---

## 3. 当前实现状态

### 3.1 LIULIAN框架

LIULIAN框架提供了完整的实体感知预测流水线：

- **实体标识符模式**：`none`、`embedding`（学习型nn.Embedding）、`onehot`、`coordinates`、`sinusoidal`、`random`、`numeric_id`、`descriptors`
- **架构支持**：LSTM、PatchTST、DLinear、Transformer、Informer、Autoformer、FEDformer、iTransformer、TimesNet、TimeMixer、TimeXer、Mamba、TimeLLM、GPT4TS等
- **实体注入机制**：
  - `EntityWrapper`：per_entity分割模式下的逐实体嵌入
  - `ChannelEntityWrapper`：multi_channel模式下的逐通道学习嵌入
  - `ChannelTransparentWrapper`：multi_channel模式下的逐通道预计算特征（独热、正弦波、随机、坐标）
  - PatchTST原生`add_after_patch`：在补丁token空间中注入嵌入
- **实验编排**：矩阵运行器支持48个实验组合、通过Ray Tune进行超参数优化、Slurm集群调度、对比报告生成

### 3.2 当前实验矩阵

| 组件 | 取值 |
|---|---|
| 数据集 | swiss-river-1990, traffic, electricity |
| 模型 | LSTM, PatchTST, DLinear |
| 标识符模式 | none, embedding, onehot, sinusoidal, random (+coordinates仅限Swiss) |
| 随机种子 | 可配置（默认：1） |
| 超参数优化 | Ray Tune + ASHA调度器，默认10次采样 |
| 评估指标 | MSE, RMSE, MAE, NSE |

---

## 4. 研究计划：多阶段规划

### 第一阶段：系统性实体标识符消融研究（第1个月）

**论文标题**："实体标识符何时以及如何发挥作用？多实体时间序列预测中跨架构的标识符类型系统研究"

**目标会议**：NeurIPS 2026 Datasets & Benchmarks Track / AAAI 2027 / ECML-PKDD 2026

#### 4.1 目标
1. 首次在5+数据集上系统比较5+标识符类型和3+架构家族
2. 实证分析标识符何时有效（数据集特征、实体相关性、实体数量）
3. 提供标识符选择的实用指导方针
4. 开源基准框架

#### 4.2 实验设计

**阶段1A：核心消融（第1-2周）**

| 维度 | 取值 |
|---|---|
| 数据集 | swiss-river-1990, traffic, electricity, PEMS04, PEMS08 |
| 模型 | LSTM, PatchTST, DLinear |
| 标识符模式 | none, embedding, onehot, sinusoidal, random, coordinates（仅限Swiss） |
| 随机种子 | 每个组合3个种子 |
| 预测长度 | 96, 192, 336, 720（标准TSL协议） |
| 评估指标 | MSE, MAE（主要）、RMSE, NSE（次要） |

总实验数：~5数据集 x 3模型 x 5模式 x 3种子 x 4预测长度 = ~900次运行

**阶段1B：扩展架构（第2-3周）**

新增模型：iTransformer、TimeMixer、TimeXer
新增负控制数据集：Weather、ETTh1（异构通道，标识符不应有帮助）

**阶段1C：分析与撰写（第3-4周）**

- 统计显著性检验（配对Wilcoxon符号秩检验）
- 数据集特征化：实体间相关矩阵、实体数量、数据规模分析
- 消融洞见：随机 vs 学习型（隔离区分性与表示质量）、坐标 vs 嵌入（领域知识 vs 数据驱动）
- 可视化：学习嵌入的t-SNE/UMAP，以地理位置着色，嵌入相似度 vs 空间距离

#### 4.3 关键贡献
- C1：首个跨架构的实体标识符类型综合基准
- C2：实证指导方针："当N > 50且实体相关时使用嵌入；当实体仅需区分时随机基线即可"
- C3：开源LIULIAN实体标识符框架和基准
- C4：新颖的瑞士河流水文数据集作为真实世界评估领域

#### 4.4 预期洞见
- 学习嵌入应在大规模同质数据集（traffic、electricity）上占优
- 随机基线应表现出令人惊讶的良好效果（实体区分比表示质量更重要）
- 独热编码应在大N时退化（traffic的862个通道 = 非常稀疏）
- 正弦波编码应对有序实体效果良好但对无序实体效果差
- 坐标应特别有助于空间相关数据（瑞士河流）
- 负控制（Weather、ETT）应无收益——通道是异构变量而非实体

#### 4.5 时间线
| 周次 | 活动 |
|---|---|
| 1 | 在集群上运行核心3x3x5矩阵（3个种子）；收集基线结果 |
| 2 | 新增扩展模型（iTransformer、TimeMixer、TimeXer）；新增PEMS/负控制数据集 |
| 3 | 统计分析、可视化、消融分析 |
| 4 | 论文撰写、图表准备、终稿 |

---

### 第二阶段：自适应实体标识符选择（第2-3个月）

**论文标题**："AdaptID: 基于元学习的多实体时间序列预测自适应实体标识符选择"

**目标会议**：ICML 2027 / NeurIPS 2027 主会

#### 4.6 目标
1. 学习自动为每个数据集/实体选择最佳标识符类型
2. 开发通过门控注意力机制跨多种标识符表示的元学习模块
3. 证明自适应选择优于任何单一标识符类型

#### 4.7 技术方案

**多视图实体表示**：
```
x_entity = Gate([emb(id), onehot(id), sinusoidal(id), random(id), coords(id)])
```

其中`Gate`是对标识符视图的学习注意力机制：
- **输入**：数据集统计量（N_entities、时间相关矩阵、空间邻近性（如有））
- **输出**：逐实体的标识符类型软权重
- **训练**：与预测目标端到端训练 + 辅助标识符判别损失

**架构**：
1. 标识符编码器：为每个实体生成K个表示（每种标识符类型一个）
2. 数据集分析器：提取数据集级别的统计量（实体数量、实体间互信息、时间自相关）
3. 门控网络：基于数据集特征的K个标识符视图上的注意力
4. 预测骨干：任意架构（LSTM、Transformer等）

#### 4.8 关键贡献
- C1：首个面向时间序列的自适应标识符选择框架
- C2：元学习形式化，实现选择策略的跨数据集迁移
- C3：理论分析：实体相关结构与最优标识符类型之间的联系
- C4：标识符融合持续优于最佳单一标识符

---

### 第三阶段：实体标识符与时空图的融合（第3-5个月）

**论文标题**："超越节点嵌入：时空预测中实体标识的统一框架"

**目标会议**：KDD 2027 / AAAI 2027

#### 4.9 目标
1. 统一实体标识符方法与图神经网络节点嵌入
2. 研究显式图结构何时有用 vs 学习型标识符何时足够
3. 开发混合方法：标识符 + 自适应图构建

#### 4.10 技术方案

**图增强的实体标识**：
- 从实体标识符（嵌入、正弦波等）出发
- 从实体表示学习自适应邻接矩阵（参照Graph WaveNet [11]）
- 比较：预定义图（如有）vs 学习图 vs 无图 + 仅标识符
- 关键洞见：实体标识符可视为无边的GNN节点特征的特例

**实体关系构建**：
- 从数据中：从时间模式学习成对实体相似度
- 从元数据中：使用地理坐标、网络拓扑、领域知识
- 从LLM中：生成实体关系描述并转换为图结构（创新点）

**零样本实体预测（受STEP启发）**：
- 在源实体上预训练实体嵌入
- 通过以下方式迁移到未见实体：
  - 从附近已知实体插值（空间方式）
  - LLM生成的描述映射到嵌入空间
  - 从有限样本进行少样本适配

#### 4.11 关键贡献
- C1：连接实体标识符与GNN节点嵌入的统一框架
- C2：基于LLM的元数据实体图构建
- C3：通过嵌入迁移实现零样本实体预测
- C4：在标准基准上全面比较：仅标识符 vs GNN vs 混合方法

---

### 第四阶段：LLM增强的实体感知预测（第5-8个月）

**论文标题**："EntityLLM: 大语言模型增强的实体表示用于通用时间序列预测"

**目标会议**：ICLR 2028 / NeurIPS 2027

#### 4.12 目标
1. 利用LLM从元数据生成丰富的实体描述
2. 对齐实体文本描述与学习嵌入
3. 通过自然语言规范实现零样本实体适配
4. 构建LLM智能体用于自动化实体感知预测流程设计

#### 4.13 技术方案

**文本增强的实体标识符**：
```
实体描述："传感器42是I-95号高速公路上的交通流量传感器，
         位于坐标(40.7, -74.0)，2015年安装，
         在郊区测量车辆计数。"
→ LLM编码器 → entity_text_emb → 与entity_learned_emb对齐
```

**面向预测流程的LLM智能体**：
- 智能体分析数据集元数据并推荐：
  - 标识符类型（基于实体数量、元数据可用性、相关结构）
  - 架构选择（基于数据集规模、预测时域、计算预算）
  - 图结构（如有空间元数据）
- 智能体可迭代：运行初始实验、分析结果、优化选择

**多模态实体表示**：
- 融合：时间序列模式 + 文本描述 + 空间坐标 + 图像（水文学的卫星图像）
- 基于注意力的跨模态融合
- 每种模态提供互补的实体信息

#### 4.14 关键贡献
- C1：首个面向时间序列的LLM增强实体表示
- C2：用于零样本适配的自然语言实体规范
- C3：用于自动化实体感知预测的LLM智能体
- C4：多模态实体融合框架

---

### 第五阶段：跨域实体感知基础模型（第8-12个月）

**论文标题**："UniEntity: 面向多域时间序列预测的通用实体感知基础模型"

**目标会议**：ICML 2028 / Nature Machine Intelligence

#### 4.15 目标
1. 构建原生支持跨域实体标识符的基础模型
2. 在来自多个领域的多样化多实体数据集上预训练
3. 通过实体感知提示实现对新领域的少样本适配
4. 展示跨域实体迁移（例如交通→水文）

#### 4.16 技术方案

**领域无关的实体框架**：
- 实体类型本体：传感器、患者、金融工具、气象站等
- 跨域共享的实体嵌入空间
- 领域特定的实体适配器（LoRA风格）

**跨域应用**：

| 领域 | 实体类型 | 实体元数据 | 关键挑战 |
|---|---|---|---|
| 交通 | 交通传感器 | 位置、道路类型、限速 | 空间相关、周期模式 |
| 能源 | 电力表 | 客户类型、位置、容量 | 多样化消费模式 |
| 水文 | 河流测量站 | 坐标、集水面积、海拔 | 网络拓扑、极端事件 |
| 医疗 | ICU患者 | 人口统计、诊断、用药 | 高变异性、缺失数据 |
| 物联网 | 建筑传感器 | 楼层、房间、传感器类型 | 异构传感器类型 |
| 金融 | 股票/金融工具 | 行业、市值、国家 | 非平稳、体制变化 |
| 环境 | 空气质量监测器 | 位置、与道路的距离 | 空间扩散模式 |

**围手术期患者聚类**（具体应用）：
- 将每位患者视为围手术期时间序列中的一个实体
- 实体特征：人口统计学、手术类型、合并症
- 任务：使用患者感知模型预测术后并发症
- 实体聚类揭示具有不同结局轨迹的患者表型

#### 4.17 关键贡献
- C1：首个面向时间序列的实体感知基础模型
- C2：跨域实体迁移学习
- C3：领域特定的实体适配器设计
- C4：围手术期患者预测的应用

---

## 5. 需要纳入的关键竞争基线

以下架构需添加到LIULIAN框架中以进行全面比较：

| 架构 | 重要性 | 工作量 |
|---|---|---|
| **iTransformer** [2] | 通过"变量即token"实现隐式实体token；最直接的竞争者 | 中等（TSL中已有适配器） |
| **STID** [6] | 简单MLP + 空间ID；证明仅标识符即可匹配GNN | 低（简单架构） |
| **CARD** [3] | Transformer中的通道特定token | 中等 |
| **TiDE** [7] | 逐变量的静态协变量（= 实体特征） | 中等 |
| **Graph WaveNet** [11] | 自适应图 + 节点嵌入；GNN基线 | 高 |
| **DCRNN** [10] | 预定义图 + 节点特征；经典GNN基线 | 高 |

---

## 6. 创新亮点

### 6.1 近期创新（第一阶段）
- **随机嵌入基线**：简单但关键的消融——隔离"实体区分"与"实体表示质量"
- **ChannelTransparentWrapper**：在multi_channel模式下注入预计算特征的新型架构
- **负控制实验**：首次研究证明标识符在异构通道数据集上不应有帮助

### 6.2 中期创新（第二-三阶段）
- **AdaptID元学习**：首个通过门控注意力实现的自适应标识符选择
- **基于LLM的图构建**：利用LLM从描述推断实体关系
- **零样本实体预测**：将实体嵌入迁移到未见实体

### 6.3 长期创新（第四-五阶段）
- **文本增强的实体标识符**：自然语言实体规范
- **LLM预测智能体**：自动化流程设计
- **跨域实体基础模型**：通用实体表示

---

## 7. 评估协议

### 7.1 数据集

**主要数据集（同质多实体）**：
| 数据集 | 实体数 | 长度 | 领域 | 来源 |
|---|---|---|---|---|
| Traffic | 862 | 17,544 | 道路传感器 | TSL基准 |
| Electricity | 321 | 26,304 | 电力客户 | TSL基准 |
| Swiss-River-1990 | ~50 | ~10,000 | 测量站 | 新数据集（LIULIAN） |
| PEMS03/04/07/08 | 170-883 | 16,992-26,208 | 交通传感器 | STGNN基准 |

**负控制（异构通道）**：
| 数据集 | 通道数 | 为何为负控制 | 来源 |
|---|---|---|---|
| Weather | 21 | 通道是不同的物理变量 | TSL基准 |
| ETTh1/ETTm1 | 7 | 通道是异构测量值 | TSL基准 |

**扩展领域（第三-五阶段）**：
| 数据集 | 领域 | 来源 |
|---|---|---|
| METR-LA, PEMS-BAY | 空间交通 | STGNN基准 |
| MIMIC-III/IV | ICU患者 | PhysioNet |
| 空气质量 | 环境 | EPA/WHO |
| 太阳能 | 光伏装置 | NREL |

### 7.2 评估指标
- **主要**：MSE、MAE（标准TSL协议）
- **次要**：RMSE、MAPE、NSE（水文学特定）
- **统计**：3+种子的均值 +/- 标准差，Wilcoxon符号秩检验

### 7.3 基线方法
- 无标识符（纯共享模型）
- 每种标识符类型独立测试
- iTransformer（隐式实体token）
- STID（MLP + 空间ID）
- Graph WaveNet（学习图 + 节点嵌入）

---

## 8. 资源需求

| 资源 | 第一阶段 | 第二-三阶段 | 第四-五阶段 |
|---|---|---|---|
| GPU小时 | ~500（集群） | ~2000 | ~5000 |
| 存储 | 50 GB | 200 GB | 1 TB |
| 人力投入 | 1个月（AI辅助） | 3个月 | 4个月 |
| 计算资源 | RTX 4090集群 | A100集群 | A100/H100集群 |

---

## 9. 风险分析与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|---|---|---|---|
| 实体标识符在大多数数据集上收益甚微 | 中 | 高 | 定位为负面结果（有价值）；聚焦于"何时"而非"是否" |
| 随机基线匹配学习嵌入 | 中 | 中 | 重要发现——实体区分 >> 表示质量 |
| 大实体数量的可扩展性问题（Traffic的862个） | 低 | 中 | 独热编码退化；嵌入良好扩展；将此作为研究发现 |
| 负控制显示出意外收益 | 低 | 低 | 调查并报告；可能揭示有趣的通道交互 |
| LLM实体描述未能优于数值嵌入 | 中 | 中 | 聚焦于零样本/迁移场景，文本在此具有独特价值 |

---

## 10. 参考文献

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

[39] Harutyunyan, H., et al. "Multitask Learning and Benchmarking with Clinical Time Series Data." Scientific Data 2019. (MIMIC基准)

[40] Tang, S., et al. "Spatiotemporal Multi-Graph Convolution Network for Ride-Hailing Demand Forecasting." AAAI 2020.

---

## 11. 第一阶段详细执行计划（1个月）

### 第1周：基础设施和核心实验

**第1-2天：框架就绪**
- [ ] 验证所有48个矩阵任务通过空运行
- [ ] 将iTransformer、TimeMixer、TimeXer添加到实验矩阵
- [ ] 将PEMS04、PEMS08数据集添加到矩阵
- [ ] 将Weather、ETTh1添加为负控制数据集
- [ ] 将默认种子数增加到3个（2024、2025、2026）
- [ ] 设置预测长度扫描（96、192、336、720）作为配置选项

**第3-4天：集群提交**
- [ ] 向集群提交阶段1A核心矩阵（swiss、traffic、electricity x LSTM、PatchTST、DLinear x 5种模式 x 3个种子 x 4个预测长度）
- [ ] 监控首批运行；修复任何流水线问题
- [ ] 提交扩展模型（iTransformer、TimeMixer、TimeXer）

**第5-7天：扩展数据集**
- [ ] 提交PEMS04、PEMS08实验
- [ ] 提交负控制（Weather、ETTh1 x 所有模型 x 仅none+embedding）
- [ ] 收集首批结果；验证合理性

### 第2周：全面实验运行

**第8-10天：结果收集与验证**
- [ ] 验证所有阶段1A结果已完成
- [ ] 运行比较报告；检查异常
- [ ] 重新运行失败的实验
- [ ] 开始在集群上运行扩展模型实验

**第11-14天：附加分析**
- [ ] 计算每个数据集的实体间相关矩阵
- [ ] 运行学习嵌入的t-SNE/UMAP可视化
- [ ] 计算嵌入相似度 vs 空间距离（瑞士河流）
- [ ] 生成综合比较表

### 第3周：分析与图表

**第15-17天：统计分析**
- [ ] 计算所有实验跨种子的均值 +/- 标准差
- [ ] 运行Wilcoxon符号秩检验：嵌入 vs 无标识符，每种模式 vs 无标识符
- [ ] 创建性能提升热图（数据集 x 模型 x 模式）
- [ ] 分析实体数量对标识符收益的影响

**第18-21天：可视化与洞见**
- [ ] 创建主结果表（所有数据集 x 模型 x 模式）
- [ ] 创建图表：性能 vs 实体数量
- [ ] 创建图表：嵌入t-SNE以地理位置着色
- [ ] 创建图表：标识符类型推荐流程图
- [ ] 起草关键发现和实用指导方针

### 第4周：论文撰写

**第22-24天：论文结构与初稿**
- [ ] 撰写引言和研究动机（对标PatchTST CI争论的定位）
- [ ] 撰写相关工作（综合版，源自第2节）
- [ ] 撰写方法论（LIULIAN框架、标识符类型、注入机制）
- [ ] 撰写实验设置

**第25-27天：结果与分析**
- [ ] 撰写主要结果部分，含表格和图表
- [ ] 撰写分析部分（标识符何时有效？为什么？）
- [ ] 撰写消融研究（随机 vs 学习型、随N缩放、负控制）
- [ ] 撰写实用指导方针部分

**第28-30天：润色与提交**
- [ ] 撰写摘要和结论
- [ ] 定稿所有图表
- [ ] 内部审阅与修改
- [ ] 准备补充材料（完整表格、代码、配置）
- [ ] 提交至arXiv；准备特定会议投稿

---

## 12. 全阶段研究贡献总结

| 阶段 | 贡献 | 创新程度 | 影响 |
|---|---|---|---|
| 1 | 系统性标识符类型基准 | **高** — 首次综合研究 | 基准论文；为该领域建立基线 |
| 2 | 自适应标识符选择（AdaptID） | **非常高** — 新型元学习模块 | 方法论文；为从业者提供实用工具 |
| 3 | 统一的标识符 + GNN框架 | **高** — 桥接两个研究社区 | 框架论文；连接标识符和GNN两个领域 |
| 4 | LLM增强的实体表示 | **非常高** — 新范式 | 愿景论文；定义新的研究方向 |
| 5 | 跨域实体基础模型 | **极高** — 宏大范围 | 若成功，具有很高的引用潜力 |

---

## 附录A：LIULIAN实体标识符架构图

```
                    ┌─────────────────────────────────────┐
                    │         实体标识符层                   │
                    │                                       │
                    │  none ──── 直通                       │
                    │  embedding ── nn.Embedding → 投影     │
                    │  onehot ──── I_N矩阵 → 投影           │
                    │  sinusoidal ─ PE(idx) → 投影          │
                    │  random ──── hash(seed:name) → 投影   │
                    │  coordinates ─ (lat,lon) → 投影       │
                    └──────────┬────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼─────────┐
    │  EntityWrapper  │  │ Channel    │  │ Channel        │
    │  (逐实体)       │  │ Entity     │  │ Transparent    │
    │                 │  │ Wrapper    │  │ Wrapper        │
    │  emb → 拼接     │  │ (学习型)   │  │ (预计算)       │
    │  → 投影         │  │            │  │                │
    │  → 内部模型     │  │ 逐通道emb  │  │ 逐通道特征     │
    └─────────────────┘  │ → 拼接     │  │ → 拼接         │
                         │ → 投影     │  │ → 投影         │
                         │ → 内部     │  │ → 内部         │
                         └────────────┘  └────────────────┘
                                ↓
                    ┌──────────────────────┐
                    │      预测模型         │
                    │  (LSTM / PatchTST /   │
                    │   DLinear / iTransf.) │
                    └──────────────────────┘
```

## 附录B：代码制品

| 文件 | 描述 |
|---|---|
| `experiments/entity_identifier/run.py` | 矩阵实验运行器 |
| `experiments/entity_identifier/matrix.py` | 矩阵定义和任务扩展 |
| `experiments/entity_identifier/compare.py` | 对比报告生成器 |
| `experiments/entity_identifier/submit_slurm.py` | Slurm集群调度器 |
| `liulian/models/torch/entity_mixin.py` | EntityWrapper, ChannelEntityWrapper, ChannelTransparentWrapper |
| `liulian/data/ts/timeseriesdataset.py` | make_entity_features和透明模式数据层 |
| `experiments/run.py` | 统一实验入口点 |
| `tests/runtime/test_entity_identifier_pipeline.py` | 流水线测试（15个测试） |
