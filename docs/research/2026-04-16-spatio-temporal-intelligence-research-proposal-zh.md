# LIULIAN：从实体感知预测到通用时空智能——综合研究计划

_日期：2026-04-16（修订版）_
_首席研究员：贾林林 (jajupmochi@gmail.com)_
_框架：LIULIAN — Liquid Intelligence and Unified Logic for Interactive Adaptive Networks_

---

## 摘要

本提案概述了一项**多年度、多阶段的研究计划**，从时间序列预测中实体标识符的聚焦研究出发，逐步扩展为**通用时空智能平台**。该计划横跨三个阶段、七个研究周期：

- **A阶段（第1–5个月）：** 实体感知预测——系统消融、自适应选择和空间图集成。三篇论文。
- **B阶段（第5–12个月）：** 时空基础模型——LLM增强的时空建模、跨域时空基础模型以及面向时空数据的Mamba/SSM架构。三篇论文。
- **C阶段（第8–18个月）：** LIULIAN平台本身——一个开源、智能体驱动、集成BI功能的时空智能平台，配备垂直领域插件。一篇系统论文 + 领域应用论文。

每个研究周期产出一篇可发表的论文。平台本身作为独立的高影响力贡献，面向顶级系统会议。跨域应用（水文、医疗、灾害管理、金融、物联网、材料科学）作为验证，并产出额外的领域论文。

**计划总产出：** 7+篇核心论文，1篇平台/系统论文，3+篇领域应用论文。

---

## 1. 问题陈述与研究动机

### 1.1 时空数据挑战

时空数据——在时间和空间上分布的测量数据——是现代传感、基础设施监测和科学发现的基石。交通网络每小时产生数百万条传感器数据；医院ICU对各床位持续产生患者生命体征；河流监测站追踪整个流域的水质；卫星影像捕捉大陆尺度的土地利用变化。

尽管时空数据无处不在，该领域在多个维度上存在**碎片化**：

1. **方法碎片化**：时间序列模型（PatchTST、DLinear、iTransformer）忽视空间结构；图神经网络（DCRNN、Graph WaveNet、MTGNN）聚焦于固定拓扑；基于LLM的方法（Time-LLM、UrbanGPT）缺乏系统性的实体/空间感知。

2. **领域碎片化**：交通预测、空气质量预测、水文建模、临床时间序列和金融分析各有其数据集、指标、基线和规范——领域间的交叉借鉴极为有限。

3. **基础设施碎片化**：研究者需要拼凑TSLib [1]用于时序模型、PyG [2]用于图模型、GluonTS [3]用于概率预测，以及自定义脚本用于领域特定预处理。没有单一平台支持从原始时空数据到科学洞见的完整流程。

4. **实体表示碎片化**：一个基本设计决策——如何在多实体系统中表示实体身份——从未被系统研究过。不同的工作使用学习嵌入 [6]、独热编码、地理坐标、图节点特征 [10,11]或文本描述 [18]，却缺乏原则性比较。

### 1.2 愿景

本研究计划通过统一方法解决所有四种碎片化：

- **方法统一**：无缝结合实体标识符、图结构、时序模型和LLM增强表示的框架。
- **领域统一**：在15+个数据集上进行跨域评估，涵盖交通、能源、水文、医疗、物联网和金融。
- **基础设施统一**：LIULIAN平台——一个开源、模块化、智能体驱动的时空智能系统，具备BI功能和垂直领域插件。
- **实体表示统一**：首次系统性的实体表示策略研究，从消融到自适应选择，再到基础模型级别的实体感知。

---

## 2. 文献综述与相关工作

### 2.1 通道独立与通道混合在时间序列中的争论

多变量时间序列模型是否应独立处理各通道还是允许跨通道交互的争论是基础性的。

- **PatchTST**（Nie等，ICLR 2023）[1]：通道独立（CI）补丁处理通常优于通道混合。
- **iTransformer**（Liu等，ICLR 2024）[4]：反转Transformer范式——变量作为token——通过隐式实体表示取得优异结果。
- **CARD**（Wang等，ICLR 2024）[5]：通道对齐鲁棒混合Transformer，添加通道特定token（可学习实体嵌入）。
- **CrossFormer**（Zhang & Yan，ICLR 2023）[6]：跨维度注意力与可学习维度/段嵌入。
- **TSMixer**（Chen等，NeurIPS 2023）[7]：基于MLP的跨变量混合捕获实体间关系。
- **Client**（Gao等，KDD 2024）[8]：整合跨变量线性组件与Transformer注意力。

**研究空白**：各自测试一种嵌入策略；没有一项跨架构系统比较标识符类型。

### 2.2 显式实体/节点嵌入

- **STID**（Shao等，CIKM 2022）[9]：简单MLP + 空间身份嵌入匹配复杂GNN。**关键洞见：空间不可区分性，而非架构复杂性，才是瓶颈。**
- **TiDE**（Das等，TMLR 2023）[10]：MLP编码器-解码器，逐变量静态协变量。
- **TimeXer**（Wang等，NeurIPS 2024）[11]：外生变量token与内生补丁token并列。
- **UniTS**（Gao等，NeurIPS 2024）[12]：统一多任务模型，使用数据集特定提示token。
- **TRA**（Lin等，AAAI 2023）[13]：时间路由适配器，用于实体感知股票预测——逐股票的时间模式路由。

### 2.3 时空图神经网络

#### 2.3.1 经典与基础性ST-GNN

- **DCRNN**（Li等，ICLR 2018）[14]：有向交通图上的扩散卷积循环神经网络。
- **Graph WaveNet**（Wu等，IJCAI 2019）[15]：从数据中学习自适应邻接矩阵。
- **MTGNN**（Wu等，KDD 2020）[16]：连接节点——通过学习图结构的多变量时序预测。
- **STEP**（Shao等，KDD 2022）[17]：预训练时空嵌入，支持对未见节点的零样本预测。
- **STNorm**（Deng等，KDD 2021）[18]：实体特定归一化作为显式标识符的替代。
- **MegaCRN**（Jiang等，AAAI 2023）[19]：带元节点库的元图卷积循环网络。
- **PDFormer**（Jiang等，AAAI 2023）[20]：传播延迟感知的动态长程Transformer。
- **STAEformer**（Liu等，CIKM 2023）[21]：时空自适应嵌入Transformer，逐节点联合时空嵌入。

#### 2.3.2 自适应图学习（超越固定拓扑）

- **TESTAM**（Lee等，ICLR 2024）[22]：混合专家（MoE），三个专家（无图、静态图、动态图）——将路由重新表述为分类问题。**新颖性检查：最接近我们的AdaptID概念，但针对图结构而非标识符类型进行门控。**
- **STID** [9]：证明仅标识符即可完全替代图。**我们工作的关键基线。**

#### 2.3.3 可扩展ST-GNN

- **SGP**（Cini等，AAAI 2023）[23]：随机化循环 + 邻接矩阵幂次实现节点级并行训练。
- **BigST**（Han等，VLDB 2024）[24]：线性复杂度空间卷积，可扩展至100K+节点。
- **EasyST**（Tang等，arXiv 2024）[25]：从教师STGNN到学生MLP的知识蒸馏。

#### 2.3.4 时空Transformer

- **AirFormer**（Liang等，AAAI 2023）[26]：解耦ST学习，确定性+随机阶段，覆盖1,085个空气质量站点。
- **EarthFormer**（Gao等，NeurIPS 2022）[27]：面向3D时空地球观测数据的立方体注意力。

#### 2.3.5 Mamba/SSM用于时空数据

- **SpoT-Mamba**（Choi等，IJCAI 2024 Workshop）[28]：通过节点特定游走序列在ST图上应用选择性状态空间模型。
- **STG-Mamba**（Li等，arXiv 2024）[29]：首次探索Mamba用于完整ST图学习的ST-S3M模块。

**研究空白**：GNN文献将实体嵌入视为节点特征，但从未比较嵌入策略或研究何时更简单的标识符即可满足需求。没有工作桥接实体标识符文献与GNN节点嵌入文献。

### 2.4 时空数据的预训练和基础模型

- **GPT-ST**（Li等，NeurIPS 2023）[30]：带自适应掩码的ST-GNN生成式预训练。
- **UniST**（Yuan等，KDD 2024）[31]：通用ST模型，使用时空知识引导的提示，在20+城市场景中实现零样本泛化。
- **OpenCity**（Li等，ACM TIST 2024）[32]：结合Transformer + GNN，在异构交通数据上预训练以实现零样本迁移。

**研究空白**：所有ST基础模型都以交通/城市为中心。没有一个支持跨域（水文、医疗、物联网、金融）的实体感知条件化。

### 2.5 时间序列基础模型

- **TimesFM**（Das等，ICML 2024）[33]：Google的decoder-only模型，在100B时间点上预训练。
- **Chronos**（Ansari等，ICML 2024）[34]：Amazon的token化时序模型，使用T5架构。
- **Moirai**（Woo等，ICML 2024）[35]：Salesforce的任意变量注意力与混合分布。
- **Timer**（Liu等，ICML 2024）[36]：面向时序的GPT式生成预训练，S3格式。
- **MOMENT**（Goswami等，ICML 2024）[37]：在Time Series Pile上的掩码预训练。
- **Lag-Llama**（Rasul等，ICML 2024）[38]：LLaMA风格，使用滞后特征进行概率预测。
- **TTM**（Ekambaram等，NeurIPS 2024）[39]：微型时间混合器——不到100万参数，零样本表现优异。
- **Time-MoE**（Shi等，arXiv 2024）[40]：十亿参数级时序基础模型，稀疏MoE。
- **TimeGPT-1**（Garza & Mergenthaler-Canseco，arXiv 2023）[41]：首个生产级API时序基础模型。

**研究空白**：所有时序基础模型将序列视为匿名通道。没有一个具有显式实体身份条件化。**这是本计划填补的核心空白。**

### 2.6 基于LLM的时间序列预测

- **GPT4TS / One Fits All**（Zhou等，NeurIPS 2023）[42]：冻结GPT-2，最小适配。
- **Time-LLM**（Jin等，ICLR 2024）[43]：通过输入变换重编程冻结LLM。
- **TEST**（Sun等，ICLR 2024）[44]：文本原型对齐嵌入用于时序。
- **UniTime**（Liu等，WWW 2024）[45]：领域特定文本指令作为条件。
- **CALF**（Liu等，NeurIPS 2024）[46]：跨模态微调对齐时序与文本。
- **S2IP-LLM**（Pan等，ICML 2024）[47]：语义空间信息提示学习。
- **LLMTime**（Gruver等，NeurIPS 2023）[48]：将数字编码为文本的零样本时序预测。
- **TEMPO**（Cao等，ICLR 2024）[49]：趋势/季节/残差分解加每组件软提示。
- **AutoTimes**（Liu等，NeurIPS 2024）[50]：基于LLM的自回归上下文学习。
- **"语言模型对时间序列预测真的有用吗？"**（Tan等，NeurIPS 2024）[51]：**关键论文**——证明LLM骨干可被简单注意力或随机嵌入替代，质疑LLM预训练是否真正迁移。

**研究空白**：没有工作研究实体标识符作为文本提示与学习嵌入和位置编码的对比。没有工作将LLM实体描述与时空图结构结合。

### 2.7 时空LLM

- **UrbanGPT**（Li等，KDD 2024）[52]：ST依赖编码器 + 指令微调LLM用于城市计算，支持零样本跨城市迁移。
- **ST-LLM**（Cao等，arXiv 2024）[53]：token化ST序列直接输入LLM，带空间位置嵌入。
- **TrafficGPT**（Zhang等，Transport Policy 2024）[54]：ChatGPT + 交通基础模型用于对话式交通分析。

**研究空白**：ST-LLM聚焦于交通/城市。没有处理任意时空域或将实体元数据作为自然语言集成。

### 2.8 面向数据分析和科学发现的LLM智能体

- **Data-Copilot**（Zhang等，arXiv 2023）[55]：自主数据查询、处理、可视化工作流。
- **InsightPilot**（Ma等，VLDB 2024）[56]：通过LLM生成分析动作的迭代数据探索。
- **The AI Scientist**（Lu等，arXiv 2024）[57]：端到端研究自动化——从想法生成到论文撰写。
- **DS-Agent**（Guo等，ICML 2024）[58]：基于案例推理的Kaggle类ML问题求解。
- **MLAgentBench**（Huang等，ICML 2024）[59]：LLM智能体ML实验基准。
- **AutoML-GPT**（Zhang等，arXiv 2023）[60]：GPT-4用于自动模型选择和流水线组合。
- **MatPlotAgent**（Li等，ACL 2024）[61]：面向出版级科学可视化的智能体。
- **ResearchAgent**（Baek等，ACL 2024）[62]：通过审阅-优化循环的迭代研究假设生成。
- **ChemCrow**（Bran等，Nature Machine Intelligence 2024）[63]：配备18个化学工具的LLM智能体。
- **Coscientist**（Boiko等，Nature 2023）[64]：LLM驱动的自主化学实验与机器人实验室。

**研究空白**：没有LLM智能体系统专门面向时空预测流水线设计——自动决定实体表示、图构建、架构选择和HPO。**这是我们提出的新颖贡献。**

### 2.9 AI驱动的商业智能与自然语言界面

- **LIDA**（Dibia，Microsoft Research，ACL 2023）[65]：多阶段LLM可视化生成流水线。
- **DIN-SQL**（Pourreza & Rafiei，NeurIPS 2023）[66]：分解式text-to-SQL与自校正。
- **DAIL-SQL**（Gao等，VLDB 2024）[67]：高效少样本text-to-SQL。
- **NL4DV**（Narechania等，IEEE VIS 2023）[68]：自然语言到Vega-Lite可视化规范。
- **Chat2VIS**（Maddigan & Susnjak，IEEE Access 2023）[69]：LLM从自然语言生成Python可视化代码。
- **TableGPT**（Li等，arXiv 2023）[70]：面向整体表格理解的微调LLM。
- **Rath**（Yu等，Kanaries，2022–2024）[71]：开源自动化数据探索与可视化。
- **Sheet2Report**（Chen等，arXiv 2024）[72]：从电子表格生成专业分析报告的LLM流水线。

**研究空白**：没有BI平台与时空ML集成。现有BI工具缺乏图感知分析、空间可视化和领域特定ST模型集成。

### 2.10 面向时间序列和图的ML/AI平台

| 平台 | 空间 | 缺失数据 | 概率 | 多任务 | 插件架构 | 实体感知 | LLM/智能体 | BI |
|---|---|---|---|---|---|---|---|---|
| **TSLib** [73] | 否 | 否 | 否 | 是(5) | 否 | 否 | 否 | 否 |
| **BasicTS** [74] | 是 | 否 | 否 | 预测 | 否 | 否 | 否 | 否 |
| **tsl** [75] | 是 | 是 | 否 | 预测+填补 | 部分 | 否 | 否 | 否 |
| **GluonTS** [3] | 否 | 否 | 是 | 预测 | 是 | 否 | 否 | 否 |
| **NeuralForecast** [76] | 有限 | 否 | 是 | 预测 | 否 | 否 | 否 | 否 |
| **Darts** [77] | 否 | 否 | 是 | 预测 | 否 | 否 | 否 | 否 |
| **PyPOTS** [78] | 否 | 是 | 否 | 是(4+) | 是 | 否 | 否 | 否 |
| **LibCity** [79] | 是 | 否 | 否 | 城市任务 | 部分 | 隐式 | 否 | 否 |
| **PyG** [2] | 是 | 否 | 否 | 通用GNN | 是 | 节点特征 | 否 | 否 |
| **LIULIAN（本文）** | **是** | **规划中** | **规划中** | **是** | **是** | **是** | **是** | **是** |

**研究空白**：没有现有平台同时结合(a)实体感知模块、(b)图+时序统一建模、(c)LLM智能体集成、(d)BI功能和(e)垂直领域插件。**LIULIAN填补此空白。**

### 2.11 跨域应用

#### 医疗/临床时间序列
- **STraTS**（Tipirneni & Reddy，MLHC 2022）[80]：不规则采样临床时序上的稀疏注意力。
- **MedGTX**（Kim等，CHIL 2023）[81]：结合患者关系图和时序EHR的图-Transformer。
- **MISTS**（Li等，AAAI 2024）[82]：面向ICU的多输入多输出不规则采样时序预测。
- **MedTsLLM**（Jiang等，arXiv 2024）[83]：临床时序 + 临床笔记的LLM多模态融合。
- **围手术期GNN**（Gao等，npj Digital Medicine 2024）[84]：患者图用于术后并发症预测。

#### 环境/水文
- **AirFormer** [26]：全国范围1,085个站点的空气质量预测。
- **GAGNN**（ACM TKDD 2024）[85]：面向全国PM2.5预测的组感知图神经网络。
- **FloodCast**（Xu等，arXiv 2023）[86]：河流网络拓扑GNN用于洪水预测。
- **ST水质GNN**（Water Research 2025）[87]：面向溶解氧和氮的多尺度ST-GNN。
- **水分配网络GNN**（Zanfei等，Water Research 2023）[88]：面向管网泄漏检测的GNN。
- **多尺度水需求STGNN**（Water Research 2025）[89]：面向水基础设施的多尺度时空建模。

#### 金融
- **FinGPT**（Yang等，NeurIPS 2023 Workshop）[90]：开源金融LLM，结合时序情感信号。
- **TRA** [13]：面向实体感知股票预测的时间路由适配器。
- **MASTER**（Li等，AAAI 2024）[91]：市场引导的股票Transformer，用于投资组合级预测。

#### 灾害管理/地球科学
- **EarthFormer** [27]：面向降水、温度、海表的立方体注意力。
- **Prithvi**（Jakubik等，IBM/NASA，arXiv 2023）[92]：地理空间AI基础模型。
- **GeoChat**（Kuckreja等，CVPR 2024）[93]：面向遥感分析的视觉-语言模型。
- **SatCLIP / GeoCLIP**（Klemmer等，NeurIPS 2023 Workshop / AAAI 2024）[94]：对齐卫星影像与地理元数据的位置感知嵌入。
- **HeatGNN**（arXiv 2024）[95]：流行病学信息驱动的GNN用于疫情预测。

#### 物联网/智慧建筑
- **Brick Schema + ML**（Balaji等，BuildSys 2022）[96]：基于GNN的跨建筑迁移学习。
- **CityLearn**（Vazquez-Canteli等，Applied Energy 2023）[97]：建筑能源数字孪生的强化学习环境。

### 2.12 时空数据的对比学习和自监督学习

- **TS2Vec**（Yue等，AAAI 2022）[98]：分层对比学习，用于通用时序表示。
- **TF-C**（Zhang等，NeurIPS 2022）[99]：时频一致性对比预训练。
- **SimMTM**（Dong等，NeurIPS 2023）[100]：基于邻居聚合的掩码时序建模。
- **AutoST**（Zhang等，WWW 2023）[101]：自动化时空图对比学习。
- **STAG**（Zhang等，ICML 2023）[102]：对抗性对比适应用于时空图学习。
- **CaST**（Xia等，NeurIPS 2023）[103]：因果视角下的时空图预测。
- **STONE**（Wang等，KDD 2024）[104]：同时处理空间和时间分布偏移。

### 2.13 时空模型中的分布偏移与鲁棒性

- **ADATIME**（Ragab等，ACM TKDD 2023）[105]：11种域适应方法在时序数据上的基准。
- **RAINCOAT**（He等，ICML 2023）[106]：频域特征对齐的时序域适应。

### 2.14 时空模型的可解释性

- **STExplainer**（Zhang等，KDD 2023）[107]：基于互信息扰动的ST-GNN事后解释。
- **TimeSHAP**（Bento等，KDD 2021）[108]：序列模型的SHAP——跨时间步的特征重要性。
- **TFT**（Lim等，IJF 2021）[109]：通过变量选择网络的内置可解释性。

### 2.15 ML框架中的插件/适配器架构

- **LoRA**（Hu等，ICLR 2022）[110]：面向冻结模型的参数高效适配器层——我们领域适配器设计的基础。
- **Ludwig**（Molino等，JMLR 2023）[111]：声明式YAML + 编码器-组合器-解码器插件架构。
- **MMEngine**（OpenMMLab，2022）[112]：注册表 + 钩子 + 运行器架构模式。
- **Hydra**（Yadan，Facebook Research，2020）[113]：可组合YAML配置与插件架构。

### 2.16 动态和时序图网络

- **TGN**（Rossi等，ICML 2020 Workshop）[114]：连续时间动态图上的记忆 + 图注意力 + 时间编码。
- **DyGFormer**（Yu等，NeurIPS 2023）[115]：面向连续时间动态图的Transformer，带邻居共现编码。
- **ROLAND**（You等，KDD 2022）[116]：面向动态社交网络的增量式GNN。

### 2.17 多模态时空数据

- **Time-MMD**（Liu等，NeurIPS 2024 D&B）[117]：跨9个领域配对时序与文本元数据的大规模基准。
- **ChatTS**（Xie等，arXiv 2024）[118]：合成时序-文本对用于LLM时序推理。
- **UniST** [31]：多模态提示（空间、时间、文本）用于城市ST预测。
- **TorchGeo**（Stewart等，SIGSPATIAL 2022）[119]：桥接GIS数据格式与深度学习训练流水线。

### 2.18 基准与公平评估

- **TSLib**（Wu等，NeurIPS 2023 D&B）[73]：公平重实现揭示许多SOTA结果是评估伪影。
- **BasicTS**（Shao等，NeurIPS 2023 D&B）[74]：公平评估下简单模型常可匹配复杂GNN。
- **TFB**（Qiu等，ICML 2024）[120]：8,068个单变量 + 25个多变量数据集挑战声称的SOTA。
- **TOTEM**（Talukder等，NeurIPS 2024）[121]：VQ-VAE离散token化创建跨域共享词表。

### 2.19 人在回路ML与报告生成

- **HITL-TSF**（Li等，Salesforce，arXiv 2024）[122]：将领域知识注入神经预测的框架。
- **Calliope**（Shi等，IEEE TVCG 2023）[123]：从电子表格自动生成可视化数据故事。
- **DataTales**（Sultan等，CHI 2024）[124]：LLM从数据集生成叙事文章。

### 2.20 数字孪生与仿真 + ML

- **TwinGen**（Thelen等，Nature Machine Intelligence 2023）[125]：物理信息神经网络用于数据高效数字孪生。
- **Transolver**（Wu等，ICML 2024 Spotlight）[126]：通用几何上的快速Transformer PDE求解器。
- **PARCv2**（Nguyen等，ICML 2024）[127]：物理感知循环卷积用于时空动力学。

---

## 3. 当前实现状态

### 3.1 LIULIAN框架架构

LIULIAN（Liquid Intelligence and Unified Logic for Interactive Adaptive Networks）提供模块化、任务驱动的时空智能流水线：

```
                    ┌───────────────────────────────────────┐
                    │        LIULIAN 平台技术栈              │
                    │                                       │
                    │  ┌─────────────────────────────────┐  │
                    │  │   LLM智能体层（规划中）           │  │
                    │  │   • 流水线设计智能体              │  │
                    │  │   • 数据摄入智能体                │  │
                    │  │   • 报告生成智能体                │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   BI与可视化层                    │  │
                    │  │   • 仪表盘（规划中）              │  │
                    │  │   • 自然语言查询界面（规划中）     │  │
                    │  │   • 自动化报告（规划中）           │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   任务层                          │  │
                    │  │   • 预测（已实现）                │  │
                    │  │   • 分类（规划中）                │  │
                    │  │   • 异常检测（规划中）            │  │
                    │  │   • 数据填补（规划中）            │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   模型层                          │  │
                    │  │   • 时序: LSTM, PatchTST,        │  │
                    │  │     DLinear, iTransformer, ...    │  │
                    │  │   • 实体感知包装器                │  │
                    │  │   • 空间: GNN模块                 │  │
                    │  │   • 基于LLM: TimeLLM, GPT4TS     │  │
                    │  │   • 基础模型: Chronos, Moirai     │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   数据层                          │  │
                    │  │   • YAML清单 + 数据溯源           │  │
                    │  │   • 图拓扑支持                    │  │
                    │  │   • 多模态数据加载器              │  │
                    │  │   • 实体元数据管理                │  │
                    │  └───────────┬─────────────────────┘  │
                    │              │                         │
                    │  ┌───────────▼─────────────────────┐  │
                    │  │   运行时层                        │  │
                    │  │   • 状态机实验管理                │  │
                    │  │   • Ray Tune超参优化              │  │
                    │  │   • Slurm集群调度                 │  │
                    │  │   • MLflow/W&B日志                │  │
                    │  └─────────────────────────────────┘  │
                    │                                       │
                    │  ┌─────────────────────────────────┐  │
                    │  │   插件层（垂直领域）              │  │
                    │  │   • 智慧地球/水文                 │  │
                    │  │   • 医疗健康                      │  │
                    │  │   • 金融                          │  │
                    │  │   • 智慧工程                      │  │
                    │  │   • 计算化学                      │  │
                    │  └─────────────────────────────────┘  │
                    └───────────────────────────────────────┘
```

### 3.2 已实现的实体标识符系统

- **模式**：`none`、`embedding`、`onehot`、`coordinates`、`sinusoidal`、`random`、`numeric_id`、`descriptors`
- **包装器**：`EntityWrapper`（逐实体）、`ChannelEntityWrapper`（逐通道学习型）、`ChannelTransparentWrapper`（逐通道预计算型）
- **实验编排**：48任务矩阵运行器、Ray Tune超参优化、Slurm调度、对比报告生成

### 3.3 已实现的空间模块

- 可配置注入点的实体嵌入
- 1-hop邻居聚合
- GNN模块（已集成至ICPR 2026关于瑞士河流水温预测的投稿）

---

## 4. 研究计划：多阶段规划

### A阶段：实体感知预测（第1–5个月，3篇论文）

---

### 第一周期：系统性实体标识符消融研究（第1个月）

**论文标题**："实体标识符何时以及如何发挥作用？多实体时间序列预测中跨架构的系统研究"

**目标会议**：NeurIPS 2026 D&B / AAAI 2027

#### 目标
1. 首次在7+数据集上系统比较6+标识符类型和6+架构家族
2. 实证分析标识符何时有效（数据集特征、实体数量、相关结构）
3. 实用的标识符选择指导方针
4. 开源基准框架（LIULIAN）

#### 实验设计

**阶段1A——核心消融（第1–2周）**：

| 维度 | 取值 |
|---|---|
| 数据集 | swiss-river-1990, traffic, electricity, PEMS04, PEMS08 |
| 模型 | LSTM, PatchTST, DLinear |
| 标识符模式 | none, embedding, onehot, sinusoidal, random, coordinates（仅Swiss） |
| 种子 | 每组合3个 |
| 预测长度 | 96, 192, 336, 720 |
| 指标 | MSE, MAE（主要）；RMSE, NSE（次要） |

总计约900次运行。

**阶段1B——扩展架构（第2–3周）**：
新增：iTransformer、TimeMixer、TimeXer。
新增负控制：Weather、ETTh1（异构通道——标识符不应有帮助）。

**阶段1C——分析与撰写（第3–4周）**：
- Wilcoxon符号秩检验
- 学习嵌入的t-SNE/UMAP可视化，以地理位置着色
- 嵌入相似度与空间距离分析
- 数据集特征化：实体间互信息、实体数量规模分析

#### 预期关键发现
- 学习嵌入在大规模同质数据集上占优（traffic N=862，electricity N=321）
- 随机基线表现出令人惊讶的良好效果→**实体区分 >> 表示质量**
- 独热编码在大N时退化（862通道非常稀疏）
- 负控制（Weather、ETT）无收益→通道是异构变量，非实体

#### 新颖性验证
- **跨架构的系统性标识符比较**：未发表。STID [9]仅测试空间身份嵌入。CARD [5]仅测试通道token。iTransformer [4]仅测试变量即token。没有工作系统比较none/embedding/onehot/sinusoidal/random/coordinates跨LSTM/Transformer/Linear家族。**新颖。**
- **随机嵌入基线**：NLP中使用（随机词嵌入），但从未作为时序实体标识符的控制消融。**时序领域新颖。**
- **ChannelTransparentWrapper**：无先前工作在multi_channel模式下为TSLib模型注入预计算特征。**新颖架构。**
- **负控制数据集**：先前工作未区分同质实体 vs 异构特征数据集进行标识符分析。**新颖框架。**

#### 贡献
- C1：首个跨架构的实体标识符综合基准
- C2：基于数据集属性的标识符选择实证指南
- C3：开源LIULIAN实体标识符框架
- C4：新颖的瑞士河流水文数据集作为评估领域

---

### 第二周期：基于元学习的自适应实体标识符选择（第2–3个月）

**论文标题**："AdaptID：基于元学习的多实体时间序列预测自适应实体标识符选择"

**目标会议**：ICML 2027 / NeurIPS 2027

#### 技术方案

**多视图实体表示与门控**：
```
x_entity = Gate([emb(id), onehot(id), sinusoidal(id), random(id), coords(id)])
```

- **Gate**：标识符视图上的注意力机制，以数据集统计量（N_entities、实体间互信息、时间自相关）为条件
- **训练**：与预测目标端到端训练 + 辅助标识符判别损失
- **架构**：标识符编码器 → 数据集分析器 → 门控网络 → 预测骨干

#### 新颖性验证
- **TESTAM** [22]使用MoE对图结构类型（无图、静态图、动态图）进行门控，而非标识符类型。输入和粒度不同。
- **TRA** [13]对股票进行实体特定的时间模式路由，而非标识符类型选择。
- **CARD** [5]使用固定通道token，无自适应选择。
- **以数据集统计量为条件的多标识符表示门控**：**未发表。新颖。**

#### 贡献
- C1：首个面向时间序列的自适应标识符选择框架
- C2：元学习形式化，实现选择策略的跨数据集迁移
- C3：实体相关结构 ↔ 最优标识符类型的理论分析
- C4：标识符融合持续优于最佳单一标识符

---

### 第三周期：统一实体标识符 + 时空图（第3–5个月）

**论文标题**："超越节点嵌入：时空预测中实体标识与图神经网络的桥梁"

**目标会议**：KDD 2027 / AAAI 2027

#### 技术方案

**关键洞见**：实体标识符（第一、二周期）是无边的GNN节点特征的特例。本工作将其统一：

1. **标识符到图**：从实体表示学习自适应邻接矩阵（参照Graph WaveNet [15]、MTGNN [16]）
2. **图到标识符**：从图结构提取实体级特征作为标识符
3. **混合**：标识符 + 自适应图 + 消息传递

**实体关系构建**：
- 从数据：从时间模式学习成对实体相似度
- 从元数据：地理坐标、网络拓扑
- 从LLM：生成实体关系描述 → 图结构（**新颖**）

**零样本实体预测**（受STEP [17]启发）：
- 在源实体上预训练实体嵌入
- 通过空间插值、LLM描述映射或少样本适配迁移至未见实体

#### 新颖性验证
- **STID** [9]证明标识符可替代图，但未研究两者组合。
- **STEP** [17]进行零样本节点预训练，但仅限同一领域。
- **基于LLM的元数据实体图构建**：**未发表。新颖。**
- **连接实体标识符和GNN节点嵌入的统一框架**：**未发表为系统研究。新颖。**

---

### B阶段：时空基础模型（第5–12个月，3篇论文）

---

### 第四周期：LLM增强的时空智能（第5–8个月）

**论文标题**："EntityLLM：面向通用时空预测的大语言模型增强实体表示"

**目标会议**：ICLR 2028 / NeurIPS 2027

#### 技术方案

**文本增强的实体标识符**：
```
实体："Aare-Bern站：Aare河上的河流测量站，
       坐标(46.95, 7.45)，海拔502m，集水面积3000 km²，
       位于Thun湖上游，每小时测量水温。"
→ LLM编码器 → entity_text_emb → 与entity_learned_emb对齐
```

**多模态实体表示**：
- 融合：时序模式 + 文本描述 + 空间坐标 + 卫星影像（水文领域）
- 基于注意力的跨模态融合
- 每种模态提供互补的实体信息

**面向ST预测流水线的LLM智能体**：
- 分析数据集元数据 → 推荐标识符类型、架构、图结构
- 迭代式：运行 → 分析 → 优化
- 自然语言实体规范用于零样本适配

#### 新颖性验证
- **Time-LLM** [43]：为时序重编程LLM但无实体特定文本。
- **TEST** [44]：对齐时序与文本原型，但非实体元数据描述。
- **UrbanGPT** [52]：使用LLM进行ST但不使用实体元数据作为文本提示。
- **传感器实体的文本描述作为LLM提示进行标识符注入**：**未发表。新颖。**
- **面向ST流水线设计的LLM智能体**：**未发表。** TrafficGPT [54]最接近但限于交通 + ChatGPT封装。**ST领域新颖。**
- **多模态实体融合（时序 + 文本 + 坐标 + 卫星）**：**未以此组合发表。新颖。**

---

### 第五周期：带实体感知的时空基础模型（第7–10个月）

**论文标题**："UniEntity：面向跨域时空预测的实体感知基础模型"

**目标会议**：ICML 2028 / Nature Machine Intelligence

#### 技术方案

**领域无关的实体框架**：
- 实体类型本体：传感器、患者、金融工具、气象站、河流测量站……
- 跨域共享的实体嵌入空间
- 领域特定的实体适配器（LoRA风格 [110]）

**预训练策略**：
- LOTSA风格 [35]多域ST语料：交通 + 电力 + 水文 + 空气质量 + 临床
- 实体条件化掩码建模：根据实体身份预测掩码补丁
- 空间感知注意力：结合时序自注意力与学习的空间注意力

**跨域实体迁移**：
- 展示：交通传感器嵌入迁移到河流测量站嵌入（两者都是具有周期模式的空间传感器网络）
- 患者作为实体：ICU时序以患者人口统计学作为实体元数据

| 领域 | 实体类型 | 实体元数据 | 关键挑战 |
|---|---|---|---|
| 交通 | 交通传感器 | 位置、道路类型、限速 | 周期模式、空间相关 |
| 能源 | 电力表 | 客户类型、位置、容量 | 多样消费、峰值需求 |
| 水文 | 河流测量站 | 坐标、集水面积、海拔 | 网络拓扑、极端事件 |
| 医疗 | ICU患者 | 人口统计、诊断、用药 | 高变异、缺失数据、不规则采样 |
| 物联网 | 建筑传感器 | 楼层、房间、传感器类型 | 异构类型、边缘部署 |
| 金融 | 股票/金融工具 | 行业、市值、国家 | 非平稳、体制变化 |
| 环境 | 空气质量监测器 | 位置、与道路/工业的距离 | 空间扩散、气象耦合 |

#### 新颖性验证
- **Moirai** [35]：任意变量但无实体条件化或空间感知。
- **UniST** [31]：ST基础模型但仅限交通/城市。
- **OpenCity** [32]：ST预训练但仅限交通。
- **实体条件化的跨域ST基础模型**：**未发表。新颖。** 最接近：UniTS [12]使用数据集特定提示token，但非实体身份条件化。
- **面向ST基础模型的LoRA风格领域适配器**：**在ST语境中未发表。新颖。**

---

### 第六周期：面向高效时空学习的Mamba/SSM架构（第9–12个月）

**论文标题**："ST-Mamba：基于选择性状态空间模型和实体感知条件化的高效时空预测"

**目标会议**：ICLR 2028 / ICML 2028

#### 技术方案

基于SpoT-Mamba [28]和STG-Mamba [29]，开发完整的ST-Mamba架构：

1. **空间Mamba**：沿空间邻域的选择性扫描（图引导游走序列）
2. **时间Mamba**：沿时间维度的选择性扫描，支持变长上下文
3. **实体条件化**：将实体标识符注入Mamba的选择机制（Δ, B, C矩阵）
4. **线性复杂度**：O(N·T) vs 全注意力的O(N²·T²)——对1000+实体网络扩展至关重要

#### 新颖性验证
- **SpoT-Mamba** [28]：用Mamba处理ST但为研讨会论文，范围有限。
- **STG-Mamba** [29]：arXiv预印本，首次探索但无实体条件化。
- **带完整空间+时间扫描的实体条件化Mamba**：**未发表。新颖。**
- **Mamba vs Transformer vs GNN面向ST预测的系统比较**：**未发表为系统研究。新颖。**

---

### C阶段：LIULIAN平台与领域应用（第8–18个月）

---

### 第七周期：LIULIAN平台系统论文（第8–14个月）

**论文标题**："LIULIAN：面向时空智能的开源智能体驱动平台，配备垂直领域插件"

**目标会议**：VLDB 2028 / KDD 2028 (Demo) / JMLR MLOSS / NeurIPS 2027 D&B

#### 平台作为贡献

LIULIAN不仅仅是一个模型库——它是一个**全栈时空智能平台**，结合：

1. **模块化ML流水线**：
   - 任务驱动设计，将任务语义与模型逻辑分离
   - ExecutableModel抽象基类与轻量级适配器模式
   - 状态机实验编排（训练 → 评估 → 推理 → 分析 → 可视化）
   - YAML清单，带语义模式、拓扑、完整性哈希

2. **实体感知和空间模块**（来自A–B阶段）：
   - 跨所有模型家族的实体标识符注入
   - 自适应图构建
   - GNN空间模块
   - 跨域实体迁移

3. **LLM智能体层**：
   - **流水线设计智能体**：分析数据集元数据，推荐模型/标识符/图结构
   - **数据摄入智能体**：基于自然语言的自适应数据映射（"一键图数据摄入"）
   - **报告生成智能体**：从实验结果自动生成分析报告
   - **人在回路交互**：领域专家用自然语言描述问题，系统推荐解决方案

4. **BI与可视化层**：
   - 交互式仪表盘（Power BI集成 + 自定义扩展）
   - 面向时空数据的自然语言查询界面
   - 自动化报告生成，含图表和洞见
   - 图可解释性可视化

5. **垂直领域插件**：

| 领域 | 插件 | 状态 | 关键特性 |
|---|---|---|---|
| **智慧地球/水文** | `plugins/hydrology/` | 部分实现 | 瑞士河流数据库、站点坐标、流域拓扑、GIS可视化 |
| **医疗健康** | `plugins/healthcare/` | 规划中 | MIMIC-III/IV集成、患者作为实体、围手术期预测 |
| **金融** | `plugins/finance/` | 规划中（N-Banker合作） | 金融时序分析、实体感知股票预测、AI智能体聊天机器人 |
| **智慧工程** | `plugins/engineering/` | 规划中（DeepSchemata） | 工程图纸拓扑提取、自动化分析 |
| **计算化学** | `plugins/chemistry/` | 规划中 | 分子属性预测、反应分析 |
| **灾害管理** | `plugins/disaster/` | 规划中（人大合作） | 多模态（卫星+人口）、疏散路径规划 |
| **数字人文** | `plugins/humanities/` | 规划中 | 文档结构分析、手写识别 |

6. **部署与MLOps**：
   - Docker容器化
   - 云部署（AWS规划中）
   - 通过GitHub Actions的CI/CD
   - Ray Tune + Slurm集成
   - 实验追踪（MLflow/W&B）

#### 新颖性验证（平台）
- **TSLib** [73]、**BasicTS** [74]、**tsl** [75]：仅模型库——无智能体、无BI、无插件、无实体感知。
- **LibCity** [79]：仅城市计算——无跨域、无智能体、无BI。
- **GluonTS** [3]、**NeuralForecast** [76]：仅时序——无空间、无图、无实体。
- **Ludwig** [111]：声明式ML但非ST专用。
- **集成实体感知模型 + LLM智能体 + BI + 垂直领域插件的全栈ST平台**：**未发表。新颖。**

---

### 领域应用论文（第10–18个月，3+篇论文）

这些论文将LIULIAN应用于特定领域，各产出一项领域贡献：

#### D1：瑞士河流水温预测（第10–12个月）
- 扩展ICPR 2026投稿，加入实体标识符消融 + 图模块
- 使用NSE指标和水文基线的领域特定评估
- **目标**：Environmental Data Science / Water Resources Research

#### D2：围手术期患者结局预测（第12–15个月）
- 与中国药科大学和伯尔尼大学医院合作
- 患者作为实体，以人口统计学、手术类型、合并症作为实体元数据
- 实体聚类揭示具有不同结局轨迹的患者表型
- **目标**：npj Digital Medicine / CHIL / ML4H

#### D3：多模态时空数据的灾害影响分析（第14–18个月）
- 与中国人民大学合作
- 多模态：卫星影像 + 人口分布 + 时间序列
- 任务：人口分布预测、损害评估、疏散规划
- **目标**：Nature Communications / KDD Applied Data Science

---

## 5. 创新亮点与新颖性总结

### 5.1 近期创新（A阶段，第1–3周期）

| 创新 | 新颖性状态 | 最接近的先前工作 | 新在何处 |
|---|---|---|---|
| 跨架构的系统性标识符消融 | **新颖** | STID [9]（仅一种类型） | 首次多类型、多架构比较 |
| 用于实体区分的随机嵌入基线 | **时序领域新颖** | NLP随机基线 | 隔离"区分"与"表示质量" |
| ChannelTransparentWrapper | **新颖架构** | 无 | multi-channel TSLib模型中的预计算特征 |
| 标识符研究的负控制数据集 | **新颖框架** | 无 | 首次明确测试标识符不应有帮助的场景 |
| AdaptID标识符类型元学习 | **新颖** | TESTAM [22]（图类型） | 对标识符类型而非图结构进行门控 |
| 基于LLM的实体图构建 | **新颖** | NLP中的LLM知识图谱 | 应用于ST传感器网络拓扑 |
| 统一标识符 + GNN框架 | **新颖** | STID [9]、GWN [15]各自独立 | 系统性桥接两个研究社区 |

### 5.2 中期创新（B阶段，第4–6周期）

| 创新 | 新颖性状态 | 最接近的先前工作 | 新在何处 |
|---|---|---|---|
| 通过LLM的文本增强实体标识符 | **新颖** | TEST [44]（文本原型） | 实体特定的元数据描述作为标识符 |
| 面向ST流水线设计的LLM智能体 | **新颖** | TrafficGPT [54]、AI Scientist [57] | 领域特定的ST流水线自动化 |
| 多模态实体融合（时序+文本+坐标+卫星） | **新颖** | Time-MMD [117]（仅时序+文本） | 完整的四模态实体表示融合 |
| 实体感知的跨域ST基础模型 | **新颖** | UniST [31]（仅城市）、Moirai [35]（无实体） | 实体条件化 + 跨域预训练 |
| 面向ST的实体条件化Mamba | **新颖** | SpoT-Mamba [28]（无实体条件化） | 在SSM选择机制中注入实体标识符 |

### 5.3 长期创新（C阶段，第7周期+）

| 创新 | 新颖性状态 | 最接近的先前工作 | 新在何处 |
|---|---|---|---|
| 全栈ST智能平台 | **新颖** | TSLib [73]、LibCity [79]（仅模型库） | 智能体 + BI + 实体感知 + 插件架构 |
| 面向ST数据的LLM驱动自适应数据摄入 | **新颖** | Data-Copilot [55]（仅表格） | 基于NL的ST数据映射，含图拓扑 |
| 带领域特定GUI的垂直领域插件 | **ST领域新颖** | Hugging Face Spaces（仅NLP） | 水文、医疗等领域特定GUI |
| 通过实体感知ST模型的围手术期患者聚类 | **新颖应用** | 围手术期GNN [84]（单任务） | 实体标识符框架应用于临床时序 |

---

## 6. 评估协议

### 6.1 数据集

**主要——同质多实体（A–B阶段）**：

| 数据集 | 实体数 | 长度 | 领域 | 来源 |
|---|---|---|---|---|
| Traffic | 862 | 17,544 | 道路传感器 | TSLib / Caltrans PEMS |
| Electricity | 321 | 26,304 | 电力客户 | TSLib / UCI |
| Swiss-River-1990 | ~50 | ~10,000 | 测量站 | LIULIAN（新颖） |
| PEMS03/04/07/08 | 170–883 | 16,992–26,208 | 交通传感器 | STGNN基准 |
| Solar-Energy | 137 | ~52,000 | 光伏电站 | NREL |
| METR-LA | 207 | ~34,272 | 交通速度 | DCRNN基准 |
| PEMS-BAY | 325 | ~52,116 | 交通速度 | DCRNN基准 |

**负控制——异构通道**：

| 数据集 | 通道数 | 为何为负控制 | 来源 |
|---|---|---|---|
| Weather | 21 | 不同物理变量 | TSLib / Max Planck |
| ETTh1/ETTm1 | 7 | 异构测量 | TSLib |
| ILI | 7 | 不同统计指标 | CDC |

**扩展领域（B–C阶段）**：

| 数据集 | 领域 | 实体类型 | 来源 |
|---|---|---|---|
| MIMIC-III/IV | ICU患者 | 患者 | PhysioNet |
| 北京空气质量 | 环境 | 监测站 | UCI |
| 纽约出租车/共享单车 | 城市出行 | 区域/站点 | NYC TLC / Citi Bike |
| 伦敦智能电表 | 能源 | 家庭 | UK Power Networks |
| 围手术期（私有） | 医疗 | 患者 | 中国药科大学 / Inselspital |

### 6.2 评估指标

- **主要**：MSE、MAE（标准TSLib协议）
- **次要**：RMSE、MAPE、NSE（水文特定）
- **统计**：3+种子的均值 ± 标准差；Wilcoxon符号秩检验；Nemenyi CD图
- **概率**（适用时）：CRPS、校准、区间覆盖

### 6.3 基线方法

| 类别 | 方法 |
|---|---|
| 无标识符 | 共享模型，所有架构 |
| 各标识符类型 | embedding, onehot, sinusoidal, random, coordinates |
| 隐式实体 | iTransformer [4]（变量即token） |
| 显式实体（简单） | STID [9]（MLP + 空间ID） |
| 通道token | CARD [5] |
| 基于图 | Graph WaveNet [15]、MTGNN [16]、DCRNN [14] |
| 基础模型 | Chronos [34]、Moirai [35]、TimesFM [33] |
| 基于LLM | Time-LLM [43]、GPT4TS [42] |

---

## 7. 资源需求

| 资源 | A阶段（第1–5月） | B阶段（第5–12月） | C阶段（第8–18月） |
|---|---|---|---|
| GPU小时 | ~1,500 | ~5,000 | ~3,000 |
| 存储 | 100 GB | 500 GB | 1 TB |
| 人力投入 | 5个月（AI辅助） | 7个月 | 10个月 |
| 计算资源 | RTX 4090集群 | A100集群 | A100/H100集群 |
| LLM API费用 | 最少 | ~$500/月 | ~$1,000/月 |

---

## 8. 风险分析与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|---|---|---|---|
| 实体标识符在大多数数据集上收益甚微 | 中 | 高 | 定位为负面结果；聚焦"何时"而非"是否" |
| 随机基线匹配学习嵌入 | 中 | 中 | 重要发现——实体区分 >> 表示质量 |
| LLM实体描述未能优于数值嵌入 | 中 | 中 | 聚焦零样本/迁移场景，文本在此有独特价值 |
| ST基础模型无法跨域泛化 | 中 | 高 | 从相似领域开始（交通→水文）；使用领域适配器 |
| 平台开发耗时超预期 | 高 | 中 | 优先研究论文；平台增量开发 |
| 合作方数据集未能按时获取 | 中 | 中 | 使用公开数据集作为备选；领域论文调整时间线 |
| Tan等 [51]的批评适用于我们的LLM方法 | 中 | 中 | 设计控制消融；聚焦实体特定贡献 |

---

## 9. 合作网络

| 合作方 | 领域 | 状态 | 角色 |
|---|---|---|---|
| 伯尔尼大学（PRG） | 水文、图ML | 活跃 | 瑞士河流数据、图匹配 |
| 伯尔尼应用科学大学（BFH） | 水文、泥石流检测 | 活跃 | 领域专长、私有数据 |
| 中国药科大学 | 医疗（围手术期） | 建立联系中 | 临床数据集、领域专长 |
| 伯尔尼大学医院（Inselspital） | 医疗（围手术期） | 建立联系中 | 临床数据、验证 |
| 中国人民大学 | 灾害管理 | 已建立联系 | 多模态地震数据集 |
| 数字金融服务研究中心 | 金融（N-Banker） | 紧密联系 | 金融AI智能体系统 |
| 西瑞士应用科学大学（HES-SO） | 文档分析、工程 | 前工作机构 | 历史文档、工程图纸 |
| 中国科学院工程热物理研究所 | 材料科学 | 已建立联系 | 熔池动力学数据 |

---

## 10. 第一周期详细执行计划（1个月）

### 第1周：基础设施与核心实验
- [ ] 验证所有48个矩阵任务通过空运行
- [ ] 将iTransformer、TimeMixer、TimeXer添加到实验矩阵
- [ ] 添加PEMS04、PEMS08和负控制数据集
- [ ] 将默认种子数增加到3个（2024、2025、2026）
- [ ] 设置预测长度扫描（96、192、336、720）
- [ ] 向集群提交阶段1A核心矩阵
- [ ] 监控并修复流水线问题

### 第2周：全面实验运行
- [ ] 验证阶段1A结果；重新运行失败实验
- [ ] 提交扩展模型实验
- [ ] 提交负控制（Weather、ETTh1 × 所有模型 × none+embedding）
- [ ] 计算实体间相关矩阵
- [ ] 运行学习嵌入的t-SNE/UMAP可视化

### 第3周：分析与图表
- [ ] 计算跨种子的均值 ± 标准差
- [ ] Wilcoxon符号秩检验：每种模式 vs none
- [ ] 性能提升热图（数据集 × 模型 × 模式）
- [ ] 实体数量对标识符收益的影响
- [ ] 标识符类型推荐流程图

### 第4周：论文撰写
- [ ] 引言，对标PatchTST CI争论的定位
- [ ] 相关工作（综合版，源自第2节）
- [ ] 方法论（LIULIAN框架、标识符类型、注入机制）
- [ ] 结果，含表格、图表、消融研究
- [ ] 实用指导方针部分
- [ ] 摘要、结论、补充材料
- [ ] 提交至arXiv；准备特定会议投稿

---

## 11. 全阶段研究贡献总结

| 周期 | 标题 | 类型 | 创新度 | 目标会议 | 时间线 |
|---|---|---|---|---|---|
| **1** | 实体标识符系统消融 | 基准 | **高** | NeurIPS D&B / AAAI | 第1月 |
| **2** | AdaptID：自适应标识符选择 | 方法 | **非常高** | ICML / NeurIPS | 第2–3月 |
| **3** | 统一标识符 + GNN框架 | 框架 | **高** | KDD / AAAI | 第3–5月 |
| **4** | EntityLLM：LLM增强实体表示 | 愿景+方法 | **非常高** | ICLR / NeurIPS | 第5–8月 |
| **5** | UniEntity：实体感知ST基础模型 | 基础 | **极高** | ICML / Nature MI | 第7–10月 |
| **6** | 带实体条件化的ST-Mamba | 方法 | **高** | ICLR / ICML | 第9–12月 |
| **7** | LIULIAN平台系统论文 | 系统 | **非常高** | VLDB / KDD Demo | 第8–14月 |
| **D1** | 瑞士河流水温 | 领域 | 中高 | Env. Data Science | 第10–12月 |
| **D2** | 围手术期患者预测 | 领域 | 高 | npj Digital Med. | 第12–15月 |
| **D3** | 灾害多模态ST分析 | 领域 | 高 | Nature Comms. / KDD | 第14–18月 |

---

## 12. 长期愿景（第2年+）

在核心18个月计划之后，LIULIAN旨在成为**时空智能的标准开源平台**，类似于Hugging Face Transformers之于NLP：

1. **社区驱动的模型库**：研究者通过适配器模式贡献ST模型
2. **基准即服务**：跨域标准化评估（回应TSLib [73] / TFB [120]的批评）
3. **智能体市场**：面向不同垂直领域的领域特定LLM智能体
4. **Innosuisse创新项目**：面向产业化部署的资助申请
5. **教育工具**：面向时空ML的交互式教程（基于MkDocs，已部分实现）

---

## 参考文献

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
[127] Nguyen, P.C.H., et al. "PARCv2: Physics-aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics." ICML 2024.

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
                    │  text ─────── LLM(描述) → 投影        │
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
              ┌──────────────────────────────────┐
              │       空间模块（可选）             │
              │  • 1-hop邻居聚合                  │
              │  • GNN消息传递                     │
              │  • 自适应图学习                    │
              │  • Mamba空间扫描                   │
              └──────────┬───────────────────────┘
                         ↓
              ┌──────────────────────────────────┐
              │          预测模型                  │
              │  LSTM / PatchTST / DLinear /       │
              │  iTransformer / TimeMixer / Mamba / │
              │  Time-LLM / Chronos / Moirai       │
              └────────────────────────────────────┘
```

## 附录B：代码制品

| 文件 | 描述 |
|---|---|
| `experiments/entity_identifier/run.py` | 矩阵实验运行器 |
| `experiments/entity_identifier/matrix.py` | 矩阵定义和任务扩展 |
| `experiments/entity_identifier/compare.py` | 对比报告生成器 |
| `experiments/entity_identifier/submit_slurm.py` | Slurm集群调度器 |
| `liulian/models/torch/entity_mixin.py` | EntityWrapper, ChannelEntityWrapper, ChannelTransparentWrapper |
| `liulian/data/ts/timeseriesdataset.py` | 实体特征和透明模式数据层 |
| `experiments/run.py` | 统一实验入口点 |
| `tests/runtime/test_entity_identifier_pipeline.py` | 流水线测试（15个测试） |

## 附录C：新颖性验证总结

| # | 提议的想法 | 是否已发表？ | 最接近的工作 | 新在何处 |
|---|---|---|---|---|
| 1 | 跨架构的系统性多类型标识符消融 | **否** | STID [9] | 多类型、多架构系统研究 |
| 2 | 时序实体标识的随机嵌入基线 | **否（时序领域）** | NLP随机基线 | 时序领域首次控制消融 |
| 3 | ChannelTransparentWrapper架构 | **否** | — | multi-channel TSLib中的预计算特征 |
| 4 | 标识符分析的负控制数据集 | **否** | — | 明确测试标识符不应有效的场景 |
| 5 | AdaptID：标识符类型门控 | **否** | TESTAM [22]（图类型） | 对标识符类型而非图结构门控 |
| 6 | 基于LLM的元数据实体图构建 | **否** | NLP的LLM知识图谱 | 应用于ST传感器网络拓扑 |
| 7 | 统一标识符 + GNN框架 | **否** | STID [9]、GWN [15]各自独立 | 系统桥接两个研究社区 |
| 8 | 实体文本描述作为LLM提示 | **否** | TEST [44]（原型） | 实体特定元数据，非通用原型 |
| 9 | 面向ST流水线设计的LLM智能体 | **否** | TrafficGPT [54]、AI Scientist [57] | ST领域特定的流水线自动化 |
| 10 | 多模态实体融合（时序+文本+坐标+卫星） | **否** | Time-MMD [117]（时序+文本） | 完整四模态实体表示融合 |
| 11 | 实体感知的跨域ST基础模型 | **否** | UniST [31]（城市）、Moirai [35]（无实体） | 实体条件化 + 跨域 |
| 12 | 面向ST的实体条件化Mamba | **否** | SpoT-Mamba [28]（无实体条件化） | SSM选择机制中的实体标识符 |
| 13 | 全栈ST平台（智能体+BI+实体+插件） | **否** | TSLib [73]、LibCity [79] | 完整平台含所有组件 |
| 14 | 通过实体感知ST的围手术期患者聚类 | **否（此框架）** | 围手术期GNN [84] | 实体标识符框架用于临床时序 |
| 15 | ST平台的领域特定垂直插件 | **否** | HuggingFace Spaces（仅NLP） | ST特定的领域插件 |
