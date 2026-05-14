---
title: LIULIAN 平台总蓝图（中文镜像）
status: living document（中文执行摘要，权威英文版见 PLATFORM_BLUEPRINT.md）
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
parent: PLATFORM_BLUEPRINT.md
---

# LIULIAN 平台总蓝图（中文版）

> **语言：** [English](PLATFORM_BLUEPRINT.md) | 中文
>
> *本文件是英文权威版的执行摘要级中文镜像；与英文版同步更新；所有 ADR /
> 详尽论证 / 代码示例 / 表格均以英文版为准。*

## 0. 第二轮（2026-05-12）变更要点

相比第一轮蓝图，本轮调整：

1. **多仓而非单仓** — 仿照 liulian（backend + frontend + agent 三个仓 + neoctl 编排），LIULIAN 拆成八个仓：`liulian-python`（核心）+ `liulian-api` + `liulian-web` + `liulian-mobile` + `liulian-agent` + `liulian-ingest` + `liulian-design-system` + `liulian-ops`。见 §4。
2. **自研 agent，不用 LangGraph** — 基于 `liulian-agent` 仓的自定义编排（~300 行），底层 GLM-4.6 / DeepSeek-V4 / Gemini-3.1 / Claude / Ollama / Mock 多 provider 抽象（继承 liulian-agent）。见 §7、ADR 0002。
3. **Tracker 统一覆盖 task + experiment + agent** 三种实体类型，单一 `run` 表 + `parent_kind` 鉴别。见 §8、ADR 0005。
4. **品牌定锚 UniBe 红 `#E20613`**（匹配 `feat/gui-demo` 分支已确认的视觉），暖纸面板上的编辑级 Swiss 美学，非黑底 SaaS 默认。见 §11、PLATFORM_DESIGN.md。
5. **分支策略**：所有平台工作在 `feat/platform-upgrade-2026-05` 上，完成后并回 `main`。
6. **gui-demo 是设计底本** — 分支视觉系统（Fraunces · Switzer · JetBrains Mono · UniBe 红 · 暖纸 · 4 标签 IA Data/Train/Inference/Insight）作为 `liulian-web` 的画布起点。

## 1. 定位（L1）

**一句话定位**：*面向时空 AI 的开源生产栈 — 研究级模型库 · 工业级 BI 画布 · 智能体工作流 · 主权部署。*

**四字版**：*为时间而生的流动智能。*

### 反向定位（我们不是这些）

| 易被混为 | 我们怎么不同 |
|---|---|
| Time-Series-Library (THU-ML) | 他们是模型库；我们是模型库**之上**的任务运行时 + BI + Agent 层（库由他们 + 我们共同策展）。 |
| ClearML / W&B | 语言无关的实验追踪器；我们是 TS/ST 专门产品，原生支持空间数据、BI 画布、领域插件。 |
| Power BI / FineBI | 通用 BI 跑在关系型数据上；我们是 TS/ST 原生 BI，预测带、预测区间、在线再训内置。 |
| HydroForecast | 闭源单垂直 SaaS；我们是开源核心 + 垂直插件（水文 / 医疗 / 能源）。 |
| Ultralytics HUB | 视觉专属；我们时序优先。 |
| liulian | 金融垂直、Java 后端、业务分析师受众；我们是科学垂直、Python 后端、研究工程师受众。借**操作模式**（多仓、neoctl 风格 CLI），不借**领域**。 |

## 2. 受众（L1）

| 人格 | 为何来 | 转化在哪里 |
|---|---|---|
| 开源贡献者（博士 / 研究员） | 新 SOTA 模型上线，要 benchmark | "把你的模型放进 `adapters/`，一行命令跑 5 数据集矩阵" |
| 领域分析师（水文 / 能源交易 / 临床研究） | 手里有 SwissRiver 量级的数据集 | BI 画布 + 预测区间 + 可导报告 |
| ML 平台工程师（评审你的雇主） | 从 CV 链接进来 | README 架构 + Helm chart + GH Actions 通过 + Grafana 截图 + `liulian-ops` CLI 演示 |
| VC 分析师 / 侦察兵 | 推荐人介绍 | 一页产品视频，"瑞士全国 N 座站点已部署"，现场 SwissRiver 演示 URL |
| 医疗 RSE 招聘人（ARTORG-AIHN 等） | 看到求职作品集 | Mobile app + FastAPI swagger + ECG demo + 隐私态势 |

## 3. M6 融资就绪的 L1 成功指标

- 2026-09 前：≥ 1 个付费 pilot（CHF 1k-5k / 月）
- 2026-09 前：≥ 25 月活（最近 28 天跑过 ≥1 次实验）
- 2026-09 前：≥ 100 非朋友 GitHub stars
- 2026-09 前：≥ 1 位顾问入资本表（Mougiakakou / Fischer / Riesen / 瑞士机器学习创业者）
- 2026-11 前：pitch deck + 18 个月财务模型 + 第一份 LOI

## 4. 多仓架构（L2）

详见 ADR 0001。八个仓 + 一个 config（claude-config）：

| 仓 | 栈 | 用途 |
|---|---|---|
| **`liulian-python`** (本仓) | Python 3.10+, numpy, pyyaml, torch (extra) | **仅研究核心**：tasks · data · models · adapters · runtime · optim · viz · plugins。保持 `pip install liulian` 干净。 |
| **`liulian-api`** | Python, FastAPI, Pydantic v2, SQLModel, Postgres-TimescaleDB, Redis, arq | HTTP 网关层 |
| **`liulian-agent`** | Python, 自研 orchestrator, DeepSeek/GLM/Gemini, pgvector | 独立 LLM agent 服务（端口 8000，FastAPI，`/health`，仿 liulian-agent） |
| **`liulian-ingest`** | Python, async httpx, playwright | 数据爬取（swisstopo BAFU 水文、SwissGrid 能源、PhysioNet 医疗等） |
| **`liulian-web`** | Next.js 14 (App Router), tRPC, Tailwind, shadcn/ui, Tremor, ECharts, MapLibre, @ant-design/x | BI 画布 + Studio + Marketing |
| **`liulian-mobile`** | Expo SDK 51, RN, Expo Router, Victory Native XL | 移动伴侣 |
| **`liulian-design-system`** | 设计 tokens（JSON + CSS vars + Tailwind 预设 + RN StyleSheet + antd ConfigProvider） | npm `@liulian/design-tokens` |
| **`liulian-ops`** | Python CLI（仿 neoctl）+ Helm + Terraform + 复用 GH Actions | 部署编排（`liulianctl`） |

详细文件级 fork 计划见 `LIULIAN_REUSE_MAP.md`；多仓诚实理由见 §4.2（已经过 research-critic 审查）。

### 分支工作流

1. 所有平台改动在 `feat/platform-upgrade-2026-05`。
2. 周期性 `git fetch origin main && git merge --no-ff main` 拉入并行 main 工作。
3. `feat/gui-demo` 孤儿分支保留，`liulian-web` 通过 copy 而非 git 关系导入它的视觉底本。
4. Sprint Day 7 合并回 `main`，打标 `v0.6.0-portfolio`。

## 5. 后端（`liulian-api`）

详见 §5。要点：

- FastAPI + Pydantic v2 + SQLModel + PostgreSQL（**Day 1 用纯 Postgres，M1 demo 上线后再开 TimescaleDB extension**，见 ADR 0003）
- Redis + arq workers
- Auth: Clerk（demo），OAuth2 + 多租户 RBAC（v2）
- 模型在线推理：Ray Serve（HPO 已用 Ray）
- 观测：OpenTelemetry → Prometheus + Grafana + Tempo + Loki

### 5.1 为什么 TimescaleDB（去标语化）

`run_metric` / `forecast` / `alert` 表是仅追加的时间键流。M2 内将达百万行级别。两条候选路径：

1. **纯 Postgres + pg_partman + pg_cron** — 能行，但要自己维护分区计划。
2. **TimescaleDB extension** — hypertable 自动分区，零迁移成本（`CREATE EXTENSION timescaledb`），SQLModel 不变。

选 TimescaleDB 的**主要理由**：消除分区管理代码。"吃狗粮" 是次要叙事。Sprint 务实派：Day 1 用纯 Postgres，extension 在 M1 demo 上线后再开。

### 5.2 API 表面（v1）

`/healthz` `/readyz` `/experiments` `/models` `/datasets` `/forecasts` `/alerts` `/agents/{name}/invoke` `/reports` `/tasks`，所有 OpenAPI 自动生成，SDK + web + mobile codegen。

### 5.3 存储 schema（关键表）

详见英文版 §5.3。要点：tracker 表用 `run` 单表 + `parent_kind` 鉴别（实验 / 任务 / agent，见 ADR 0005）；`run_metric` 是 TimescaleDB hypertable；`agent_run_step` 记录每步 LLM 调用 + token 成本。

## 6. 前端（`liulian-web`）

详见 PLATFORM_DESIGN.md（L3）。栈摘要：

- Next.js 14 (App Router, RSC) + tRPC + openapi-typescript
- Tailwind + CSS 变量 + brand tokens
- shadcn/ui 主，@ant-design/x 仅用于聊天侧边栏，Tremor 用于 KPI（ADR 0010 hybrid 栈）
- ECharts（大数据 + brush + geo）
- MapLibre GL + swisstopo 瓦片
- react-mosaic-component（仿 liulian 的 BI 画布平铺）
- Clerk 鉴权，next-intl 国际化（en · zh-CN · de-CH）
- Vitest + Playwright 测试

视觉底本：`feat/gui-demo` 分支的设计（UniBe 红 + Fraunces + Switzer + JetBrains Mono + 4 标签 IA）。

## 7. Mobile（`liulian-mobile`）

详见 §7。栈：Expo SDK 51 + RN 0.74 + Expo Router + Victory Native XL + Reanimated 3 + react-hook-form + zod + Expo Notifications + Expo Document Picker + Clerk + expo-localization。

**为什么 Expo 而非裸 RN**：单代码库 iOS + Android、OTA、EAS Build（Linux 上不用 Xcode）、Expo Go QR 演示。

三标签 UX：Home（提醒 + 最近运行）· Forecast（单站点带预测带）· Alerts（严重程度色带）。

## 8. Tracker（三合一）

详见 §8 + ADR 0005。

`run` 单表，`parent_kind` 鉴别：`task` / `experiment` / `agent`。统一活动 feed 在 `/studio/activity`。`agent_run_step` 表记录每步 LLM 调用、tokens、cost。`run_metric` 是 hypertable。

兼容 MLflow tracking REST API 子集，让 MLflow 客户端可以透明地切到 LIULIAN。

## 9. Agent 层（自研，`liulian-agent`）

详见 §9 + ADR 0002。

3 个 persona：data / model / bi。每个 persona 有专属工具集（Pydantic 类型）。Orchestrator 是手写的 ~300 行状态机（PLAN → CALL_TOOL → REFLECT），仿 liulian-agent 的 `agent/loop.py`。

LLM provider 矩阵（用户已有 token）：

| Provider | 默认用于 | 价格 / 1M token（2026-05） |
|---|---|---|
| **DeepSeek V4 Flash** | 默认 | $0.14 / $0.28（最便宜的生产级） |
| **DeepSeek V4 Pro** | 重推理任务 | $0.435 / $0.87（促销） |
| **GLM-4.6** | 中文任务 | ~$0.20 / $0.60 |
| **Gemini 3.1 Pro** | 长上下文（>200k）+ 多模态 | $2 / $12 |
| **Gemini 3.1 Flash-Lite** | 工具调用路由 | $0.10 / $0.40 |
| **Ollama + qwen2.5-7b** | 主权 / 离线部署 | $0 + 硬件 |

provider 抽象：仿 liulian `llm/gateway.py` 的多区域 chain；同样的 `Gemini blocked in CN` fallback 模式。

安全：默认 provider 在 `LIULIAN_OFFLINE=1` 时切到 Ollama；原始 signal 数组绝不发给云 LLM（只发摘要统计）；每次运行有 USD 上限。

## 10. 数据接入（`liulian-ingest`）

详见 §10。仿 `liulian-ingest`。

源适配器：swisstopo-bafu / meteoswiss-precip / swissgrid / physionet-mit-bih / caltrans-pems / electricity-uci。

写到 MinIO `s3://liulian-raw/{source}/{date}/...` + manifest YAML 自动 PR 到 `liulian-python/manifests/`。

部署：K8s CronJob（生产）或 systemd timer（单 VM staging）。

## 11. 品牌与视觉（L3 — 详细见 PLATFORM_DESIGN.md）

锚色 UniBe 红 `#E20613`。详细 token 集见 PLATFORM_DESIGN.md §2。要点：

- Canvas: `#FBFBFA` 暖纸；绝不 `#FFFFFF`
- Ink: `#131313`；绝不 `#000`
- 字体：Fraunces（serif，wonk-1 italic accent）+ Switzer（geometric sans）+ JetBrains Mono（tabular-nums）
- Inter / Roboto / Open Sans 禁用

视觉禁忌（impeccable + minimalist-ui 双重）：

- 渐变文本
- 1px 以上侧条强调（用 6×6 像素红点替代）
- 大面积主色背景
- 鸡精氛围的 hero metric 模板
- 卡片 grid 千篇一律
- 优先 modal
- 大滴影
- emoji
- AI cliché 用语

详见 `conventions/UI_AUDIT_CHECKLIST.md`（每 PR 必过 46 项）。

## 12. CI/CD 与运维（`liulian-ops`）

详见 §12。每仓有自己的 `ci.yml`（调用 `liulian-ops/.github/workflows/` 复用工作流：python-lint / python-mypy / python-pytest / python-image / js-lint / js-tsc / js-vitest / js-playwright / js-image / release-tag / release-notes）。

`liulianctl` CLI（仿 neoctl）：`bootstrap` / `deploy <service>` / `deploy all` / `logs <service>` / `restart <service>` / `manifest sync` / `tunnel ollama`。

环境分层：dev (docker-compose) → demo (Railway + Vercel) → staging (Hetzner k3s) → prod (EKS/AKS)。

观测预置 Grafana dashboard：API 健康 / 推理性能 / 训练队列 / 预测质量 / Agent cost。

## 13. 隐私与安全

详见 §13。

- 数据驻留：单租户默认部署在客户自有 infra；默认无第三方 LLM
- PII：零；demo 用公开 PhysioNet MIT-BIH + swisstopo
- 本地 LLM 路径：Ollama + qwen 已文档化
- Secrets：1Password Connect / AWS SSM；绝不 `.env` 提交
- 审计：所有写 API 落 audit-log + agent_run_step，留 365 天
- 合规态势：未认证（工程 MVP）；GDPR / 瑞士 FADP / HIPAA 路径已文档化

## 14. 文档结构（4 层）

详见 §14、`LIULIAN_REUSE_MAP.md §11`。

```
docs/strategy/
├── PLATFORM_BLUEPRINT.md       L1-L2: 愿景 + 架构
├── PLATFORM_DESIGN.md           L3: 视觉表面（品牌、BI 8 面板、agent 流、mobile UX）
├── ONE_WEEK_SPRINT.md           L4: 一周执行
├── LIULIAN_REUSE_MAP.md       具体 fork-and-adapt 计划
├── REFERENCE_DESIGNS.md         12 平台调研档案
├── AUDIT_REPORT_<date>.md       审计记录（impeccable + research-critic）
├── adr/                          独立决策记录 0001-0010
├── conventions/                  L0: 规则
│   ├── UI_AUDIT_CHECKLIST.md    每 PR 必过的 46 项
│   └── ...
└── *.zh.md                       双语镜像（en canonical, zh mirror）
```

**双语规则**（采纳自 jajupmochi/claude-config:bilingual-docs）：每个面向人类的 repo 级文档必须 `NAME.md`（英文 canonical）+ `NAME.zh.md`（中文镜像）双文件，顶部带语言切换。

例外：`CLAUDE.md` / `CLAUDE.local.md` / `SKILL.md` / `RULE.md` / 内部研究草稿。

## 15. 六个月路线图（三档时间线）

第一轮假设全职；实际 Linlin 是博士后兼职。三档并行：

| Milestone | 激进档（零滑） | 现实档（一次滑期）| 保守档（多次滑期） | 交付物 |
|---|---|---|---|---|
| **M1: Portfolio-ready** | 2026-05-19 | 2026-05-19 | 2026-05-19 | 演示 URL + 8 仓初始 + ARTORG 投出 |
| **M2: BI 旗舰** | 2026-06-30 | 2026-08-01 | 2026-09 | 8 面板 SwissRiver + Chronos 零样本 + agent v1 |
| **M3: 多租户云** | 2026-07 | 2026-09 | 2026-10 | Helm + Terraform on EKS + Clerk + 状态页 |
| **M4: 垂直试点** | 2026-08 | 2026-10 | 2026-12 | 能源 + 医疗 ECG 两个 case study |
| **M5: Agent autopilot** | 2026-09 | 2026-12 | 2027-Q1 | 夜间异常扫描 + 再训决策 + drift 检测 |
| **M6: Pre-seed** | 2026-11 | 2027-Q1 | 2027-Q2 | Pitch deck + 18 月财务模型 + 第一份 LOI |

Plan-of-record：**现实档**。对外沟通现实档；对内向激进档冲；保守档守底。

## 16. 长期可扩展

加一个新垂直 = 配置，不是代码：

1. 数据：丢一份 manifest 到 `manifests/{vertical}/`（或 `liulian-ingest` 自动生成）
2. 插件：可选加 `plugins/{vertical}/`
3. BI：注册 `vertical.json` 描述默认 panels、地图投影、阈值
4. Agent：领域工具作为 Pydantic 类型可调用注册
5. 部署：Helm values overlay 控制 flags、默认模型、告警通道

需要动 `liulian-python` 的垂直走核心扩展 ADR。

## 17. 融资就绪 — VC 想看的

M2：2 分钟 demo 视频 + 3+ 非朋友 GitHub stars + 第一篇非团队博客
M3：付费 pilot（CHF 1k-5k / 月）— 顺序：swisstopo / WSL / Eawag（水）→ Inselspital / ARTORG-AIHN（医）→ Axpo / BKW（能源）
M4：≥ 25 月活
M5：顾问入资本表
M6：12-slide deck + 18 月财务模型 + 第一份 LOI

叙事：**面向受监管行业的垂直 TS/ST AI** — 角度是"时空原生" + "主权数据驻留"。

## 18. ADRs

详见 `adr/`：

- 0001 多仓拆分
- 0002 自研 agent（非 LangGraph）
- 0003 TimescaleDB（M3 再考虑 TDengine）
- 0004 UniBe 红作锚
- 0005 tracker 三实体单表
- 0006 fork-and-adapt from liulian
- 0007 拒绝 Refine.dev UI；手卷 /studio
- 0008 canvas-orchestrator 复用
- 0009 Spring Boot → FastAPI 模式翻译
- 0010 hybrid shadcn + antd 栈

## 19. 仍未决（跟踪中）

- 模型服务：FastAPI vs BentoML vs Ray Serve（plain FastAPI 到 M2；M3 revisit）
- agent 长期记忆：pgvector vs Qdrant（M5 ≥ 100k embedding 时再评）
- iOS-only vs Android-also 截图（取决于 Mac 可得性；sprint 先 Android）

---

*详细英文权威版：`PLATFORM_BLUEPRINT.md`。L3 视觉深度：`PLATFORM_DESIGN.md`。一周执行：`ONE_WEEK_SPRINT.md`。具体 fork 计划：`LIULIAN_REUSE_MAP.md`。12 平台调研：`REFERENCE_DESIGNS.md`。独立决策记录：`adr/`。*
