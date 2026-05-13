---
title: LIULIAN — 从 neo-banker 的具体复用映射（中文镜像）
status: living document
parent: NEOBANKER_REUSE_MAP.md
---

# LIULIAN — 从 neo-banker 的具体复用映射（中文版）

> **语言：** [English](NEOBANKER_REUSE_MAP.md) | 中文

每个 LIULIAN 仓在有对应 neo-banker 仓的地方都从 **fork** 开始。本文逐文件列出我们复制什么、保留什么、改造什么、删什么。它是第三轮平台计划的真源；blueprint 和 sprint 都依赖它。

## 0.1 神圣规则 — 代码复用 vs 视觉原创（不可妥协）

> **代码 / 架构 / 管道 / 词汇 / 部署模式** — 自由地从 neobanker 与 12 参考平台复用 / 改造 / fork。
>
> **视觉设计 / 品牌语态 / UI 构图 / 图标语汇 / 字型选择 / 微交互** — **原创**。借方向，绝不抄。LIULIAN 必须有 *它自己的* 美学。

具体落地：本文后面引用的 *reuse fractions* 仅适用于**代码行 + 文件结构**。它们 **不** 适用于设计。即使我们复用 `CanvasOrchestrator.tsx`（其逻辑的 90%），每一个可见像素都按 LIULIAN tokens 重新设计，每段文案都重写，每个形状与节奏都由*我们*的品牌决定，锚在 `feat/gui-demo` 编辑级 Swiss canon（§14.5）。

每个 PR 跑 4 项测试（在 `conventions/UI_AUDIT_CHECKLIST.md`）：
- AI-slop 测试：会有人毫不犹豫说"AI 做的"吗？→ 拒
- 类别反射测试：看色板能猜出领域吗？→ 拒
- neobanker-copy 测试：截图像神-banker 换了文案吗？→ 拒；重做到血脉看不到
- gui-demo 交叉校验：暖纸 + UniBe 红做唯一 spot color + 编辑级排版还在做主吗？→ 不然解释为什么

## 0. 为什么 fork-and-adapt 而非绿地

五个 neo-banker 仓（agent / crawler / frontend / neoctl / dev-env）都是**生产级 / 经过测试 / 双语文档化**的解，恰好对应 LIULIAN 需要的同款架构：Python core + 多 provider LLM 网关的 FastAPI agent 服务 + Next.js 带 AI 聊天的前端 + 部署 CLI + 统一 Codespaces dev env。重新推导成本是几周、零增益。Fork + 改造成本是几天、同等生产就绪 *并且* 保留运维肌肉记忆（同样的 CLI 动词、同样的 SSE 事件名、同样的 .env 词汇）。

## 1. 仓到仓 fork 映射

| LIULIAN 新仓 | fork 自 | 分支模式 | 变更集交叉链接 |
|---|---|---|---|
| **`liulian-agent`** | `neo-banker/neobanker-agent` | fork → 改名 → squash → push 到 `liulian-ai/liulian-agent` | §2 |
| **`liulian-ingest`** | `neo-banker/neobanker-crawler` | fork → 改名 → push | §3 |
| **`liulian-web`** | `neo-banker/neobanker-frontend-MVP-V3` | fork → 改名 → push；只重写 brand 层 | §4 |
| **`liulian-ops`** | `neo-banker/neoctl` | fork → 改名 → push | §5 |
| **`liulian-dev-env`** | `neo-banker/neobanker-dev-env` | fork → 改名 → push | §6 |
| **`liulian-api`** | （无；绿地）| 新仓；模式借自 neobanker-backend-MVP-V2（Spring Boot → FastAPI 翻译）| §7 |
| **`liulian-mobile`** | （无；Expo template）| 新仓；视觉 tokens 借自 `liulian-web`；SSE 事件形状借自 `liulian-agent` | §8 |
| **`liulian-python`** | 本仓 | 已存在；仅小整理（包边界、gui-demo 留 orphan branch）| §9 |
| **`liulian-design-system`** | （无）| 新仓；从 `liulian-python/.worktrees/gui-demo/styles/main.css` 起种 | §10 |

加 organisation 级 config 仓：

- `.claude/` per-repo 来自 `jajupmochi/claude-config` 通过 `/init-claude-config`
- 文档规则来自 `claude-config/rules/bilingual-docs/RULE.md`，落为 `docs/strategy/conventions/DOCUMENTATION_RULES.md`

## 2. `liulian-agent` ← `neobanker-agent`

`neobanker-agent` 是端口 8000 的 FastAPI 服务，有 `/health` 和 SSE 流式 `/agent/chat`。架构（见英文版 ASCII 图）：

- `main.py` FastAPI bootstrap + 生命周期 + 路由
- `agent/`：loop / intent / planner / state / provider_registry / provider_policy / conversation_cache / catalog_cache / context_memory / demo_scenarios / reliability / suggestions / error_log
- `llm/`：config / harness / gateway / providers (claude / gemini / glm / ollama / mock)
- `tools/`：db_reader / calculator / web_search / bank_matcher
- `routers/`：chat (SSE) / settings
- `datasource/`：client (CSV / backend / auto)
- `prompts/`：intent / planner / summary / suggestions 模板
- `fixtures/`：demo scenarios + bank-data JSON
- `neobanker-agent.service`：systemd unit
- `mkdocs.yml` + `docs/`：双语文档（10x .md + .en.md 对）

**SSE 事件流**：`thinking → trace → intent → tool_call → tool_result → response → suggestions → done`

**Provider chain 模式**：`GatewayDecision` 持有 region + chain + used/skipped/errors；Gemini 被 CN 区块时静默跳过，按序试 GLM / Claude / Ollama。同模式对 LIULIAN 不变。

### 2.2 文件级计划

**整段保留 / 仅改名**：

| 文件 / 目录 | 状态 | 备注 |
|---|---|---|
| `main.py` | 改引用 | FastAPI 入口 |
| `agent/loop.py` | 保留 | 核心循环 |
| `agent/intent.py` | 保留 | LLM 意图分类 |
| `agent/planner.py` | 保留 | 工具编排 |
| `agent/state.py` | 保留 | 会话状态 |
| `agent/provider_registry.py` | 整段保留 | 通用 |
| `agent/provider_policy.py` | 整段保留 | 通用 |
| `agent/conversation_cache.py` | 保留 | 记忆层 |
| `agent/catalog_cache.py` | 改造 | catalog = manifest 列表 |
| `agent/context_memory.py` | 整段保留 | 通用 |
| `agent/reliability.py` | 保留 | 5 级可信度直接复用 |
| `agent/suggestions.py` | 保留 | 通用 |
| `agent/error_log.py` | 保留 | 通用 |
| `agent/intent_shortcuts.py` | 改造 | shortcuts 改成 forecasting-specific |
| `llm/config.py` | 整段保留 | ProviderSpec |
| `llm/harness.py` | 整段保留 | LLM 调用包装 |
| `llm/gateway.py` | 整段保留 | gateway + region routing |
| `llm/providers/*` | 整段保留 | 5 个 provider |
| `routers/chat.py` | 保留 | SSE chat router |
| `routers/settings.py` | 改造 | settings UI knob 不同 |
| `datasource/client.py` | 改造 | "bank data" → "forecasting catalog" |
| `prompts/intent.py` 等 | 重写 | LIULIAN intent 不同 |
| `neobanker-agent.service` | 改名 | `liulian-agent.service` |
| `mkdocs.yml` | 保留 | docs build |
| `docs/*` | 重写内容，保留结构 | 双语 |

**Tools 整套重写，沿用 registry 模式**：

| neobanker tool | LIULIAN 对应 |
|---|---|
| `db_reader` | `query_forecasts` |
| `calculator` | `compute_metric` |
| `web_search` | 保留 |
| `bank_matcher` | `station_matcher` |
| – | `recommend_model`（新）|
| – | `propose_hpo_space`（新）|
| – | `diagnose_failed_run`（新）|
| – | `add_panel`（新）|
| – | `create_alert_rule`（新）|

### 2.4 新 intents

`forecast_query` / `forecast_compare` / `alert_setup` / `model_recommendation` / `run_diagnosis` + 保留 `conversation` / `data_lookup` / `comparison`

### 2.5 LLM provider 重指向

neobanker-agent 已有 GLM / Gemini / Claude / Ollama / Mock 5 个 provider。我们加 DeepSeek（OpenAI-compat，可能用 GLMProvider 的 base-URL knob 或写 30 行 DeepSeekProvider）。总工作 ≤ 1h。

### 2.6 reuse fraction（Day 1 spike 后定）

代码行原样：**~50 到 70%** — 取决于 `state.py` / `conversation_cache.py` / prompts 之外的银行域耦合深度。
改造：**~25 到 40%**。
新写：**~5 到 15%**。

## 3. `liulian-ingest` ← `neobanker-crawler`

源结构：`src/crawler/` + `config/sources.yaml + data_dictionary.yaml` + `tools/` + `docs/specs/` + systemd unit。

文件级计划：

- `src/crawler/` → 改包名 `liulian_ingest/`，HTTP retry / idempotency / manifest writer 保留
- `config/sources.yaml` → 重写：swisstopo-bafu / meteoswiss-precip / swissgrid / physionet-mit-bih / caltrans-pems / electricity-uci
- `config/data_dictionary.yaml` → 重写 LIULIAN manifest schema（镜像 `liulian-python/manifests/_schema.yaml`）
- `tools/scan_java_columns.py` → 删；无关
- `docs/specs/` → 保留结构，重写为 LIULIAN 数据契约
- `neobanker-crawler.service` → `liulian-ingest.service`
- `pyproject.toml` → 改项目名，保留 dev 工具（uv / ruff / mypy strict / pytest）

新增：manifest auto-PR — 爬虫写 parquet 到 MinIO + 自动 PR 一份 manifest YAML 到 `liulian-python/manifests/{vertical}/`。

reuse fraction（Day 1 spike 后定）：50–80% / 10–30% / 5–15%

## 4. `liulian-web` ← `neobanker-frontend-MVP-V3`

源栈：Next.js 14 + @clerk/nextjs + antd 5.26 + @ant-design/x + antd-style + echarts 5.6 + echarts-for-react + @amcharts/amcharts5 + react-mosaic-component + react-resizable-panels + contentlayer + framer-motion + lucide-react + react-markdown + next-themes + next-intl + storybook + vitest + playwright

### 4.2 hybrid antd + shadcn 决策

`neobanker-frontend-MVP-V3` 全 antd。LIULIAN 品牌是暖纸编辑级 Swiss + UniBe 红，与 antd 企业蓝默认相冲。但 `@ant-design/x` 是 AI 聊天 UI 之冠，`react-mosaic-component` 是平铺仪表盘之冠。

| 层 | 选 | 因 |
|---|---|---|
| Marketing (`/`) + Studio (`/studio`) | **shadcn/ui + Tailwind** | 编辑级品牌；像素级控制 |
| Forecast 画布 frame | **shadcn/ui** | 品牌表面 |
| Forecast 画布**内**面板 | **react-mosaic-component**（来自 neobanker 栈）| 平铺 + 拖拽 + 缩放是 BI 习语 |
| Charts | **echarts-for-react** | 与 neobanker 同 |
| Map | **MapLibre GL**（LIULIAN 加，非 amcharts5）| swisstopo 开源瓦片 + WebGL |
| Chat sidebar | **@ant-design/x** | AI 聊天的特长；antd 通过 `antd-style` 主题改色到品牌 |
| 鉴权 | **@clerk/nextjs** | 同 neobanker |
| i18n | **next-intl** | 同 neobanker；我们也采纳双语规则 |
| MDX 内容 (docs + blog) | **contentlayer** | 同 neobanker |
| Animation | **framer-motion** | 同 |
| Tests | **vitest + playwright** | 同 |
| Storybook | **storybook** | 同 |
| 主题切换器 | **next-themes** | 同 |
| 图标 | **Phosphor Icons**（Regular）；Lucide fallback | |
| 可缩放面板 | **react-resizable-panels** | 站点列表 ↔ 画布 分割 |
| Markdown | **react-markdown** | agent 响应 |

### 4.3 文件级计划

- `app/` → 保留 App Router 结构；路由段从 `/dashboard /products /companies /news` 改为 `/forecast /studio /agents /admin`
- `components/` → 保留 `chat/*` `charts/*` `mosaic/*`；删 `banking/*`
- `config/` → 重写（新环境变量）
- `content/` → 重写（银行新闻 → LIULIAN 发布说明 + 文档）
- `contentlayer.config.js` → 保留架构，改 content types
- `contexts/` → 保留通用，重写领域
- `deploy.sh` / `dev.sh` / `dev-docker.sh` / `ecosystem.config.js` → 保留改主机名
- `e2e/` → 保留 playwright config，重写 scenarios
- `hooks/` → 保留通用，重写领域
- `i18n.ts` / `middleware.ts` / `messages/{en,zh}.json` → 保留，重写字符串表
- `next.config.js` / `postcss.config.js` / `tailwind.config.ts` → 保留，brand tokens 覆盖来自 `liulian-design-system`
- `lib/` → 保留通用，重写领域
- `playwright.config.ts` / `vitest.config.ts` → 保留

### 4.4 品牌覆盖层

通过 `antd-style` + `ConfigProvider` 把 antd 所有 token 覆盖到 LIULIAN brand（详细 theme 块见英文版 §4.4 或 ADR 0010）。

### 4.5 视觉底本

`feat/gui-demo` 分支是**视觉权威方向**（编辑级 Swiss · UniBe 红 `#E20613` · Fraunces + Switzer + JetBrains Mono · 暖纸 · 4 标签 IA Data/Train/Inference/Insight · 12-col × 8-row bento grid）。所有 `liulian-web` 的视觉决策都引用 gui-demo 的 `styles/main.css` 和 `docs/design-report.md`。

reuse fraction（Day 1 spike 后定）：30–50% / 40–60% / 10–20%。`assistant/Canvas*.tsx` 子集复用率最高（40–60%），省 ~2 周。

## 5. `liulian-ops` ← `neoctl`

源结构：`neoctl/cli.py + config.py + deploy/ + detect.py + doctor.py + llm_setup.py + ssh.py + tunnel.py` + `docs/{architecture,commands-reference,deployment-guide,manual-deployment,troubleshooting,index}.md`。pyproject：click + paramiko + pyyaml + rich + httpx。

文件级计划：

- `neoctl/cli.py` → 保留架构，改名 `neoctl` → `liulianctl`，换 service 名
- `neoctl/config.py` → 整段保留；config 改读 `~/.liulian/config.yaml`
- `neoctl/deploy/` → 保留架构，换 services (`backend` → `api`, `frontend` → `web`, `agent` 保留，加 `ingest`，加 `mobile` 通过 EAS Build)
- `neoctl/llm_setup.py` → 整段保留 — Ollama (37434) + vLLM (38000) bootstrap 一字不差
- `neoctl/ssh.py + tunnel.py` → 整段保留。autossh tunnel 对 LIULIAN GPU host 完全一样
- `neoctl/detect.py + doctor.py` → 整段保留；非常通用
- `docs/manual-deployment.md` → 保留结构，换 service 名

新增：`liulianctl/deploy/mobile.py`（包 `eas build`）+ `liulianctl/deploy/web.py`（包 `vercel deploy --prod`）+ `liulianctl/manifest_sync.py`（跑 `liulian-ingest` 后 PR manifest 到 `liulian-python`）

reuse fraction（Day 1 spike 后定）：70–85% / 10–25% / 5–15%

## 6. `liulian-dev-env` ← `neobanker-dev-env`

源：Codespaces / devcontainer config for neobanker frontend/backend/agent。

改造：加 `liulian-python` / `liulian-api` / `liulian-agent` / `liulian-ingest` / `liulian-web` / `liulian-mobile` 到 workspace；预装 uv / pnpm / docker compose / terraform / helm / kubectl / eas-cli / gh；预 seed `.env` 占位（DeepSeek / GLM / Gemini / Clerk / Sentry）；一键 `make dev` → `docker compose up` 起 timescaledb + redis + minio + prometheus + grafana + loki + tempo。

reuse fraction（Day 1 spike 后定）：60–80% / 20–40% / 0–10%

## 7. `liulian-api`（绿地，但借模式）

`neobanker-backend-MVP-V2` 是 Java/Spring Boot，不能 fork，但 **API 形状**（controller / DTO / pagination / error envelope）翻译到 FastAPI 干净。具体借：

- Pagination：`{items, total, page, page_size}` 一字不差保留
- Error envelope：`{code, message, details}` （RFC-7807-ish）一字不差保留
- audit-log row schema 保留
- Clerk-JWT 校验中间件（Spring Filter → FastAPI middleware 翻译）
- CORS allowlist 模式（`AGENT_CORS_ALLOWED_ORIGINS` 环境变量名一字不差）

Tracker + 实验 runner 包 `liulian-python` 是全新（neobanker 无对照）。Schema 见 PLATFORM_BLUEPRINT §5.3。详见 ADR 0009 模式翻译表。

## 8. `liulian-mobile`（Expo template 绿地）

无 neobanker mobile 可 fork。从 `pnpm create expo-app --template tabs-typescript` 起。

借自 `liulian-web`：

- Design tokens (`@liulian/design-tokens`)
- API 类型（codegen 自 `liulian-api` OpenAPI）
- SSE consumer 模式 — neobanker-frontend 的 EventSource hook 直接移植到 RN 的 `EventSource` polyfill

## 9. `liulian-python`（本仓）— 最小改动

按用户指令 *"liulian-python 应该在一个专门的 branch 上做"*，所有平台改动在 `feat/platform-upgrade-2026-05`。本仓 sprint 期间只改：

1. 策略文档在 `docs/strategy/`（本文集）
2. 文档惯例在 `docs/strategy/conventions/`
3. ADRs 在 `docs/strategy/adr/`
4. `liulian/` core *不动*（Python 包保持 API 兼容）。第一轮提的"包边界整理"推后 — `liulian-api` 把 `liulian` 作为 PyPI 版本固定依赖
5. `experiments/` 和 `manifests/` 不动（研究护城河）
6. `README.md` 最后改（sprint Day 6），指向新姊妹仓

`feat/gui-demo` orphan branch 保留；`liulian-web` 通过 copy 而非 git 关系导入其视觉底本。

## 10. `liulian-design-system`（绿地，从 gui-demo 起种）

源：`liulian-python/.worktrees/gui-demo/styles/main.css` + gui-demo 的设计报告。

输出：`tokens.json`（真源）、`tokens.css`（CSS vars）、`tailwind.preset.js`、`tokens.ts`（TS const）、`tokens.rn.ts`（RN StyleSheet）、`antd-theme.ts`（聊天侧栏）、Figma 库链接。

发布：`@liulian/design-tokens` GHCR npm registry（初期私有；M2 公开）。

## 11. 文档规则（采纳自 jajupmochi/claude-config）

用户的 `claude-config` 仓有 **9 条工作流规则**。LIULIAN 采纳重点子集，落为 `docs/strategy/conventions/DOCUMENTATION_RULES.md`。

### 11.1 bilingual-docs（采纳）

- 每个 repo 级面向人类的文档须 `NAME.md`（英文 canonical）+ `NAME.zh.md`（中文镜像）
- 顶部语言切换：`> **Language:** English | [中文](NAME.zh.md)` 和反向
- 同标题、同 TOC、同代码块；只翻译散文
- 不翻：代码、标识符、文件名、JSON/YAML keys、URLs、层级 ID
- 例外：CLAUDE.md / CLAUDE.local.md / SKILL.md / RULE.md / hook README / 内部研究草稿

### 11.2 双语方向

neobanker 是 zh 为 canonical + `.en.md` mirror。我们按 **claude-config** 反过来：**EN canonical + `.zh.md` mirror**。fork neobanker docs 时 swap 后缀。

### 11.3 四层文档结构

```
docs/
├── strategy/              L1-L2 + L0 conventions + ADRs
├── L3-design/             视觉表面（暂留 strategy/PLATFORM_DESIGN.md）
├── L4-runbooks/           运维面
└── api-reference.md       从 OpenAPI 自动生成
```

四层：L1 愿景（why）→ L2 架构（how-at-system）→ L3 设计（how-at-surface）→ L4 实现 + runbook（what / when）。L0 conventions 在底层。

### 11.4 其他 claude-config 规则

| 规则 | 采纳？| 落在 |
|---|---|---|
| chinese-output | yes（personal）| 每轮最终用户回复中文 |
| pre-edit-confirmation | yes（universal）| 编辑前必先列目标 + 1 行计划 + 等 go |
| phased-planning | yes（universal）| 大任务必分阶段 |
| plugin-preflight | yes（universal）| 调插件先核实 |
| output-brevity | yes | 不重复结尾摘要 |
| tool-proactivity | yes | 已装 skill 匹配自动触发 |
| no-reread-files | yes | 信任 session 内存 |
| ui-iteration-loop | yes（ui-project）| 8 轮自主 UI 改版含 chrome-devtools 截图 |
| bilingual-docs | yes（opt-in）| 本文整体应用 |

通过 `/init-claude-config` 注入新仓。

## 12. CI/CD 复用自 neobanker

每个 Python 仓的 `.github/workflows/ci.yml` 同模板（仿 neobanker-crawler）。JS 仓类似 parallel set。

复用工作流提升到 `liulian-ops/.github/workflows/`：`python-lint.yml` / `python-mypy.yml` / `python-pytest.yml` / `python-image.yml` / `js-lint.yml` / `js-tsc.yml` / `js-vitest.yml` / `js-playwright.yml` / `js-image.yml` / `release-tag.yml` / `release-notes.yml`。

每仓 workflow 6 行 caller。

Deploy workflow：每仓 `release.yml` build image → push GHCR → 通过 SSH 触发 `liulianctl deploy <service>`。同 neobanker 模式（`neoctl deploy all` 从 CI runner with SSH creds）。

## 13 / 14 / 15+. 详见英文版

见 `NEOBANKER_REUSE_MAP.md` §13 起 — 全面 reuse map 启用什么。Section 14 含：

- §14.1 后端 Spring Boot → FastAPI 模式翻译表（详见 ADR 0009）
- §14.2 前端 assistant/Canvas* 深度复用（详见 ADR 0008）
- §14.3 现代专业 dashboard 框架决策（**拒绝 Refine.dev**，手卷 Linear-meets-Bloomberg editorial Swiss /studio；详见 ADR 0007）
- §14.4 12 参考平台具体复用
- §14.5 gui-demo 视觉 canon
- §14.6 ADR 0006-0010 落地
- §14.7 总 reuse fraction 表（**全部以范围表示**，Day 1 spike 后实测替换）

---

*详细英文权威版：`NEOBANKER_REUSE_MAP.md`。架构上游：`PLATFORM_BLUEPRINT.md`。一周执行：`ONE_WEEK_SPRINT.md`。视觉表面：`PLATFORM_DESIGN.md`。决策档案：`adr/0001-0010`。*
