---
title: LIULIAN 一周冲刺计划（2026-05-13 → 2026-05-19，中文镜像）
status: time-boxed
parent: ONE_WEEK_SPRINT.md
---

# LIULIAN 一周冲刺（2026-05-13 → 2026-05-19，中文版）

> **语言：** [English](ONE_WEEK_SPRINT.md) | 中文

把蓝图（PLATFORM_BLUEPRINT.md）在 7 天集中工作中落地成可演示、招聘人能点开、VC 能转的状态。

## 0. 双交付目标

两条交付线在 Day 7 汇合：

1. **ARTORG-AIHN 投递**（截止 2026-05-19）—— 需要 mobile + 后端 + 医疗 demo + 隐私态势
2. **通用 ML / Platform Engineer 招聘**—— 需要 K8s/Helm 骨架 + FastAPI 网关 + Grafana + BI 画布 + agent 集成

两套需求 80% 重叠。

## 1. Sprint 前预检（5-13 上工前做）

- [ ] Python 3.10 + `uv` 已装
- [ ] Node 20 + `pnpm` 9 已装
- [ ] Docker 24+ + compose v2 已装
- [ ] `gh` 已 auth；`vercel` CLI、`railway` CLI 已装
- [ ] 域名 `liulian.app`（或 `.dev`）已注册并开 DNSSEC
- [ ] 当前分支 `feat/platform-upgrade-2026-05`

## 2. Day-by-day

### Day 1 — Tue 2026-05-13 — fork-and-measure spike + 骨架 + 身份

> 第三轮（审计驱动）改动：先做 fork-and-measure spike 替代第一轮的 reuse-fraction 估值；Day 1 用纯 Postgres，TimescaleDB extension 等 M1 demo 上线后开（ADR 0003）。

**Flagship**：可截图的营销 landing 在 `localhost:3000`，hero + 三条编辑带 + 单条 text-arrow CTA，用 LIULIAN brand tokens。
**Scaffold**：8 仓初始化并 fork（按 `LIULIAN_REUSE_MAP.md §1`）；纯 Postgres 通过 docker-compose 起来；FastAPI `/healthz` 返 200；`@liulian/design-tokens` 发到私有 npm 供 web 消费。

#### 任务序列

**0. Fork-and-measure spike**（先做，90 分钟硬上限）

替代 LIULIAN_REUSE_MAP §14.7 的"估值"行为实测数：

每个可 fork 的仓，顺序：agent → crawler → neoctl → frontend → dev-env：

- `gh repo fork liulian-ai/liulian-<x> --clone --org=jajupmochi`
- 远程改名 (`gh repo rename liulian-<x>`) + 本地改名
- 删银行域代码：`bank_*` 模块、银行 prompts、银行 fixtures、银行 scenario JSON
- `cloc .` → 数余下 LOC
- 用实测数更新 reuse map 对应章节的 *Estimated reuse fraction*
- 每仓 commit rename + strip 成一个 PR

若超时，shipped what's measured，Day 2 morning 补完。

**1. Workspace 骨架（在 `liulian-python` 内，60 分钟）**

- `liulian/` 包内容保持不动（不再迁到 `packages/liulian-core/`，因为多仓决策让包本身留在 `liulian/`，只动元数据 + 边界）
- `pytest` 跑通确认零回归
- 本仓 sprint 期间只装 `docs/strategy/` + 元数据小改

**2. Design tokens（90 分钟）—— `liulian-design-system` 新仓**

- 把 `gui-demo/styles/main.css` 复制为 `tokens.css`
- 生成 Tailwind preset + TS / RN 导出
- 发到 GHCR npm registry（私有）为 `@liulian/design-tokens@0.1.0`

**3. liulian-api 骨架（60 分钟）**

- 新仓，FastAPI + `liulian-core`（pip 从本仓 tag）+ 纯 Postgres 通过 `docker-compose.dev.yml`
- `/healthz` + `/readyz` 端点
- 一个 smoke 端点：`GET /models` 列 adapter discovery（无 auth）

**4. liulian-web landing（剩余时间）**

- fork 自 `liulian-web`（spike 已本地 clone）
- 改写 `app/(marketing)/page.tsx` 按 `PLATFORM_DESIGN.md §6`
- Brand tokens 通过 Tailwind preset 注入
- 所有 antd UI 通过 `liulianAntdTheme` ConfigProvider 主题化（ADR 0010）
- 确认 LIULIAN wordmark 的 wonk-1 italic `U` 在 Chrome / Firefox / Safari 渲染正确

**5. UI 审计跑一遍（5 分钟）**

- 对 landing 跑 `conventions/UI_AUDIT_CHECKLIST.md`
- 任何不通过的项作为 Day 2 修复

#### 验收

- `curl localhost:8000/healthz` → 200
- 8 个仓在 `liulian-ai/liulian-*` 可见
- reuse map 里 reuse 比例已是实测数
- landing 截图视觉与 gui-demo 同调

---

### Day 2 — Wed 2026-05-14 — 后端深度

**Flagship**：`/api/docs` Swagger UI 工作，`/experiments` `/forecasts` `/models` 三组端点齐全；招聘人能点 `GET /models` 拿到列表。
**Scaffold**：SQLModel schema + Alembic 迁移；SDK 包从 OpenAPI 自动 codegen 类型化方法。

#### 任务

1. DB + ORM（2h）：docker-compose 起 postgres + redis + minio；SQLModel 按 §5.3 建表；Alembic init + 首迁移
2. 服务层 + 端点（3h）：experiments / models / forecasts / datasets 服务包
3. OpenAPI codegen（1h）：`apps/web/` 跑 `openapi-typescript`，`pnpm gen:api` 脚本
4. Python SDK（1h）：`liulian-sdk` 仓 + `openapi-python-client` 生成
5. Auth stub（0.5h）：写 `Authorization: Bearer <demo-token>` 中间件，硬编码 demo token
6. 测试（1h）：每端点一条 happy-path pytest

#### 验收

- Swagger UI 在 `localhost:8000/docs`，`GET /models` 返 30+ 模型
- `liulian-sdk` 从 fresh venv 装上能跑

---

### Day 3 — Thu 2026-05-15 — BI 画布 v0 + 第一张真图

**Flagship**：`/forecast` 页显示 ECharts 时序图，数据来自后端真实 SwissRiver 预测。
**Scaffold**：tRPC 层；ECharts wrapper 组件；地图占位；数据拉取模式。

#### 任务

1. 种子真数据（1h）：把 `experiments/swiss_river/` 跑一遍生成 1 实验 / 1 run / 5 forecast
2. tRPC 层（1h）：`apps/web/src/server/trpc/` routers experiments / forecasts / models / agents
3. 画布 shell（1h）：路由 `/forecast`，四象限布局（stations 侧栏 · 地图 · 时序 · 分布）
4. canonical 时序图（3h）：`<ForecastChart />` 组件包装 ECharts；layers 按 `PLATFORM_DESIGN §4.2`
5. 地图占位 + 真实站点（1h）：MapLibre + 28 站点 markers，click → cross-filter
6. Loading + Error state（0.5h）：shadcn `<Skeleton>` + 错误 toast

#### 验收

- 浏览器开 `/forecast`，看到带预测带的真实 fan chart；点地图 marker 后图更新

---

### Day 4 — Fri 2026-05-16 — Mobile + BI 加深

**Flagship-mobile**：Expo Go QR 跑通 iOS Simulator + Android emulator + 真机，Forecast tab 展示与 web 同源逻辑的图。
**Flagship-web**：加 panels — 残差分布、多模型 overlay、KPI 条。
**Scaffold**：共享图表逻辑在 `packages/liulian-charts/`。

#### 任务

1. `packages/liulian-charts/`（1.5h）：纯函数 `computeQuantileFan` 等，web + mobile 共用
2. Mobile 屏（2.5h）：Home + Forecast + Alerts 三页 + EXPO_PUBLIC_API_URL
3. Web 残差直方图（1.5h）：`<ResidualHistogram />` ECharts custom-series + 密度曲线
4. 多模型 overlay（1h）：顶栏 multi-select + 多 fan
5. KPI 条（1h）：Tremor `<Metric />` 卡 MAE / RMSE / CRPS / Coverage@90

#### 验收

- 手机扫 Expo Go QR 跑通，Forecast 图加载
- web `/forecast` 四个 panel 工作

---

### Day 5 — Sat 2026-05-17 — Agent + ML 集成 + 地图拓扑

**Flagship**：BI agent 在聊天侧边栏可以说"显示 Q95 上周超阈值的站点"，画布响应。加 Chronos-2 零样本预测按钮。
**Scaffold**：自研 agent state graph；LiteLLM 代理；地图拓扑覆盖。

#### 任务

1. Chronos-2 adapter（1.5h）：`liulian/adapters/chronos/` 包 `Chronos2Pipeline`；capabilities `["zero_shot", "probabilistic", ...]`；UI 加按钮
2. TSL adapter 提升（1h）：把 `experiments/adapt_tsl_lib/` 提为 `liulian/adapters/tsl/`，至少暴露 `STConvNet`
3. 地图拓扑（1h）：manifest 加 river-network edges；SVG 覆盖 MapLibre
4. Agent 层 v0（3h）：`liulian-agent` 仓 fork 自 liulian-agent，改名 → 工具替换 → prompt 重写。LiteLLM 代理 + DeepSeek / GLM / Gemini provider 配置。FastAPI 端点 `POST /agents/{name}/invoke` SSE 流。web 侧栏 `<ChatPanel />` 用 Vercel AI SDK `useChat()`
5. 一条 canned 交互（0.5h）：prompt "为 Q95 > 850 的站加 alert next week" → agent 调 `create_alert_rule` → BI 更新

#### 验收

- 聊天侧栏 "show me Bern station" → 时序过滤到 Bern
- 点 Chronos-2 零样本按钮 → fan 在 ~2 秒内更新
- 地图显示拓扑图；点 edge 显示 lead-lag scatter

---

### Day 6 — Sun 2026-05-18 — 云部署 + 观测 + 文档

**Flagship**：三个 deploy URL 在 README 里都返 200；Grafana dashboard 可访问；演示视频上传。
**Scaffold**：GitHub Actions CI 通过；Helm chart 编译；Terraform 模块 lint 通过。

#### 任务

1. 后端部署（1.5h）：`liulian-api/Dockerfile`（多阶段 distroless）；Railway 连仓部署；Postgres add-on；`https://liulian-api.up.railway.app/healthz` → 200
2. Web 部署（1h）：`vercel link` + `vercel deploy --prod`
3. Mobile 分发（1h）：`eas build --profile preview --platform all`；APK 上传可下载 URL；Expo Go QR 印 README
4. 观测（1.5h）：本地 `docker-compose -f infra/compose/observability.yml up`（prom + grafana + loki + tempo）；Grafana dashboard 预置
5. CI（1h）：`.github/workflows/ci.yml` per `liulian-ops` 复用工作流；让 CI 绿
6. Helm + Terraform 骨架（1h）：`liulian-ops/helm/liulian-platform/` chart `helm template` 通；`infra/terraform/{aws-eks,hetzner-k3s}/` `terraform validate` 通
7. 文档（1.5h）：重写 `README.md`；架构图（Mermaid）；截图；"Try it now"；隐私 + 安全章节；`docs/ARTORG-PORTFOLIO.md`；`docs/DEMO.md`
8. Demo 视频（1h）：90 秒：landing → forecast canvas → 站点点 → 聊天侧栏 → 零样本 → 移动 QR → 同样的图。上 Loom

#### 验收

- 三个 deploy URL 返 200
- `gh workflow run ci.yml` 10 分钟内绿
- `helm template infra/helm/liulian-platform` 产出有效 YAML
- README 在 github.com 上渲染干净

---

### Day 7 — Mon 2026-05-19 — 收尾 + 投递 + 打标

**Flagship**：ARTORG 应用 PDF 投递，作品集章节引用 live URL。`v0.6.0` 打标。
**Scaffold**：端到端 smoke 在 live URL 上绿；README 加 "什么还没做（故意的）" 章节。

#### 任务

1. 端到端 smoke（1.5h）：fresh browser → web URL → forecast canvas 工作；fresh phone → Expo Go QR → app 跑通；`liulian-sdk` 从 fresh venv 装上能跑
2. "什么还没做（故意的）" 章节（0.5h）：多租户 auth、设备端推理、联邦学习、合规审计
3. ARTORG 投递（1h）：copy `docs/ARTORG-PORTFOLIO.md` 块到应用 PDF；确认 live URL 匹配文档；按 JD 投
4. 打标 + release（0.5h）：`git tag v0.6.0-portfolio` → push；GitHub release notes
5. Week 2 移交笔记（0.5h）：把 BLUEPRINT §15 的 M2 deliverables 复制到 `docs/strategy/M2_PLAN.md` seed 下周
6. Em-dash sweep（0.5h）：策略文档全局清理 em dash（按 UI_AUDIT_CHECKLIST 的 F 段必须）
7. 双语镜像收尾（0.5h）：把执行摘要级的 .zh.md 升级为完整 prose 镜像
8. 余量 buffer（1.5h）：前 6 天累积超时

#### 验收

- 应用投出；确认邮件保存
- 标 `v0.6.0-portfolio` 在 github.com 可见
- BLUEPRINT §9 验收清单全部勾绿

## 3. 削减清单（若 sprint 跑紧）

按顺序削：

1. Helm + Terraform 骨架（Day 6 末小时）→ 推到 M2
2. 多模型 overlay（Day 4）→ 单 fan 即可
3. 地图拓扑覆盖（Day 5）→ 单站 marker 够 demo
4. Mobile 原生 build（.apk）→ Expo Go QR 单走够
5. TSL adapter 提升 → 推到 M2；Chronos-2 单走够
6. Agent BI 工具表面 → 保留 Q&A 即可

**绝不削**：FastAPI + Swagger UI、画布 fan chart、一屏 Expo Go mobile、README。

## 4. 风险与预案

| 风险 | 预案 |
|---|---|
| 现有测试在 `liulian/` 边界微调时挂 | Day 1 第一步；不绿不动其他 |
| Chronos-2 权重大 / 装慢 | sprint 用 Chronos-Bolt（小），Chronos-2 标 opt-in |
| Railway Postgres-TimescaleDB free 限额 | Fall back 纯 Postgres + pg_partman；Timescale 留 staging |
| MapLibre + swisstopo attribution | 读 swisstopo OpenData license；attribution 烧进地图脚 |
| Agent 把敏感数据发给第三方 LLM | 默认 LiteLLM 切本地 Ollama；cloud LLM 在 per-tenant feature flag 后 |
| 无 Mac 不能测 iOS | 借 iPhone 跑 Expo Go 拍一张截图；主 demo Android emulator + Expo Go |
| Day 6 部署级联失败 | Day 7 早上留备用部署窗口；Day 6 推上去前先在本地测 commit |

## 5. End-of-sprint 验收清单

- [ ] GitHub repo 公开；README 干净渲染
- [ ] `https://liulian-api.up.railway.app/healthz` → 200
- [ ] `https://liulian-web.vercel.app/forecast` → 渲染预测图
- [ ] Expo Go QR 在至少一台真机上跑通
- [ ] `gh workflow view ci.yml` 最新跑绿
- [ ] `helm template infra/helm/liulian-platform` 产 valid YAML
- [ ] `terraform validate` 两模块都过
- [ ] `docs/ARTORG-PORTFOLIO.md` 文本拷进应用 PDF
- [ ] 90 秒 demo 视频在 README 嵌入
- [ ] 隐私 + 安全章节在 README
- [ ] 无 PHI / 无真实病人数据进仓
- [ ] `v0.6.0-portfolio` 标已推
- [ ] `docs/strategy/M2_PLAN.md` 创建带 M2 交付物

## 6. Sprint 后 — 面试 / VC 通话话术

详见英文版 §6。

要点：
- "为啥 FastAPI" → OpenAPI auto-gen → SDK + web codegen
- "为啥 TimescaleDB" → 我们是 TS 产品，吃自己狗粮（次要叙事；主要是分区管理自动化）
- "为啥 ECharts 不是 Plotly" → 浏览器侧大数据强、theme engine 服品牌
- "为啥 Expo 不是 native" → 单代码库 iOS+Android + OTA + 无 Xcode hostage
- "为啥自研 agent 不用 LangGraph" → 5 工具 surface 的开销不值；维护成本可控
- "为啥 Helm + Terraform 现在" → 客户签下后第二天能部署到他的 AWS / Azure，骨架让承诺落地

医疗角度（ARTORG）：ECG demo + 公开 PhysioNet + 设备端推理路径 + M5 联邦学习

时空护城河（VC）：30+ 模型 zoo + 17 数据集 benchmark + TSL 集成 + Chronos-2 零样本降低新垂直冷启动成本

---

*详细英文权威版：`ONE_WEEK_SPRINT.md`。架构上游：`PLATFORM_BLUEPRINT.md`。视觉表面：`PLATFORM_DESIGN.md`。复用计划：`LIULIAN_REUSE_MAP.md`。*
