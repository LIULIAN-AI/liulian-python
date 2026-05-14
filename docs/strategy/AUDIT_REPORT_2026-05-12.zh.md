---
title: LIULIAN 策略文档 — 第三轮审计报告（中文镜像）
status: closed
parent: AUDIT_REPORT_2026-05-12.md
---

# LIULIAN 策略文档 — 第三轮审计报告（中文版）

> **语言：** [English](AUDIT_REPORT_2026-05-12.md) | 中文

审计者：`impeccable` 技能（设计审）+ `research-critic` 技能（声明逐条审）

目标文档：PLATFORM_BLUEPRINT.md / LIULIAN_REUSE_MAP.md / ONE_WEEK_SPRINT.md / REFERENCE_DESIGNS.md

## A. impeccable 设计审

### A.1 AI-slop / category-reflex 测试

| 测试 | 判 | 详 |
|---|---|---|
| AI-slop test | 通过 | 编辑级 Swiss + UniBe 红 + Fraunces 对 TS/ST 工具罕见；血脉清楚（gui-demo iter 2）|
| Category-reflex test | 通过 | 水文默认 = 冰川蓝 + 白；我们 = 暖纸 + UniBe 红，定违反射 |
| 三动词 hero "Observe. Forecast. Decide." 借自 k-dense.ai | 弱通过 | 模式 OK；文案普通。替换为 `Manifest · Train · Reason · Forecast`（来自 gui-demo design report）|

### A.2 绝对禁令审

| 禁令 | 状态 | 修 |
|---|---|---|
| 渐变文本 | clean | – |
| 侧条 > 1px | **违规** in LIULIAN_REUSE_MAP §14.3 ("选中行 2px 左条") | **已修**：换 6×6 px 红点 |
| Hero-metric 模板 | clean | – |
| 同款 card grid | clean（bento grid 尺寸不一）| – |
| modal 第一反应 | clean（用 row hover peek 替代）| – |
| 玻璃化默认 | clean | – |
| `#000` / `#fff` | clean (`#131313` / `#FBFBFA`)| – |
| 散文中 em dash (`—`) | **违规**（4 文档 100+ 处）| **延后**：full sweep 是 multi-hour edit；放到 sprint Day 7 polish 任务。未来在这些文档的写入**必须**用逗号 / 冒号 / 分号 / 句号 / 括号替代 |
| 卡片作懒答案 | clean（forecast bento 需要 tile 分离做报告 builder）| – |

### A.3 鲜明度审（/studio 够独特吗？）

审前："Linear-meets-Bloomberg editorial Swiss" 方向强但 implementation move 太少。审后加了 6 个更狠的招到 LIULIAN_REUSE_MAP §14.3：

1. 状态用**排版**（Fraunces italic "running" / 粗 roman "failed" / Switzer regular "completed"）而非彩色 pill
2. 科学新闻报眉每页：`LIULIAN · Studio · timestamp · run-coordinates`
3. 数字 [0.01, 1000] 之外默认科学计数法
4. 全程 tabular numerals（隐含 → 明确）
5. 单 Easter egg：阈值穿越时 `流` 涟漪一次。领域真切、独特、永不重复
6. 选中行 6×6 像素红点替代被禁的侧条

审后判：通过。融合方向不再可被误认为 2026 任何其他 admin panel。

### A.4 /(marketing) section 不足规定

详见英文版。已在 PLATFORM_DESIGN.md §6 完整规定（concrete moves：swisstopo Aare 卫星瓦片背景 + 88% 暖纸覆盖 + Fraunces 大字 + 17 字副线 + 实时计数 + 三纵向规则带 + 内嵌 demo 视频 + 单文本箭头 CTA）。

## B. research-critic 声明逐条审

六问审（可证伪 / 设计测假设 / 公平对比 / 泄露 / 比例 / 替代方案排除）。

### B.1 一人 + AI 团队的多仓 vs 单仓

| Q | 判 |
|---|---|
| 可证伪？| 部分（"CI < 5 min" 可测；大部分是偏好）|
| 设计测假设？| 否；未测 CI |
| 公平对比？| 否；对的是不带 Turborepo 的单仓。Turborepo 选择性构建专门解决 > 15 min 问题 |
| 泄露？| – |
| 比例？| 过度声明；"sprint 省 2 天" 是先验猜测 |
| 替代排除？| 否；hybrid (Python 多仓 + JS Turborepo 单仓) 未考虑 |

**诚实重述**：*我们选多仓三个可辩护理由：(a) liulian 操作肌肉记忆，(b) OSS 贡献者更小 per-repo 表面，(c) mobile / web / Python core 真正发散的发布节奏。我们**不**声称 CI 更快；那需要测量。决策可逆：若 M3 时跨仓协调变痛，可把 JS 三仓塌缩成 Turborepo，保 Python 多仓。*

→ `PLATFORM_BLUEPRINT.md §4.2` 已在第三轮 commit 修正。

### B.2 TimescaleDB "吃自己狗粮"

| Q | 判 |
|---|---|
| 可证伪？| yes |
| 设计测假设？| 部分（"hypertable 消除手分区" 可测但未测）|
| 公平对比？| yes |
| 泄露？| – |
| 比例？| "engineer-reviewer 在招聘环节中被打动" 是**氛围**不是证据。过度声明 |
| 替代排除？| 纯 Postgres + pg_partman + pg_cron 在 M4 前更简单且够用。文档承认但未充分掂量 |

**诚实重述**：*我们为 run_metric / forecast / alert 表选 TimescaleDB 因 hypertable 在零迁移成本下消除手动时间分区。"吃狗粮" 框架是次要叙事，不是主要技术理由。若纯 Postgres 路径让我们更快出 M1-M2，TimescaleDB extension 可延后到 M3 启用。*

→ `PLATFORM_BLUEPRINT.md §5.1` 已在第三轮 commit 修正。

### B.3 fork-and-adapt reuse 比 70 / 80 / 85%

| Q | 判 |
|---|---|
| 可证伪？| yes（fork 后可测）|
| 设计测假设？| 否；比例是猜测；还没 fork |
| 公平？| 部分；假设 domain swap (finance → TS/ST) 只触 `bank_matcher.py` 类表面，但银行域假设可能更深 (prompt 模板 / context cache 结构 / intent 词汇 / demo 场景)|
| 泄露？| 可能；tool I/O 形状不同 (预测有 time + station + quantile 结构 vs liulian 表格数据)|
| 比例？| 假精度。70 / 80 / 85 读起来像实测；其实估计 |
| 替代排除？| 绿地考虑且拒绝，OK |

**诚实重述**：*估计 reuse 比例（Sprint Day 1 spike 后定）：agent ~50-70%、crawler ~50-80%、neoctl ~70-85%、frontend canvas-orchestrator ~40-60%。完整审计在 Day 1 fork 时发生。平台成功**不**依赖打中这些数字；只依赖出 M1 demo end-to-end。*

→ `LIULIAN_REUSE_MAP.md §2.6 / §3.4 / §4.6 / §5.4 / §14.7` 全部已在第三轮 commit 改为范围 + "(TBD Day 1)" 备注。

### B.4 "/studio 手卷 3 周但 week-1 pitch 回本"

| Q | 判 |
|---|---|
| 可证伪？| 部分。"回本" 模糊 |
| 设计测假设？| 否；无"普通看相" vs "独特看相" pitch 的 a/b 测试 |
| 公平？| 否；比较 3 周工作量与"首次 pitch 印象"非对称 |
| 泄露？| 有；选择偏差选挑 recruiter/VC 注意设计。许多根据技术深度判断 |
| 比例？| 过度声明。独特 UI 有帮助。不是决定轴 |
| 替代排除？| 否；那 3 周可用于 (a) 更深 Chronos 集成、(b) 真客户 pilot、(c) 技术博客。任一可能更高杠杆 |

**诚实重述**：*手卷 /studio 比 Refine.dev 模板多 ~2-3 周。边际收益是品牌差异化，这是评估的**一**条轴，很少是**决定**轴。投资的合理理由是不同的：品牌**就是**融资级独特性的产品定位。M2 评估的反向投资：(a) Chronos 对 Time-Series-Library leaderboard 的深度 benchmark、(b) swisstopo / Eawag pilot 对话、(c) SwissRiver 数据集技术博客。*

→ `LIULIAN_REUSE_MAP.md §14.3` 已在第三轮 commit 修正（最被过度声明的段落）。

### B.5 兼职博士后 6 月达 pre-seed

| Q | 判 |
|---|---|
| 可证伪？| yes (2026-11 关 pre-seed 可观察)|
| 设计测假设？| 部分（每个 side-bet 可测）|
| 公平？| 否；与全职创始人对比；6 月内 pre-seed 几乎都是全职 |
| 泄露？| 研究截止日（ICPR 终稿、SNSF 周期）会撞期。路线图未计 |
| 比例？| 兼职过度声明。12 月更现实 |
| 替代排除？| 未考虑：(a) 12 月含研究滑期、(b) 兼职咨询资助、(c) 先 SNSF/ERC 联合研究 grant |

**诚实重述**：*M1-M6 假设每周 ~20 小时可持续在 LIULIAN（博士后角色之外）。若研究截止日撞期，路线图按每次撞 N 周滑。三档时间线相计：*

- 激进档（如原写）：M1 2026-05-19，M2 2026-06-30，pre-seed 2026-11。需零滑
- 现实档：M1 2026-05-19，M2 2026-08-01（2.5 月），pre-seed 2027-Q1（8-9 月）。一次 ICPR 或 SNSF 滑可容
- 保守档：M1 2026-05-19，M2 2026-09，pre-seed 2027-Q2（12 月）。多次滑可容

*三档都保 M1（ARTORG 是硬外部截止）。*

→ `PLATFORM_BLUEPRINT.md §15` 已在第三轮 commit 改为三档表。

## C. 本轮已应用修复

| 修 | 状态 | 位置 |
|---|---|---|
| 侧条违规 | DONE | LIULIAN_REUSE_MAP §14.3 (6×6 点替 2px 条) |
| 6 个更狠的 /studio move | DONE | LIULIAN_REUSE_MAP §14.3 |
| /(marketing) 具体规定 | DONE 完整 | PLATFORM_DESIGN.md §6 |
| UI 审计 checklist 形式化 | DONE | conventions/UI_AUDIT_CHECKLIST.md |
| 多仓诚实理由 | DONE | PLATFORM_BLUEPRINT.md §4.2 |
| TimescaleDB 去标语化 | DONE | PLATFORM_BLUEPRINT.md §5.1 |
| reuse 比例改为范围 + "(TBD Day 1)" | DONE | LIULIAN_REUSE_MAP §2.6 / §3.4 / §4.6 / §5.4 / §14.7 |
| 3 周 /studio 过度声明软化 | DONE | LIULIAN_REUSE_MAP §14.3 末段 |
| M1-M6 三档时间线 | DONE | PLATFORM_BLUEPRINT.md §15 |
| em-dash 扫描 | 延后到 Sprint Day 7 polish | 全部文档 |

## D. Sprint Day 1 的两个直接改动

两条审计发现立刻影响 Day 1：

1. **reuse-fraction spike**：Day 1 第一个 90 分钟（在 workspace 骨架后）做结构化的 "fork-and-measure" — clone liulian-agent、剥银行域代码、count 剩余 LOC、重算 reuse 比例。**用数据替换猜测**，sprint 估值落地前。
2. **纯 Postgres 优先**：Day 1 docker-compose 用纯 Postgres（不用 TimescaleDB-HA image）。TimescaleDB extension 在 M1 demo 上线**后**开启。从 sprint 移除一个风险向量。

两改已反映在第三轮 ONE_WEEK_SPRINT.md commit。

## E. 收尾打分

审前复合 "审稿人想见你" 估值：6/10。独特方向有，但若干俗气（em dash、hero verbs）和若干过度声明（reuse % / 时间表）拉分。

审后估值（队列修复落地后）：**8/10**。剩 2 分要等真屏幕做出来（M2+）。策略文档现在生产级；剩工作是实现。

---

*详细英文权威版：`AUDIT_REPORT_2026-05-12.md`。*
