---
title: LIULIAN 平台设计（L3，中文镜像）
status: living document
parent: PLATFORM_DESIGN.md
---

# LIULIAN 平台设计（L3，中文版）

> **语言：** [English](PLATFORM_DESIGN.md) | 中文

LIULIAN 的可视表面：品牌语态、色彩、排版、BI 画布逐面板、agent 对话流、移动端屏幕。

## 0. 视觉原创契约（最重要规则）

LIULIAN 从 liulian + 12 参考平台复用代码 / 架构 / 操作模式。**每一个可见像素都是原创**。任何 LIULIAN 表面截图都不应被误认为：

- liulian 截图换文字
- Refine.dev 模板换 logo
- 通用 shadcn-admin 启动模板
- 任何 2026 SaaS 默认 admin
- Vercel / Linear / Stripe 表面换我们的文案

每个 PR 跑 `conventions/UI_AUDIT_CHECKLIST.md §I` 的 4 项测试：AI-slop / category-reflex / liulian-copy / gui-demo cross-check。任何一项失败阻塞合并。

## 1. 品牌语态

### 1.1 四字定位

**Liquid Intelligence for Time.**（*为时间而生的流动智能。*）

`/(marketing)` 的 wordmark caption。"L*U*iquid" 和 "LIUL*I*AN" 中的字母 *U* 用 Fraunces 可变字体，WONK 1 轴 =1，weight 600，斜体，UniBe 红。

### 1.2 25 字电梯

*面向时空 AI 的开源生产栈。研究级模型库、工业级 BI 画布、智能体工作流、主权部署。诞生于伯尔尼。*

### 1.3 语态属性

| 属性 | 是 | 否 |
|---|---|---|
| Register | 科学的，编辑的 | 商务的，活泼的，营销腔 |
| 节奏 | 考究的，稀疏 | 急促的 |
| 动词 | 具体（observe / manifest / train / forecast / alert）| 抽象（revolutionize / transform / unlock）|
| 代词 | "我们"罕；多用客观科学口吻 | "你的团队" / "you" 过度 |
| 数字 | 精确带单位 | 空"10x" / "100x" |
| 形容词 | 克制，技术 | 最高级（best / smartest）|

**禁词**：seamless / unleash / elevate / next-gen / game-changer / delve / harness / leverage（作动词）/ revolutionize / transform（营销语境）。

### 1.4 反向参考

- Tableau / Power BI 通用 dashboard：太模板，太蓝，太"企业"
- Crypto / Web3 霓虹仪表盘：刺眼，分散
- 默认 AI 工具的青 + 紫渐变：AI-slop 默认
- Vercel marketing：聪明但已被模仿 1000 次
- Linear UI：我们的 /studio *向其纪律靠齐* 但用不同字体 + 调色板防止 look-alike

### 1.5 我们致敬并明确借鉴

- *Monocle* 杂志：排版层级、纸感、照片节制、句子节奏
- Bloomberg Terminal 血脉：信息密度通过微排版 + 留白做，绝不通过视觉杂乱
- Linear：command palette 作主要 nav，克制 chrome
- Müller-Brockmann / Swiss 排版传统：网格、不对称平衡、克制颜色
- Edward Tufte：墨-数据比纪律，sparkline 灵感
- gui-demo iteration 2（本仓）：LIULIAN 已落地的品牌典籍

## 2. 设计 tokens

源在 `liulian-design-system/tokens.json`（新仓），生成：

- `tokens.css`（CSS 自定义属性）
- `tailwind.preset.js`
- `tokens.ts`（TS const）
- `tokens.rn.ts`（RN StyleSheet）
- `antd-theme.ts`（聊天侧栏 antd ConfigProvider）

### 2.1 颜色（OKLCH；hex 仅参考）

```css
/* Canvas + Ink */
--canvas-warm     : #FBFBFA
--surface-pure    : #FFFFFF
--ink-charcoal    : #131313
--ink-muted       : #5C6066
--ink-faint       : #8E9296
--hairline        : #EAEAEA

/* UniBe 红（品牌锚）*/
--unibe-red       : #E20613
--unibe-red-tint  : #FDEBEC
--unibe-red-deep  : #B00510

/* 状态 pastels（稀有）*/
--pastel-green    : #EDF3EC
--pastel-blue     : #E1F3FE
--pastel-yellow   : #FBF3DB

/* UniBe 次要（仅 chart series）*/
--unibe-ocean     : ~#0066B3
--unibe-green     : ~#509A39
--unibe-apricot   : ~#E6863A
```

**色彩策略**：*Committed*（按 impeccable 分类）。UniBe 红承载身份约占总墨色 5%；暖纸色占 92%；pastels + 次要色合占 3%。

**规则**：

1. UniBe 红 `#E20613` 仅用于：wordmark `U` / 活跃站点 marker / 预测线 / CI 带颜色 / elevated 严重度 pill / 阈值穿越 marker / 选中行前 6×6 px 点 / input focus ring / destructive-confirm 按钮背景。多数页面**每屏可见最多 2 个红元素**；alert canvas 上限 3。
2. 绝不 `#000` / `#FFF`。
3. 状态 pastels 只出现在 pills / severity 带，绝不大面积填充。
4. 次要 UniBe 色仅作 chart series 多于一时；绝不作按钮 / 背景。

### 2.2 字体

```css
--font-display : 'Fraunces', 'Instrument Serif', 'Charter', serif
--font-body    : 'Switzer', 'Inter Tight', system-ui, sans-serif
--font-mono    : 'JetBrains Mono', 'IBM Plex Mono', monospace
```

Fraunces 变量轴：`opsz` 9..144 / `wght` 300..900 / `SOFT` 0..100 / `WONK` 0..1（WONK=1 用作斜体 accent `U`）。

**尺度**（1.25 比，几何级数）：详英文版 §2.2 表。要点：

- Display 1：96–144px Fraunces / 500 italic on accent char / -0.035em
- Body L：17px / 1.6 Switzer / 400 / -0.005em（长文）
- Body M：15px / 1.55（UI 文本）
- Mono：13/15px JetBrains Mono / tabular-nums

**规则**：

1. body 行宽 ≤ 72ch
2. 层级比 ≥ 1.25
3. 数字列必 `font-variant-numeric: tabular-nums`
4. Inter / Roboto / Open Sans **禁用**

### 2.3 间距 + 圆角 + 边

8 步等比间距 `--space-1..-10`（4 → 128px）。半径：`--radius-sm` 4 / `--radius-md` 10（card 默认）/ `--radius-lg` 14 / `--radius-xl` 20 / `--radius-pill` 9999。卡片：1px hairline 边 + 10px 半径 + **零阴影**。

### 2.4 动效

ease-out-quart（`cubic-bezier(0.16, 1, 0.3, 1)`）默认。`transform` + `opacity` 唯二可动属性。**禁 bounce / 弹性**。`prefers-reduced-motion` 须 honor。

### 2.5 图标

Phosphor Icons (Regular 1.5px stroke) 主，Lucide fallback（liulian 用，代码复用一致）。领域特定 glyph 自画 24px grid 1.5px stroke。表格 cell 内**禁图标**。

## 3. 信息架构

`liulian-web` 路由地图：

```
/                       marketing landing
/(marketing)/about      长定位
/(marketing)/research   论文 + benchmark
/(marketing)/blog       contentlayer 帖子
/(marketing)/pricing    M3+

/forecast               BI 画布（杀手 demo）
/forecast/r/:slug       分享报告（只读公开）

/studio                 4-tab workspace shell
/studio/data            数据集 + manifests
/studio/train           实验 + runs + HPO
/studio/inference       预测 + 反事实
/studio/insight         报告 + 定时摘要

/agents/data            data agent chat
/agents/model           model agent chat
/agents/bi              BI agent chat

/admin                  租户 + 用户 + 审计（M3+）

/docs                   MkDocs 技术文档
/api/*                  FastAPI 代理（服务端）
```

`/studio` 内 4-tab 的 IA 匹配 gui-demo 的 **Data / Train / Inference / Insight**（典型 LIULIAN 心智模型）。

### 3.2 页面 chrome

除 `/(marketing)` 外，每页顶部带**科学新闻报眉**：

```
LIULIAN · Studio · 2026-05-12 14:38 UTC · swiss-river-1990 / lstm / entity=none / seed=42
```

JetBrains Mono 11px uppercase letter-spacing 0.04em，色 `--ink-faint`，sticky 32px 高。无面包屑。

## 4. BI 画布 — `/forecast`（杀手 demo）

12-col × 8-row CSS Grid。详见英文版 §4.1 ASCII 图。

### 4.2 8 canonical panels

详见英文版 §4.2，要点：

1. **交互式河网地图**：MapLibre + swisstopo + 拓扑覆盖 + 站点同心圆 markers + click 交叉过滤 + right-click peek + long-press 反事实
2. **预测时序带（标准图）**：ECharts 自定义主题。观测（charcoal solid）+ 预测均值（red dashed）+ Q05-Q95 fan（red tint α 0.18）+ 阈值 markers + 自定义注释。Easter egg：阈值首次出现时 `流` 字符涟漪 300ms
3. **多模型叠加**：3 fan 不同色系（red / ocean / green）+ KPI 条
4. **跨站相关性矩阵**：按拓扑重排（上游→下游），不字母序
5. **异常 / 告警严重度带**：Datadog 血脉重做。Watch / Elevated / Critical 三层
6. **反事实 / scenario**：参数表单 → agent 调反事实工具 → 第三 fan（dashed grey）
7. **站点剖面 modal-less peek pane**：从右栏滑入，多 tab
8. **报告生成器**：drag-drop layout + `/forecast/r/{slug}` 分享 URL + PDF 导出

### 4.3 聊天侧栏（右栏 360px）

接 `liulian-agent` SSE。Tool calls 渲染成科学笔记本风格：

```
> 显示 Bern 站点 Q95 上周
  [bi-agent] query_forecasts(station_id="aare-bern-2.1", metric="q95", window="-7d..now") -> 17 forecasts
  [bi-agent] add_panel(...) -> panel added
  结果：Bern 的 Q95 历史正常区间内，除 2026-05-09 上游降水推高 12%（你画布右上新面板可见）。
```

JetBrains Mono 用于 method calls；Switzer 用于 plain text；Chicago 样式上标脚注引用 manifest / papers。**无头像气泡**。是笔记本，不是聊天。

### 4.4 空 / 加载 / 错误状态

- **Loading**：暖纸 shimmer skeleton 匹配最终布局
- **Empty**：单行 Fraunces 斜体 + 慷慨留白；无插画
- **Error**：1px hairline 边卡片 + 一行 plain-language 错误 + cmdk-keyed retry + JetBrains Mono 错误码；无图标

### 4.5 键盘纪律

`⌘K` / `Ctrl+K` 命令面板（主导航）/ `j` `k` 行导航 / `gg` `G` 顶 / 底 / `o` 开 peek / `e` 编辑 / `?` 键参考 / `/` 聚焦搜索 / `Esc` 关 / `b` BI agent 侧栏 / `m` 切换地图 / 表格

## 5. `/studio` — Linear-meets-Bloomberg editorial Swiss

`/studio` 是研究工程师之家。密集、键盘优先、读起来像科学仪器。

### 5.1 侧栏

56px collapsed 默认，hover 展到 220px。条目：Data · Train · Inference · Insight · Agents · Admin · Docs。活跃条目：左侧 6×6 红点（替代侧条）。

### 5.2 列表页范式

不对称两列：**60% 表** + **40% Fraunces 斜体问答解释器**。表格密集 monotype 数字，j/k 行导航，右键 peek。锚图：每页一条 sparkbar。

### 5.3 详情页范式

三纵带：标题带 + 指标带 + Logs/Params/Forecasts/Audit 带（h/l tab nav）。

### 5.4 命令面板 (⌘K)

600×400 modal 顶部居中。Switzer 24px 输入。结果分组（Data / Train / Inference / Insight / Agents / Settings / Help）。每行 14px Switzer 标 + 11px JetBrains Mono 右键提示。选中行左侧 6×6 红点。

**⌘K 是 `/studio` 主导航**。侧栏只是 fallback。

### 5.5 表单

单列，max-width 480px，慷慨节奏。Labels 在 input 上（Caption 风格 uppercase）。Focus ring 1px `--unibe-red-deep`，无 glow。主提交按钮 `--ink-charcoal` 填 / 暖纸字，6px 半径。Cancel 是文字链接。

## 6. `/(marketing)` — landing

唯一允许编辑级 *修辞* 的地方。杂志感、克制、科学。

### 6.1 Hero

全幅 swisstopo 卫星瓦片（Aare 流域 Brienzersee），88% 暖纸覆盖。中心 display 文字：

> **Liquid Intelligence for Time.**

Fraunces 96-144px responsive，weight 500 roman，"L*U*iquid" 和 "LIUL*I*AN" 字母 `U` 用斜体 WONK 1 weight 600，UniBe 红。

副线（恰好 17 词）：

> *Open-source production stack for spatio-temporal AI: a research-grade model zoo wrapped in production-grade BI.*

实时计数条：`2,143 sensors live · 12 models in benchmark · 4 agents ready`。JetBrains Mono 13px uppercase。

### 6.2 三带

不靠卡片，靠**纵向规则**（1px hairlines）分。每带：Fraunces 28px 段标题 + Switzer 17px 段（max 65ch）+ 一张来自 live 平台的实图 + 段尾斜体锚句：

1. **Manifest** — *每个数据集都从契约开始。*
2. **Train** — *三十个模型，一个运行时。*
3. **Forecast** — *从预测到水文学家的风险简报，二十秒。*

### 6.3 内嵌 demo 视频

1440×900 BI 画布截屏，auto-play muted loop，60 秒，viewport entry 触发。1px hairline 边，10px 半径。

### 6.4 CTA

页脚单句：

> *Open SwissRiver demo →*

无按钮 / 无 badge / 无 "Schedule a demo" 表单。**箭头就是按钮**。hover 时箭头右移 4px / 200ms ease-out-quart。

## 7. Agent 对话流

3 agent persona；同引擎，不同工具集。SSE 事件形（继承自 liulian-agent）：

```
event: thinking      data: { message: ... }
event: trace         data: { step, detail }
event: intent        data: { intent, entities }
event: tool_call     data: { tool, input }
event: tool_result   data: { tool, output }
event: response      data: { text, references }
event: suggestions   data: [...]
event: done          data: null
```

### 7.1-7.3 三 agent

- **data agent**：tools `list_files / summarise_csv / propose_manifest / validate_manifest / detect_topology / detect_seasonality`。开场：*"我把数据集塑造成 manifest。丢一个 CSV 路径或 MinIO URI 给我；你总在我写之前批准。"*
- **model agent**：tools `list_models / recommend_model / propose_hpo_space / read_run_logs / diagnose_failed_run / compare_runs`。开场：*"给我数据集 + 视野，我推荐模型 + HPO 空间。我读失败的 run log。不你点头，我不训。"*
- **bi agent**：tools `query_forecasts / add_panel / set_filter / create_alert_rule / export_report`。开场：*"用大白话问。例：'显示 Q95 上周超 850 的站。' '比较 TimesNet 和 Chronos 在 Bern。' '上游有站到 elevated 就提醒我。'"*

### 7.4 成本 + 安全 UX

每消息成本指示器（右下，JetBrains Mono 10px `--ink-faint`）：`$0.0023 · 1840 tokens in / 412 out · DeepSeek V4`

成本上限到达 banner：`--pastel-yellow` 背景 + 一行 Fraunces 斜体提示

Provider 降级 banner（如 Gemini 在 CN 不可用）：`--pastel-blue` 背景 + 同 liulian `DegradationBanner.tsx` 形状

## 8. Mobile UX

详见英文版 §8。三 tab（Expo Router）：Home / Forecast / Alerts。Brand tokens 从 `@liulian/design-tokens` RN 导出。字体通过 `expo-font` 加载离线 ttf。

- **Home**：科学新闻报眉（mono 10px）+ Fraunces 28px 问候 + Switzer 15px 摘要 + 单 CTA + 最近告警卡片
- **Forecast tab**：单站 picker + Victory Native XL fan chart（与 web panel 2 同品牌）
- **Alerts tab**：严重度带垂直版
- **Quick forecast 模态**：选站 + 视野（24h/48h/7d）→ Chronos-2 零样本 → fan chart 内嵌
- **Push notifications**：alert 规则触发；点开站点深链接

## 9. 文档表面 `/docs`

contentlayer + MDX。body 用 **Fraunces**（是的，serif body）— 读起来像论文，不像 Stripe API 参考。code block JetBrains Mono。figure caption Switzer。

### 9.2 Tufte 血脉两栏

prose 列（max 65ch）+ margin notes 列（13px Switzer 斜体 / 脚注引用 / "see also" 链接）

### 9.3 内嵌实图

每当文档示例需要 `print(metric)` 输出时，实际 ECharts viz 内嵌（数据从 `/api/public/...` 拉）。文档吃平台的狗粮。

## 10. 无障碍

- 暖纸上 body 文字对比度 14.5:1
- 所有动作键盘可达；focus ring 1px `--unibe-red-deep`
- `prefers-reduced-motion` 把所有动效塌缩到 instant
- 所有图表带隐藏的 `<table>` 给屏幕阅读器
- 颜色绝非唯一状态信号（状态也用排版）
- 通过 `next-intl` 全栈本地化；按 `claude-config:bilingual-docs` 规则双语（en + zh canonical）

## 11. 品牌语态字段库

详见英文版 §11 表。文案在 `liulian-design-system/copy.json`。重点 keys：

| key | EN | ZH |
|---|---|---|
| hero.title | Liquid Intelligence for Time. | 流动的智能，关于时间。 |
| hero.subtitle | Open-source production stack for spatio-temporal AI: a research-grade model zoo wrapped in production-grade BI. | 时空 AI 的开源生产栈：研究级模型库，工业级商务智能。 |
| hero.cta | Open SwissRiver demo → | 打开瑞士河流演示 → |
| band.manifest.title | Every dataset starts with a contract. | 每份数据集都从一份契约开始。 |
| band.train.title | Thirty models, one runtime. | 三十个模型，一个运行时。 |
| band.forecast.title | From prediction to a hydrologist's risk brief, in twenty seconds. | 从预测到水文学家的风险简报，二十秒。 |

## 12 / 13. 开放问题 / PR-merge 审计

详见英文版。每个前端 PR 跑 `conventions/UI_AUDIT_CHECKLIST.md`（46 项 11 节）。reviewer 把审计块粘到 PR body，未勾选项阻塞合并。

非可妥协项：视觉原创 + 品牌典籍 + AI-slop test + category-reflex test + gui-demo cross-check。

---

*详细英文权威版：`PLATFORM_DESIGN.md`。架构上游：`PLATFORM_BLUEPRINT.md`。一周执行：`ONE_WEEK_SPRINT.md`。复用计划：`LIULIAN_REUSE_MAP.md`。每 PR 必过项：`conventions/UI_AUDIT_CHECKLIST.md`。*
