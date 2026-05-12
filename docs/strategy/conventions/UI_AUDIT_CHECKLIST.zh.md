---
title: LIULIAN UI 审计清单（中文镜像）
status: convention（每个触前端 PR 必过）
parent: UI_AUDIT_CHECKLIST.md
---

# LIULIAN UI 审计清单（中文版）

> **语言：** [English](UI_AUDIT_CHECKLIST.md) | 中文

每个增删可视表面 (HTML/CSS/JSX) 的 PR 在合并前必须每行通过。reviewer 在 PR body 勾每框。任何未勾行阻塞合并。

## A. 神圣规则（不可妥协）

- [ ] **代码复用 vs 视觉原创清晰隔离**。新表面截图不应像 neobanker / Refine.dev / shadcn-template 换文字。(NEOBANKER_REUSE_MAP §0.1)
- [ ] **双语规则尊重**（claude-config bilingual-docs）：用户面向字符串都通过 next-intl key，en + zh 都译，切换可见

## B. 颜色

- [ ] 用 OKLCH-defined tokens（绝不在 `liulian-design-system` 之外用原 hex）
- [ ] 计算样式无 `#000000` / `#FFFFFF`
- [ ] 中性色向品牌色调倾斜（chroma 0.005–0.01）
- [ ] 页面声明色彩策略（Restrained / Committed / Full palette / Drenched）
- [ ] UniBe 红 `#E20613` 仅用在 (a) 品牌 accent、(b) 状态（alert / elevated）、(c) destructive confirm。**绝不**在背景 / 按钮 / hover 上

## C. 排版

- [ ] body 行宽 ≤ 75ch
- [ ] 层级比 ≥ 1.25
- [ ] 字体：Fraunces (display) / Switzer (body) / JetBrains Mono (code, numbers)。**禁** Inter / Roboto / Open Sans
- [ ] 数字列必 tabular numerals (`font-variant-numeric: tabular-nums`)
- [ ] 状态尽量用**排版** (italic / weight)，不用彩 pill

## D. 布局

- [ ] 间距有节奏变化（不是处处一致 padding）
- [ ] 无嵌套卡片
- [ ] 容器只在必要时用
- [ ] 主带间慷慨间距（≥ 96px）
- [ ] hairline 边 (`1px solid var(--hairline)`)；静态无 drop shadow

## E. 动效

- [ ] 仅动 `transform` + `opacity`（不动 layout property）
- [ ] ease-out 指数曲线 (quart / quint / expo)。无 bounce / 弹性
- [ ] Scroll-entry 用 IntersectionObserver，不用 scroll listener
- [ ] hero 无自动 video，除非 muted + loop + ≤ 4 MB

## F. 绝对禁令（match-and-refuse；自动 block）

- [ ] 无 1px 以上**侧条**做色 accent on card / list / callout / alert
- [ ] 无**渐变文本** (`background-clip: text` + 渐变 bg)
- [ ] 无**玻璃化**做装饰（罕见且目的明确）
- [ ] 无**hero-metric 模板**（大数字 + 小 label + sparkline + ring）
- [ ] 无**同款 card grid**（同尺寸 card 重复堆，icon + heading + text）
- [ ] 无 **modal 第一反应**（必先穷尽 inline / progressive 备选）
- [ ] card 或主按钮无 `rounded-full`
- [ ] 标签、代码、alt 内无 **emoji**
- [ ] 无 **AI cliché 文案** ("Elevate" / "Seamless" / "Unleash" / "Next-Gen" / "Game-changer" / "Delve")
- [ ] 用户面向文案无 **em dash** (`—`)。用 逗号 / 冒号 / 分号 / 句号 / 括号
- [ ] 无 **占位名** ("John Doe" / "Acme Corp" / "Lorem Ipsum")。用 `manifests/` 真实站点代码

## G. 品牌典籍检查（LIULIAN-specific）

- [ ] Canvas 色：`--canvas-warm` (#FBFBFA) 或 `--surface-pure` (#FFFFFF)
- [ ] Body 文字：`--ink-charcoal` (#131313)；绝非纯黑
- [ ] Card 边：1px solid `--hairline` (#EAEAEA)，10px 半径
- [ ] 多数页面每视口最多一个 UniBe 红元素可见。forecast 画布上限二（活跃 station marker + 阈值穿越 marker）

## H. 独特 move（正向检查，每页 ≥ 1）

每有意义页选至少一个签名 move：

- [ ] Fraunces wonk-1 italic accent 在单字符
- [ ] 科学新闻报眉（date · run-coordinates）
- [ ] 数字 [0.01, 1000] 之外默认科学计数法
- [ ] Bento grid 不对称（混 tile 尺寸；相邻不同款）
- [ ] 编辑级两栏不对称（60% 表 + 40% 解释器）
- [ ] 领域真切 Easter egg（每页最多一次）

## I. 四项测试（reviewer 在 PR 末跑）

- [ ] **AI-slop 测试**：viewer 会毫不犹豫说"AI 做的"吗？→ 拒
- [ ] **类别反射测试**：色板能从"水文" / "TS forecasting" 独猜出吗？→ 重做色彩策略
- [ ] **neobanker-copy 测试**：页像 neobanker 换文字吗？→ 重做风格
- [ ] **gui-demo 交叉校验**：暖纸 + UniBe 红做唯一 spot + 编辑级排版还在主导吗？→ 不然辩护

## J. 无障碍（尽量委托自动工具）

- [ ] Lighthouse a11y ≥ 95 在改动页（跑 `make audit-a11y`）
- [ ] 所有交互元素键盘可达
- [ ] 文字色对比 AA（body ≥ 4.5:1，大字 ≥ 3:1）

## K. 性能（委托）

- [ ] LCP ≤ 2.5s（M1 MacBook over throttled 4G，跑 `make audit-perf`）
- [ ] 字体 swap 时无 layout shift（`font-display: swap` + size-adjust matched）

---

## Reviewer 笔记格式

PR body 粘这块，每框勾或为未勾框写简短理由：

```markdown
## UI 审计 (UI_AUDIT_CHECKLIST.md)

A. 神圣规则       ✓ 视觉原创（无 neobanker 看相）；双语 ✓
B. 颜色          ✓ Restrained；UniBe 红仅阈值 marker
C. 排版          ✓ Fraunces 32/24/16；指标 tabular nums
D. 布局          ✓ 间距有节奏；hairline 边
E. 动效          ✓ 仅 opacity + transform
F. 绝对禁令     ✓
G. 品牌典籍     ✓
H. 独特         ✓ wonk-1 italic 在 hero `U` + bento 不对称
I. 四测试       ✓ 全通
J. A11y         ✓ Lighthouse 97
K. 性能         ✓ LCP 1.8s

Reviewer: <name>
Date: <iso8601>
Branch: <branch>
```

任何 `~`（部分）或 `✗`（拒），PR 在解决前**不能**合并。

---

*源：`impeccable` 技能（共享设计法）、`minimalist-ui` 技能（高端实用极简协议）、`claude-config:bilingual-docs`、`gui-demo design-report.md`（品牌典籍）。*
