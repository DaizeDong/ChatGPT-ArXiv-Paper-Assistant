---
title: AI 热点管线 — 全 Agent 原生重写 设计方案
date: 2026-06-02
status: 设计冻结草案（spec-only，本轮不写生产代码）
source: 多轮对抗 subagent 辩论（4 立场发散 → 对抗批判 → 首席综合 → 对抗收敛评审，3 轮收敛 stable=true，14 agent）
related:
  - docs/competitive_landscape_2026-06.md（竞品全景，复用源依据）
  - docs/investigation_staleness.md
  - docs/investigation_cross_day_dedup.md
  - docs/investigation_x_coverage.md
---

# 导读（落盘前置说明）

## 用户锁定的 4 项决策（不可违背）
1. **架构**：全 agent 原生重写（编排 + 工具调用 subagent 群），非修修补补。可牺牲开发时间换效果，但实现须优雅简洁、功能强大、不过度工程。
2. **复用**：聚合他人成品输出为一等数据源 **+ 集成竞品开源组件**；目标"下限 = 市面所有产品并集"，再用自有检索查缺补漏 + 交叉验证。
3. **运行**：本地/VPS 上 Claude Code headless 定时跑（cron），吃订阅额度而非按调用计费。
4. **本轮交付**：仅稳定的设计 spec 文档。

## 三大根因落点（辩论中经仓库代码复核后修正，见 §0）
- **跨天重复 ~60%** ← 根因是**缺持久身份**（`story.py` 每天重算 SHA1 story_id、无 centroid/first_seen），而非靠 `apply_cross_day_penalty` 补丁能治。
- **时效性（老论文混入）** ← 根因是信任 source 自报日（**HF `publishedAt` 是平台元数据，≠ arXiv v1 真首发日**）；注意 `get_freshness_date` 的 `fetched_at` 通道在现版本已基本堵上。
- **X 覆盖 ≈ 0** ← 瓶颈在**数据通道**（x_official 需 Pro $5000/月、x_paperpulse 已死），过滤器误杀已部分修复；出路是 twitterapi.io。

## ✅ 4 个开放问题 — 已由用户拍板（v1.3 决策，已整合入正文）
1. **老论文热度复燃通道** → **开启**，且作为**专门通道**呈现。见新增 **§C.4 Resurgence 复燃通道**：触发 = arXiv 版本跃升 OR 多竞品同日聚集超阈值；入门日用**我方观测到的复燃 run-date**（非源自报日，结构上抗污染），独立"复燃"版块呈现，**不污染主 NEW 流的新鲜度保证**。
2. **arXiv 版本计数轮询配额** → **以最高筛选质量为准（CC 定）**：每 run 对窗口内每个 arxiv_id 轮询一次（run 内按 id 去重，不跳过多日 TTL，以最大化版本跃升检出质量），用 arXiv API `id_list` **批量**（每调用 ≤100 id）+ 礼貌限频。见 **§B.4.1 轮询节奏**。
3. **gate_date 跨时区 tie-breaker** → **采纳**。见新增 **§B.3.1 权威整日锚**：存在 arXiv 官方 announced 日 / Crossref 注册日时，`gate_date` 直接取该官方**整日**（本就日粒度、机器无关），覆盖 WebSearch 推导的亚日值，把更多项推入确定性机器无关路径。
4. **竞品上游污染二阶告警** → **采纳**。见 **§E 监控**新增：单一竞品源 `intentionally_dropped_stale_competitor` 比例相对滚动基线突增 → 告警，区分"正常策展老项"与"竞品自身被污染"。

## ✅ 论文筛选管线 — 已由用户拍板（v1.3 新增 §H）
**不重写**论文筛选管线；而是**引入 agent 作为一种新的筛选模态**——把 Claude Code agent 作为与"作者规则 / API 打分"并列的可插拔筛选策略，使筛选不再仅限单次 API 调用，而能做多步推理 + 工具使用（读全文 PDF、查引用、对照近期文献）。见 **§H 论文筛选的 Agent 模态**。

---

# AI 热点管线 — 全 Agent 原生重写统一设计方案 (v1.3, spec-only)

> **v1.3 增量（用户拍板 5 项决策后整合，均为加法，不回退 v1.2 已稳定部分）**：§B.3.1 权威整日锚（gate_date tie-breaker）、§B.4.1 轮询节奏（质量优先+批量）、§C.4 Resurgence 复燃专门通道（观测日入门、抗污染、独立栏）、§E 二阶污染告警、§H 论文筛选 Agent 模态（可插拔策略接口 + 级联路由，不重写现状）。

> 状态：设计冻结草案，本轮**不写生产代码**。
> 骨架：单写者确定性 Kernel + 单编排 Agent + 无状态短生命周期 subagent 群。
> 核心数据模型：持久 Story Store（稳定 ID + 多语言质心 + first_seen + verified_first_date + evidence ledger）。
> 抗漂移规约：拓扑写在代码（不在 LLM 计划里）+ 每个随机 agent 后接确定性 verifier + 永久按内容哈希缓存判定。
>
> **v1.2 变更（本轮，仅修订三处未决问题，不回退已稳定部分）**：
> 1. **§D.3 GapFill 闸门自相矛盾修复**：`⊇` 验收断言**作用域收窄到通过我方核验的竞品条目**，与 §B.5/§D.1 的 max_age + verified_first_date 硬闸一致；被硬闸合法丢弃的"陈旧但被竞品策展"的项不计入 `⊇` 义务，改记 `run_journal` 的 `intentionally_dropped_stale_competitor` 通道，**不再误报为闸门失败**。
> 2. **§B.4/§C.3.1/§G.4 T2 版本刷新路径**：澄清 arXiv **版本计数轮询是一条独立的廉价确定性 Tier-0 读**（每 run 对带 arxiv_id 的 story 重读 `/abs` 版本数），写入 `Story.arxiv_versions`，**与不可变的 `verified_first_date` 判定解耦**；"缓存即冻结"只冻结**首发日**，不冻结单调递增的版本计数。
> 3. **§B.5/§C.3/§E/§G T2-及-gravity 的跨机 first-run 抖动修复 + 诚实契约**：所有**驱动 featured/RESURFACE 门**的日期一律先过**确定性日粒度取整 + 单调 earliest-min**（`gate_date := floor_to_day(min(credible_dates))`），使 sub-day 的 WebSearch 抖动**无法翻转门**；并把 §G 净效果声明**诚实收窄**为"replay/缓存命中 bit 级可复现；从未见过的新项首核是 `[真首发日, claimed]` 内的有界发散，非跨机 bit 一致——这是契约本身"。
>
> v1.1 已稳定部分（单写者 Kernel、固定 DAG、三层日期核验阶梯、TDT 持久质心、闭式 `resurface(S)` 谓词主体、复用 tiering、宪法 1-8 主体）**全部保留不回退**。

---

## 0. 设计前的代码事实核验（影响整份方案，已逐条在仓库验证）

落盘前以**代码事实**为准（本轮再次复核，结论不变）：

| 断言 | 核验结果（文件:行） | 对设计的影响 |
|---|---|---|
| `Story.story_id` 是持久身份 / 已有 centroid 脚手架 | **假**。`story.py:88-93` `_story_id` 是对当日 item 的 `canonical_url+title` 排序后 SHA1；`group_into_stories` 每次重算；`Story`(line 66-77) **无 centroid、无 first_seen**，但**已有 `entity_names: set[str]` 字段(line 71)** | 持久身份是**全新 schema 工作**。`entity_names` 既有字段可直接被 §C.3.1 确定性谓词 T3 复用 |
| `get_freshness_date` 盲目优先 `fetched_at` | **已过时**。`hotspot_sources.py` 现在只有 `_FETCHED_AT_VALID_SOURCES={"github_trend"}` 才用 fetched_at | staleness 的 `fetched_at` 通道**已基本堵上**；残留风险是 HF 用的 `publishedAt`（HF 平台元数据，非 arXiv v1 真首发日） |
| HF 老论文 staleness | `hotspot_hf_papers.py`：用 `paper.publishedAt`，**不等于 arXiv v1 提交日** | Tier-0 必须改读 **arXiv v1 submission date**。这是 staleness 真正根因落点 |
| 跨天去重靠 `apply_cross_day_penalty` | **真**，是 band-aid（`story.py:315-332` difflib+Jaccard 阈值 0.5；`pipeline.py:1718` 调用） | 用持久质心匹配 + gravity-from-first_seen **替换**它，不叠加 |
| 官方 X 账号被 SELF_WORK_PATTERNS 误杀 | **已部分修**（`hotspot_x_common.py` 官方账号放行；误杀只在非官方路径） | X≈0 瓶颈**不在过滤器**，在数据通道（x_official 需 Pro、x_paperpulse 死），twitterapi.io 是出路 |
| pipeline 体量 | `pipeline.py` ~1839 行单体，`story.py` 384 行 | strangler-fig 抽阶段函数有意义，确定性核心保留复用 |

**结论**：力气投在三处真实根因：(1) 持久身份缺失导致跨天重复，(2) 信任 source 自报日（含 HF publishedAt）而非核验真首发日，(3) X 数据通道而非过滤器。

---

## A. 全 Agent 架构形态

### A.1 骨架

```
            ┌─────────────────────────────────────────────┐
   cron ───▶│  run.py  (Kernel：纯 Python 确定性内核)        │
            │  · 拓扑写死在代码（固定阶段 DAG，非 LLM 规划）   │
            │  · 唯一的 Story Store 写者 (single-writer)    │
            │  · 每阶段 (date, stage) 幂等 checkpoint        │
            └───────┬──────────────────────────────────────┘
                    │ 仅在"需要判断/检索/写作"处 dispatch
        ┌───────────┼───────────┬───────────┬─────────────┐
        ▼           ▼           ▼           ▼             ▼
   DateVerify   DedupAdjud   GapFill     XNewsworthy   Synthesize
   (无状态)      (无状态)     (无状态)     (无状态)       (无状态)
        └─ 每个：typed JSON in → typed JSON out，温度0，schema 校验
```

- **编排器 = 薄确定性 Python（run.py），不是 LLM 规划器**。抗漂移头号决策：每天 DAG 形状固定，不由 agent 临场决定跑哪些阶段/谁先核验。
- **subagent 是无状态纯函数**：吃 typed input bundle，吐 typed result；不跨天记忆、不互相通信、不写状态、不决定策略。工具型 worker。
- **唯一状态写者**：只有 Kernel 写 Story Store。采纳 blackboard 的 schema 与"daily output = 持久状态的确定性投影"框架，但用单写者实现，零仪式。

### A.2 固定阶段 DAG（确定性骨架 + 5 个 agent 触点）

| # | 阶段 | 性质 | 说明 |
|---|---|---|---|
| 1 | Harvest | 纯 Python | fan-out 现有 28+ adapter + 复用层 adapter → 统一 `NormalizedItem`（含 `source_tier`+`provenance`） |
| 2 | DateVerify | **agent 触点①** + Tier-0 纯 Python | 结构化权威源零 LLM 直读；残差走 subagent。产出 `verified_first_date/confidence/evidence`。**含每 run 的 arXiv 版本计数 Tier-0 刷新（§B.4.1）** |
| 3 | Freshness/Gravity gate | 纯 Python | 用 `verified_first_date` 经**日粒度取整**做 max-age 硬闸 + HN gravity 衰减（§B.5.1） |
| 4 | Embed | 纯 Python | 多语言 Matryoshka 句向量(title+lede)，模型版本 pin |
| 5 | Cluster(intra-day) | 纯 Python | Union-Find + cosine 自动合并（复用现有逻辑） |
| 6 | StoryStore-Match(cross-day) | 纯 Python(主) + **agent 触点②** | 当日簇 vs 持久质心匹配，分配稳定 `story_id`，标 NEW/ONGOING；模糊带 escalate |
| 7 | GapFill(复用差集补缺) | **agent 触点③** | 竞品并集差集 → 定向 fetch+强制 DateVerify；**差集闸门作用域见 §D.3** |
| 8 | Score | 纯 Python | Altmetric 分级权重的确定性 5 因子打分 |
| 9 | Route+Synthesize | 强/弱确定性 + **agent 触点④⑤** | confidence-aware 路由；X 新闻性判定 + 双语合成；后接确定性 verifier |
| 10 | Project/Render | 纯 Python | Story Store 的确定性 VIEW → web_data + _zh + 日/月/年归档 |

> **过度工程取舍**：拒绝"9-stage content-addressed artifact store + 每 stage JSON Schema + golden file"重型机制。阶段间用普通 JSON checkpoint 落盘即可。拒绝 blackboard 的 4 状态机——日报只需 `NEW/ONGOING` + 一个 `last_surfaced`。

### A.3 工具/MCP 清单

| 用途 | 工具 | 归属阶段 |
|---|---|---|
| arXiv v1 提交日 **+ 版本计数轮询** | arXiv API (`/abs` versions) | Tier-0 DateVerify（首发日**一次性**冻结）+ 版本计数（**每 run 刷新**，§B.4.1） |
| DOI 首发日 | Crossref / DataCite | Tier-0 DateVerify |
| 权威元数据+发布日 | Semantic Scholar / OpenAlex API | Harvest + DateVerify |
| 抗污染下界 | Wayback Machine CDX API | Tier-1 DateVerify |
| 最早提及兜底 | WebSearch + WebFetch | Tier-2 DateVerify / GapFill |
| X 数据通道 | `twitterapi-mcp`(已集成 #10) | Harvest(X) + XNewsworthy |
| 竞品成品吸收 | HF Daily Papers / AINews / agents-radar / Horizon RSS/Pages (WebFetch) | Harvest(复用层) |
| 嵌入 | fastembed/sentence-transformers 本地库(免费，多语言 Matryoshka) | Embed |

---

## B. 日期核验子系统（用户钦定的根因子系统）

**第一原则**：item 自报日/网站日**一律不可信**；Store 存独立**核验**出的 `verified_first_date`——**绝不是 fetched_at、绝不是 source 自报日**——同时驱动 max-age 硬闸**和** gravity 衰减。这使任何"自报日"通道（含 HF publishedAt）在结构上失效。

### B.1 核验范围的硬决策：每条都过 Tier-0

威胁模型是"故意把旧内容盖上今天日期"的项——它有貌似新鲜的 raw date、多源回声、非整月整日，**不触发任何可疑信号**。统一裁决：
- **每一条 featured 候选都必须有 verified_first_date**。
- 但**绝大多数项零成本**：每条先过 Tier-0 结构化直读，只有 Tier-0 拿不到权威日期的**残差**才付 LLM 成本。
- 可疑触发**降级为优先级排序信号**（先核可疑的），**不作为是否核验的门**。

### B.2 三层核验阶梯（命中即停，成本递增）

- **Tier-0（零 LLM，确定性，conf 0.95，覆盖绝大多数）**：
  - 有 `arxiv_id` → 查 arXiv API **v1 submission date**（直接治掉 HF 老论文 bug：读到真 v1 老日期 → gravity 沉底）。
  - 有 DOI → Crossref/DataCite 注册日。
  - GitHub → `created_at`；HF paper → `publishedAt` 仅作 Tier-0 候选之一，与 arXiv v1 取**最早**。
  - **github_trend 例外**：合法使用 observed-trending 日期而非首发日。
- **Tier-1（廉价交叉，news/blog/X，可能含实时 WebSearch，machine-dependent）**：DateVerify subagent 至少交叉 2 个独立信号：(a) **Wayback CDX 最早快照时间戳**（抗污染主力）；(b) 页面 `article:published_time` / JSON-LD `datePublished`；(c) 标题最早可信报道搜索。
- **Tier-2（罕见升级）**：仅"不确定且将被 featured"的项做深搜。

### B.3 判定规则：earliest-credible-date-wins（锁定）
污染几乎总是**向前 backdate 以显新**，故取**最早可信日**击败它。`verified_first_date = min(可信日集合)`；若 claimed_date 明显晚于 Wayback/最早提及 → 标 `stale_date_pollution=true` 用更早日；多源一致→high；无法核实→保守 `min(claimed, fetched)` + low confidence（降权置于折叠线下，不丢弃、不崩溃）。**边界判定做成确定性**（不交给 agent confidence 阈值），避免日间翻转。

### B.3.1 权威整日锚（gate_date tie-breaker，v1.3 决策3）
跨时区边界发散（§G.4 的残余）来自对**亚日时刻**的依赖。修复：当存在**权威机构指派的整日**时，直接以它作 `gate_date`，绕过 WebSearch 推导的亚日值——
- **arXiv announced date**：arXiv 对每篇论文指派一个稳定的 UTC 公布日（与 v1 submission 同源，机器无关、本就日粒度）。带 `arxiv_id` 的项 `gate_date := arXiv announced day`（仍服从 §B.3 的 earliest-min：与其它可信日取最早）。
- **Crossref / DataCite 注册日**：带 DOI 的项 `gate_date := 注册日（整日）`。
- **优先级**：`gate_date := floor_to_utc_day( min( 权威整日锚 ∪ 其它可信日 ) )`；权威整日锚一旦存在即为该项提供**机器无关**的整日下界，把绝大多数学术/官方项推入确定性路径。
- **残余**：仅对**既无 arXiv/DOI 权威锚、又只能靠 WebSearch 定日**的纯网页项，跨机整日发散仍可能存在——这是 §G.4 诚实契约承认且被 `[真首发日, claimed]` 单调界约束的少数情形，不再扩大承诺。

### B.4 成本控制 = 永久缓存（也是 determinism lock）
核验结果按 `content_hash`(url/doi/arxiv_id) **永久存** Store `date_verdicts` 表。**首发日永不改变 → 一条一生只核验一次**；跨天命中缓存零成本零漂移。`date_verdicts` 表是文本快照的**必含部分**（§E），随快照旅行到任何重建机，使重建继承冻结判定。

#### B.4.1 版本计数轮询：可变量，与冻结的首发日解耦（v1.2 修订，回应 issue#2）

**问题**：`date_verdicts` 以 `content_hash(arxiv_id)` 为键；arXiv vN 与 v1 共享同一 arxiv_id → 同 content_hash → 永久缓存命中 → T2 依赖的版本跃升**永不被观测**。

**裁决（明确区分"冻结量"与"单调量"）**：
- **`verified_first_date`（冻结量）**：arXiv **v1** submission date，进 `date_verdicts`，**一生只核一次，永不更新**。这是"缓存即冻结"唯一约束的对象。
- **`Story.arxiv_versions: dict[arxiv_id, int]`（单调量）**：由一条**独立的廉价确定性 Tier-0 读**维护——每个 run，对带 arxiv_id 的活跃 story（在滚动窗口内），重读 arXiv `/abs` 的**版本数**（一次轻量 HTTP，无 LLM、无 WebSearch、确定性），写回 `Story.arxiv_versions`。
- **关键不变式**：`arxiv_versions[id]` **只能单调不减**（`new := max(old, fetched_count)`）；版本计数轮询**永不触碰** `date_verdicts`、**永不改写** `verified_first_date`。
- 因此 §G.4 的"freeze"措辞精确化为：**冻结的是 first_date，不是版本计数**。版本计数的每 run 刷新是确定性的（同一天读同一 arXiv 状态得同一计数），不引入漂移，且让 T2 的版本跃升分支**真正可被观测**。
- 退化处理：arXiv `/abs` 读失败 → 沿用 `arxiv_versions` 旧值（不降级、不阻塞），记 `run_journal` partial。
- **轮询节奏（v1.3 决策2，以最高筛选质量为准）**：每个 run 对滚动窗口内每个 `arxiv_id` 轮询**恰一次**（run 内按 id 去重，避免同 id 重复读）；**不设跨多日 TTL 跳过**——因为我们就是要尽早检出版本跃升来驱动 §C.3.1 T2 与 §C.4 复燃，质量优先于省调用。礼貌性靠**批量**实现：用 arXiv API `id_list` 一次查询多个 id（每调用 ≤100 id），调用间隔遵守 arXiv 限频（~1 req/3s）。窗口内 story 即便数百，也仅数次批量 HTTP，成本可忽略。

### B.5 下游消费
gravity + freshness gate 一律用 `verified_first_date`；`max_age` 硬闸（默认 14d，per-source-family 可配）在入 Store 前丢弃过期项（github_trend 例外）。彻底替代旧 fetched_at 补丁与 HF publishedAt 信任。

#### B.5.1 门用日：日粒度取整 + 单调 min（v1.2 修订，回应 issue#3）

**问题**：first-seen 项首核含 Tier-1/2 实时 WebSearch（machine-dependent），其返回的 `verified_first_date` 可能在**亚日（sub-day）尺度**抖动；该日期同时喂给**连续 gravity** 与 **T2 谓词**，使新项在某天的 featured/not-featured 边界处可能因机器/时刻抖动而翻转。

**裁决（让门只吃确定性的日粒度量）**：定义**门用日**为所有"驱动 featured 闸 / RESURFACE 闸"的日期入口的唯一形式：

```
gate_date(item) := floor_to_utc_day( min( credible_dates(item) ) )
```

- `floor_to_utc_day` 把任意带时刻的 `verified_first_date` 截断到 UTC **整日**（丢弃 H:M:S）。亚日抖动被取整吸收 → **同一天内的 WebSearch 时刻差异不改变 gate_date**。
- `min(credible_dates)` 是 earliest-credible 单调规则；污染只能更早不能更晚，发散夹在 `[真首发日, claimed]`。
- **gravity 在闸用途上读 `gate_date`**（日粒度），使"6 天后压到折叠线下"的衰减判定确定性化；连续 gravity 的细粒度仅用于**同一天内的排序展示**（非门），排序抖动不影响"是否 featured"这一离散结论。
- **T2 谓词读 `gate_date`**（见 §C.3.1），比较 `verified_first_date` 处一律用日粒度值，使 T2 翻转只在**真发生了跨日的更晚首发日**时触发，而非亚日抖动。
- **净效果**：date 驱动的离散门（featured/RESURFACE）对亚日 WebSearch 抖动**鲁棒**；跨机首核即便返回同一日内不同时刻，`gate_date` 相同 → 门结论相同。残余的真发散仅在"两台机对**同一新项**核出了不同 UTC **整日**"这一更罕见情形，且仍被 `[真首发日, claimed]` 单调界约束（见 §G.4 诚实契约）。

---

## C. 去重子系统（根治 ~60% 跨天重叠）

根因是**缺持久身份**（每天独立处理 + 每天重算 SHA1 story_id）。修复 = TDT 持久质心身份，跨天去重成为**身份的涌现属性**而非补丁。

### C.1 三层去重

- **L0 intra-day exact（确定性，零成本）**：`canonicalize_url` 归一 + MinHash/Jaccard(标题) 合并近重复（复用现有逻辑）。
- **L1 semantic + 跨语言（确定性 + 嵌入）**：多语言 Matryoshka 句向量(title+lede)。cosine **>0.72 自动合并**（calibrated 2026-06, mpnet-base-v2; was 0.90 pre-measurement）；中英 `_zh` 同空间编码，同事件中英对 cosine 仍 >0.72 → 合并为一 story。
- **L2 cross-day 持久匹配（治本）**：当日每个存活簇 vs 滚动窗口(14d)内活跃 story 质心做匹配。

### C.2 跨天匹配的两处必须钉死的细节

1. **匹配条件用"质心为主，URL-Jaccard 为辅/加权"，绝不用 AND**。规则：`cosine>=0.72` 即归并；URL-Jaccard 高只作**加分确认**，不作必要条件。
2. **intra-day→story 反收敛**：单一真实事件当日可能裂成 2-3 个簇。必须先把当日簇内部对同一既有 story 的多个匹配**合并指向同一 story_id**，再决定 NEW/ONGOING，否则质心库会为一个事件 accrete 出重复 story。

### C.3 跨天呈现：确定性 Novelty Gate

**RESURFACE 决策必须是确定性代码，绝不交给 LLM novelty 判断**（6 天重复 bug 正藏在这个缝里）：
- item 归并到既有 story → 该 story 标 **ONGOING**，保留原始 `story_id` + `first_seen`。
- gravity 从 `first_seen` 计时（用 §B.5.1 的 `gate_date` 日粒度）：6 天后 T≈144h，衰减**数学上**压到折叠线下，**无需 agent 投票**。
- 仅当 `new_evidence_delta` 越过**确定性阈值**才重新 featured。

#### C.3.1 `new_evidence_delta` 的闭式确定性谓词（锁定，v1.2 微调 T2 比较为日粒度）

```
对每个被标 ONGOING 的 story S，记 last = S.last_surfaced 那次 surface 的状态快照。
定义（全部为 Store-resident 事实，无任何自由文本、无 URL-set churn、无 LLM）：

resurface(S) := T1 OR T2 OR T3

  T1 (tier 跃升):
     max(e.source_tier for e in S.evidence_added_since(last))
        >  max(e.source_tier for e in S.evidence_before(last))
     # 严格大于：出现了比此前任何证据更权威的源（如先前仅 tweet/blog，现出 official/news）

  T2 (更晚的已核验真首发日 / 新 arXiv 版本) —— 全部用日粒度 gate_date 比较:
     EXISTS e in S.evidence_added_since(last) such that
        gate_date(e)  >  last.surfaced_verified_max     # 后者亦为日粒度
     OR  EXISTS arxiv_id in S such that
        S.arxiv_versions[arxiv_id]  >  last.surfaced_arxiv_versions[arxiv_id]
     # 严格晚于上次 surface 时 story 的最大 gate_date（日粒度，亚日抖动不触发）；
     # 或同一 arXiv 论文版本计数严格增加（由 §B.4.1 的独立 Tier-0 轮询喂入，非冻结判定）

  T3 (新具名实体):
     ( S.entity_names  \  last.surfaced_entity_names )  ≠  ∅
     # 出现了上次 surface 时 story 尚未包含的具名实体（复用既有 Story.entity_names 字段）

明确排除（NOT triggers）:
  - URL-set churn：evidence URL 集合变化量（即便 churn 40%）不触发任何 T。
  - 任何 free-text "progress"/"significance"/"novelty" 判断——RESURFACE 路径零 LLM。
  - source_tier 未严格上升的同级证据增加（如又来 3 条同级 tweet）不触发 T1。
  - 亚日（sub-day）的 verified_first_date 抖动——T2 比较前一律 floor_to_utc_day，亚日差异被取整吸收。
```

**关键性质**：
- T1/T2/T3 全部是对 Store 中已落库结构化字段的纯布尔比较：`source_tier`（整数）、`gate_date`（**日粒度** `verified_first_date`）、`arxiv_versions`（整数计数，单调）、`entity_names`（既有 `set[str]`）。无一项需读 evidence 正文或做语义判断。
- `evidence_added_since(last)` / `evidence_before(last)` / `last.surfaced_*` 均为 Store 每次 surface 时确定性记录的快照字段。
- RESURFACE 是 story 状态的**确定性函数**，两次运行 bit 级一致，跨天不漂移。**40% URL churn 与亚日抖动均被构造性排除**。
- **替换**（非叠加）现有 `apply_cross_day_penalty`，消除双抑制无优先级问题。
- Store schema 须新增以支撑此谓词（默认值向后兼容）：`Story.last_surfaced`(date)、`Story.surfaced_verified_max`(**日粒度** date)、`Story.surfaced_entity_names`(set)、`Story.surfaced_max_tier`(int)、`Story.arxiv_versions`(dict[arxiv_id,int])、`Story.surfaced_arxiv_versions`(dict[arxiv_id,int]，上次 surface 时的版本计数快照)；evidence ledger 每条带 `added_at`(run date) 以支撑 `*_since(last)`。

### C.4 Resurgence 复燃通道（v1.3 决策1：专门通道，抗污染）

**动机**：§B.5/§D.3 的 max_age 硬闸（默认 14d）正确地把"被竞品重新策展的合法老论文"挡在**主 NEW 流**外（verification 优先于 recall）。但确有"真热度复燃"的老论文（出了重磅新版、或一夜间被全行业重提）值得呈现。用户裁决：**开一条专门通道**，而非放松主流闸门。

**抗污染的关键设计——入门日用"我方观测日"而非"源自报日"**：复燃通道**不信任**任何源声称的"新日期"（那正是污染向量）；它只信任**我方自己观测到复燃事件的 run-date**（`Story.resurged_at`）。这是我方一手观测、**无法被上游 backdate**，因此复燃通道在结构上与 §B 的抗污染第一原则同构，且因独立成栏**不削弱主 NEW 流的新鲜度保证**。

**确定性触发谓词（零 LLM，仅读 Store 结构化事实）**：

```
仅对 gate_date(S) 超出 max_age 的 story（即"老"项）评估复燃：

resurge(S) := R1 OR R2

  R1 (版本跃升复燃，严格 gate against 上次复燃快照):
     EXISTS arxiv_id in S such that
        S.arxiv_versions[arxiv_id] > S.surfaced_arxiv_versions[arxiv_id]
     # 由 §B.4.1 独立 Tier-0 轮询喂入；老论文出新版 = 作者主动更新 = 真信号。
     # 因比较 against surfaced_arxiv_versions 快照（surface 时更新），每个新版本仅触发一次。

  R2 (跨竞品同日聚集复燃，冷却闸编入谓词 —— v1.3 修订，闭合 INV4):
     count_distinct_competitor_sources( S.evidence_added_today
                                        where provenance ∈ 复用层竞品源 )
        >= RESURGE_MIN_COMPETITORS                                  # 默认 3，可配
     AND ( S.surfaced_resurged_at IS None
           OR run_date - S.surfaced_resurged_at >= RESURGE_COOLDOWN_DAYS )   # 默认 7，可配
     # ≥3 个相互独立的竞品聚合器同一 run 同时重提 = 真复燃；冷却闸消费 surfaced_resurged_at
     # 快照，使"同一组竞品连续多日重提同一老项"在冷却期内只触发一次，构造性关闭每日重复
     # （与 §C.3.1 T-谓词同纪律：去抖由对 surface 快照的闭式比较决定，不靠散文约定）。
```

**呈现与去抖**：
- **两个不同的快照字段，职责分离**：`Story.resurged_at` = **首次**触发复燃的 run-date（一旦置定，永不覆盖，用于 gravity 计时起点）；`Story.surfaced_resurged_at` = **每次**在复燃栏 surface 时置为当次 run-date（用于 R2 冷却闸比较）。两者初值均 None。
- 复燃通道的 gravity 从 `resurged_at` 计时（用 §B.5.1 的日粒度 `gate_date(resurged_at)`），同样 6 天后自然沉底。
- 在独立的 **"复燃 / Resurgence"** 版块呈现，每条**同时标注**原始 v1 首发日（诚实）+ 复燃原因（`vN 新版` 或 `N 家聚合器同日重提`）。
- **每次复燃栏 surface 时，Kernel 确定性地更新快照**：`surfaced_resurged_at := run_date` 且 `surfaced_arxiv_versions := arxiv_versions`（与 §C.3.1 的 `record_surface` 同机制）。这使 R1 的版本比较与 R2 的冷却比较都 gate against 已更新的快照——**再复燃只在出现更新版本（R1）或冷却期满后又一次 ≥3 竞品聚集（R2）时发生**；URL churn、亚日抖动、同源重复、以及"同一组竞品连续每日重提"均不触发，构造性关闭复燃栏的每日重复退化路径（INV4 在 R2 上闭合）。
- **与 R2 抗"竞品共有污染"的关系**：§D.4 指出多竞品**共同**污染时多数表决会失效——但 R2 的产出**不进主 NEW 流、不参与 featured 新鲜度承诺**，只进显式标注"复燃"的隔离栏，且每条都带原始 v1 老日期可见；即便偶有共有污染混入，影响被限制在一个诚实标注的次级栏内，不损害主流可信度。
- Store schema 新增：`Story.resurged_at`(date|None)、`Story.surfaced_resurged_at`(date|None，上次在复燃栏 surface 的快照)。

---

## D. 复用子系统（下限 = 市面并集，可验证契约）

复用是**一等 ingestion TIER**；"下限=市面并集"从口号升格为每日可验证的集合运算闸门。

### D.1 复用契约（tiering discipline）
每个复用项进**同一** `NormalizedItem` schema，带 `provenance` + `source_tier` 权重（Altmetric 式：news 高 … tweet 低，**直接复用不自研权重**），并**强制过同一 DateVerify + dedup 闸门**——继承 recall 不继承 staleness/重复（reuse 从属于 verification，吸收竞品永不吸收其陈旧）。

### D.2 一等复用源（先证明扩大并集再加，避免源蔓延）
HF Daily Papers（社区投票，免费）、AINews recap（已部分实现）、Scholar Inbox / Semantic Scholar Research Feeds、OpenAlex、agents-radar / Horizon 的 RSS/Pages 输出、twitterapi.io。**开源组件吸收而非重写**：Horizon 的跨平台 URL 归一作 L0 前置；多语言嵌入库作 L1/L2 共享原语（免费本地，不训练、不付费 embedding API）。

### D.3 GapFill 可验证闸门（核心契约，v1.2 修订：作用域收窄到通过核验项，回应 issue#1）

**自相矛盾根因**：v1.1 的 `our_coverage ⊇ 任一竞品当日条目` 与 §B.5/§D.1 的 max_age 14d + verified_first_date 硬闸相撞——设计上我们**就是要**丢弃竞品重新策展的合法老项（如 v1 日期 >14d 的 arXiv 论文）。无作用域的 `⊇` 会在每个竞品 surface 陈旧策展项的日子**误报**，把"下限=并集"契约变成永久假警。

**修订裁决（recall 继承以核验为条件，与 §D.4 一致）**：

每日做集合运算，但先对竞品条目应用我方核验闸定义"应被覆盖的对象"：

```
competitor_items                := ⋃(HF, AINews, agents-radar, Horizon, Scholar Inbox 今日条目)

eligible_competitor_items       := { c ∈ competitor_items :
                                        passes_dateverify(c)               # 通过我方 Tier-0/1/2
                                        AND within_max_age(gate_date(c)) }  # 日粒度 verified_first_date 在 max_age 内

dropped_stale_competitor_items  := competitor_items \ eligible_competitor_items
```

- **差集闸门**作用于 `eligible_competitor_items`：`gap := eligible_competitor_items \ our_coverage` → GapFill agent 自动定向 fetch + 强制 DateVerify 后补入（差集 = 别人有、且通过我方核验、我们却没有）。
- **可验证 acceptance test（收窄后）**：
  ```
  assert our_coverage ⊇ eligible_competitor_items
  # 即：assert our_coverage ⊇ { c ∈ competitor_items : passes_dateverify(c) AND within_max_age(gate_date(c)) }
  ```
  仅对**通过我方 DateVerify 且在 max_age 内**的竞品条目负 `⊇` 义务。
- **被合法丢弃的项不计入闸门失败**：`dropped_stale_competitor_items` 显式**排除**于 `⊇` 义务之外（recall 继承以 verification 为条件，与 §D.4 同构），改写入 `run_journal` 的专用通道 `intentionally_dropped_stale_competitor`（带每条的 `gate_date`/`reason`），作为**可观测的设计行为**而非告警。
- 这样：竞品 re-feature 一篇 v1 日期 2023 的论文 → 该项进 `dropped_stale_competitor_items`、记 journal、**不触发假警**；而真正"别人有的新热点我们漏了" → 进 `gap`、被 GapFill 补回。两类被干净分离。
- 再用自有 WebSearch/官方源检索找"连竞品都漏的真热点"形成**超集**。

### D.4 抗"复用=继承错误"的循环陷阱
若多个竞品**共同**把 2023 老论文当今日 trending（污染信号是聚合器**共有**的），多数表决 cross-validator 会**批准**而非拒绝。**唯一真正救场的是 verified_first_date 闸门**。故本方案**不设独立 cross-validator agent**（与 GapFill 职责重叠）；交叉验证语义由 **DateVerify 闸门**承担——任何复用项的日期都被 arXiv/Wayback 硬锚覆盖，多竞品共识无法 override 硬锚。§D.3 的作用域收窄正是这一原则在验收断言上的落地：被硬锚判旧的竞品共识项理应落在 `⊇` 义务之外。

---

## E. 运行 / 可靠性

- **执行模型**：VPS/本地 cron 每日触发 `claude -p` headless，Kernel 驱动 DAG，dispatch subagent。**吃 Claude 订阅额度而非按调用计费**，适配重推理（DateVerify/Synthesize）。
- **幂等**：每阶段按 `(target_date, stage)` checkpoint 落盘 + Store upsert(url 唯一键)。重跑从最后未完成阶段 resume；已完成阶段 no-op；同日重跑结果一致（date/dedup 缓存命中）。复用现有 `remedy_missed_dates` 回填语义。
- **失败重试**：阶段级有界重试(指数退避 max3)。单源 adapter 失败 → degrade 跳过(记 partial)。agent 失败 → 确定性回退：DateVerify 失败 → `min(claimed,fetched)`+low conf 折叠线下；Synthesize 失败 → 现有 `_heuristic_takeaways` 兜底。版本计数轮询失败 → 沿用旧 `arxiv_versions`。**保证永远出报告**。
- **密钥**：twitterapi.io key / LLM key / X token 在 VPS `.env` / systemd EnvironmentFile，**不进 repo、不进 Actions**。
- **与 Actions 关系（职责分离）**：VPS 跑完产出 `web_data + _zh` 提交到 repo；**GitHub Actions 退化为纯 Publisher**——只做翻译资产校验(防 `_merge_zh` 静默失败)+ 静态站日/月/年部署。
- **状态持久与备份**：Story Store(SQLite + 嵌入 sidecar)驻 VPS。**不把二进制 SQLite 提交进 git**。每日 run 后 **dump 为 schema 化 JSON/SQL 文本快照**推到专用分支做审计+可重建；二进制库定期对象存储备份。架构身份**不依赖单台机器**。
  - **跨机复现的精确边界（v1.2 精确化，回应 issue#3）**：跨机决定性由**随文本快照旅行的 `date_verdicts` 缓存**保证（必含部分）。重建机继承所有已冻结首发日判定。仅**从未进缓存的首见项**触发一次新鲜（含 Tier-1/2 实时 WebSearch，可能机器相关）核验；该首核被两道确定性收敛夹住：(1) **earliest-credible 单调 min**（污染只能更早，发散夹在 `[真首发日, claimed]`）；(2) **§B.5.1 的日粒度 `gate_date` 取整**——驱动 featured/RESURFACE 门的只是 UTC 整日值，亚日 WebSearch 抖动**无法翻转门**。因此：**"一生只核验一次"是 per-Store 的；跨独立首核不保证逐 bit 相同，但 (a) 快照随迁使新机几乎总命中缓存不触发首核，(b) 即便触发，门只吃日粒度值，亚日抖动被吸收，仅"两机核出不同 UTC 整日"这一罕见情形才可能在边界翻转，且仍被 `[真首发日, claimed]` 单调界约束**。推快照到审计分支**务必包含 `date_verdicts` 表**，否则重建退化为全量重新首核。
  - 版本计数 `arxiv_versions` **不进 `date_verdicts`**（它是单调可变量，每 run 由 Tier-0 轻读重建），故不需随快照冻结；新机重建时按当时 arXiv 状态重读即可，单调不减性质保证不回退。
- **监控**：每 run 写 `run_journal`(JSON：每源条数、核验队列大小、去重归并数、agent token/延迟、stage 计时、`intentionally_dropped_stale_competitor` 列表)。异常(X 产出=0、核验失败率高、GapFill `⊇`(收窄后) 断言失败)推送告警(Feishu/PushNotification 可选)。
- **二阶污染告警（v1.3 决策4）**：对每个复用层竞品源，按 run 统计其 `intentionally_dropped_stale_competitor` 占该源当日条数的比例，与该源**滚动基线**(trailing N-run 中位数，默认 N=14)比较；当某单一源的 stale-drop 比例相对基线**突增**(默认 ≥2× 基线且绝对占比 ≥30%)→ 推送**专项告警**："竞品源 X 疑似上游被污染"。这把"正常策展老项"(各源平稳的小比例 drop)与"某竞品突然大量重提老内容"(单源比例突刺)在确定性阈值上区分开；阈值/基线窗口可配，纯读 `run_journal` 聚合，零额外抓取。
- **VPS 单点缓解**：幂等可补跑 + 保留 Actions 降级路径（关 agent 跑确定性精简版）+ run 失败告警。

---

## F. 优雅性 / 模块边界 / 迁移

### F.1 模块边界（每个：做什么 / 接口 / 依赖）

| 模块 | 做什么 | 接口 | 依赖 |
|---|---|---|---|
| **Kernel(run.py)** | 拓扑写死、单写者、checkpoint、dispatch | `run(date, stage?, force?) -> manifest` | Store, 各阶段函数 |
| **Harvest** | 28+原生源+复用源→NormalizedItem | `harvest(date)->list[NormalizedItem]` | adapters(复用现有) |
| **Story Store** | 持久身份+质心+verdict 缓存+surface 快照+版本计数 | `match_or_create / upsert_evidence / active_stories(window) / get_verdict / put_verdict / record_surface(story) / refresh_arxiv_versions(story)` | SQLite + 嵌入 sidecar |
| **DateVerify** | Tier0 纯函数 + subagent + **版本计数 Tier-0 轮询(§B.4.1)** | `verify(item)->{verified_first_date,confidence,evidence}` / `poll_arxiv_versions(arxiv_id)->int` | arXiv/Crossref/S2/Wayback/Web |
| **GateDate(纯函数)** | `min`+`floor_to_utc_day` 把任意 verified date 转门用日粒度值 | `gate_date(item)->date` | 无（纯确定性） |
| **Dedup** | L0/L1 确定性 + L2 adjudicator | `cluster_intraday / match_crossday` | 嵌入库 |
| **NoveltyGate** | 闭式 `resurface(S)` 谓词(§C.3.1) | `resurface(story)->bool`（纯函数，零 LLM） | Store 快照字段 + GateDate |
| **Score** | 确定性 5 因子 + gravity-from-first_seen(日粒度门) | `score(stories)->stories` | source_tiers 权重 + GateDate |
| **GapFill** | 竞品差集补缺(作用域=eligible) | `gapfill(our, eligible_competitors)->new_items` | 复用 adapters + DateVerify |
| **Synthesize** | 双语 headline/summary + verifier | `synth(story)->{en,zh}` then validate | LLM(温度0) |
| **Project/Render** | Store→web_data+_zh+归档 | `render(store, date)` | 现有 renderers(不动) |

每模块**可独立单测**（agent 环节用 record/replay 缓存响应做确定性单测；`GateDate`/`NoveltyGate.resurface` 因纯函数可用 golden fixtures 全覆盖）。

### F.2 迁移路线图（strangler-fig，一 commit 一阶段，docs 同步）

- **阶段0（地基）**：建 Story Store(SQLite + verdict 缓存表 + 嵌入 sidecar + surface 快照字段 `last_surfaced/surfaced_verified_max/surfaced_entity_names/surfaced_max_tier/arxiv_versions/surfaced_arxiv_versions`) + run_journal；为 `HotspotItem`/Story 加 `verified_first_date/provenance/story_id(持久)/first_seen/centroid` 字段(默认 None，向后兼容)。**不要先建重型 observability**。
- **阶段1（纯 Python 治本，无 agent）**：max-age 硬闸 + gravity on `gate_date(verified_first_date)` + Tier-0 arXiv v1 直读修 HF 老论文。**此步证明 8/41 老论文不需要 agent 就能沉底**。golden snapshot 锁回归。
- **阶段2（持久去重）**：StoryStore-Match(质心匹配 + 反收敛 + **闭式 NoveltyGate §C.3.1**)**替换** `apply_cross_day_penalty`；L0/L1 确定性去重。验证 6 天重复塌缩为单次 NEW。**回填须先去重**（30 天历史报告本身含 6 天重复，naive backfill 会 mint 6 个 first_seen 污染锚）——回填用一次性离线 job，对历史先跑同一去重再 seed first_seen。NoveltyGate 谓词随此阶段附 golden fixture 单测（构造 URL-churn-only 与亚日抖动输入断言 `resurface=False`；构造 tier 跃升/新 entity/新 arXiv 版本/跨日更晚首发日各断言 `resurface=True`）。
- **阶段3（DateVerify subagent）**：Tier-1/2 上线(news/X)，永久缓存；**版本计数 Tier-0 轮询(§B.4.1)上线**；确认快照 dump 含 `date_verdicts`、不含 `arxiv_versions` 冻结。
- **阶段4（复用层 + GapFill）**：新增复用 adapter + GapFill **作用域收窄差集闸门(§D.3)** + 每日 `⊇ eligible` 断言 + `intentionally_dropped_stale_competitor` journal 通道。
- **阶段5（X 通道）**：twitterapi.io MCP harvest（过滤器已基本对，重点是通道与量）。
- **阶段6（编排骨架 + 运行迁移）**：generate_daily_hotspot_report 改写为 Kernel DAG(阶段化+checkpoint)，接入 Synthesize/Dedup adjudicator agent；VPS cron + headless；Actions 降级 Publisher。

每阶段独立上线/回滚/测试；双语 + 日/月/年归档**全程保留**（渲染层不动到阶段6）。现有资产**大量复用**，只重写"每天独立处理"的控制流与身份语义。

---

## G. 抗 Agent 不确定性规约（统一宪法，保证每日可复现）

1. **机械占多数**：fetch/normalize/exact-dedup/embed/score/render/deploy 全纯 Python，给定输入 bit 级可复现 → 管线大部分**无法漂移**。
2. **拓扑写在代码**：DAG 与阶段顺序在 run.py，**绝不在 LLM 计划里**——每天 run 形状固定。
3. **每个随机 agent 后接确定性 verifier**：DateVerify 输出被 Wayback/arXiv **硬锚覆盖**；Synthesize 输出 schema 校验（双语对齐 + **每条 cited evidence URL 必须真实存在于 story** = 廉价强力抗幻觉门），不合格则重试→确定性回退。
4. **缓存即冻结（含跨机边界与版本计数区分，v1.2 精确化）**：DateVerify **首发日**判定按 content_hash 永久缓存，一条一生判一次，跨天稳定。**"冻结"仅作用于 `verified_first_date`**；`arxiv_versions` 是**单调递增的版本计数**，由独立廉价确定性 Tier-0 每 run 刷新（`new := max(old, fetched)`），**不进 `date_verdicts`、不被冻结**——这给 §C.3.1 的 T2 版本分支提供可观测的刷新路径，同时不破坏首发日的不可变性。跨机层面，"一生一次"是 per-Store 保证：`date_verdicts` 随快照旅行，重建机继承冻结判定；只有从未进缓存的**首见项**才触发新鲜核验，被 (a) earliest-credible 单调 min + (b) **日粒度 gate_date 取整** 双重收敛。审计分支快照**必含 `date_verdicts`**。
5. **持久身份作锚 + 闭式 Novelty Gate**：story_id + first_seen 一旦确定即不可变，同事件永映同 id，headline 确定后 ONGOING 沿用，根除"同事件每天换措辞"漂移。**RESURFACE 由 §C.3.1 的闭式布尔谓词 `resurface(S)` 决定，仅读 Store-resident 结构化事实（source_tier 整数、日粒度 gate_date、arxiv 版本计数整数、entity_names），零 LLM、零自由文本、零 URL-set churn、零亚日抖动**——6 天重复 bug 被构造性关闭的核心机制。
6. **温度0 + 强制结构化输出 + pin 模型/嵌入版本**：模型 ID、嵌入模型 ID、温度记入 manifest；两次 run 漂移可定位到单一阶段 diff。
7. **嵌入模型版本绑定 centroid**：每条 centroid 存 `model_id`；升级模型须 re-embed 迁移。
8. **门只吃日粒度日期（v1.2 新增，回应 issue#3）**：所有"是否 featured / 是否 RESURFACE"的离散门，输入日期一律经 `GateDate = floor_to_utc_day(min(credible_dates))`。连续 gravity 的亚日精度只用于**同日内排序展示**，不参与离散门判定。这使亚日级 WebSearch 抖动在结构上**无法翻转门结论**。
9. **回归快照 + nightly replay-diff**：固定历史 raw 输入断言阶段输出 story 集合与打分稳定；每夜对前一天 replay 并 diff 检测回归（丢弃 9-stage golden files 重型基建）。

**净效果（v1.2 诚实收窄，回应 issue#3）**：
- **Replay / 缓存命中**：重跑过去某天 **bit 级可复现**——所有判定（首发日、dedup、resurface、score、render）命中冻结缓存或纯确定性计算。
- **前向 run**：仅在**世界真的变了**（新 verified evidence 满足闭式 `resurface(S)`，或版本计数单调增）时改变离散门结论；亚日抖动与 URL churn 被构造性排除。
- **从未见过的新项首核（诚实契约，非掩盖）**：在一台**全新机器**上对某新项首核时，Tier-1/2 实时 WebSearch 是非确定的——但 (a) 门只吃 `gate_date`（UTC 整日），亚日抖动无法翻转；(b) 残余发散仅限"两机核出不同 UTC **整日**"这一情形，且被 `[真首发日, claimed]` 单调界夹住。**故契约逐字为**：*"replay/缓存命中跨机 bit 可复现；从未见过的新项首核是 `[真首发日, claimed]` 内、日粒度上的有界发散，不保证跨机 bit 一致；这是契约本身，而非完全前向确定性的暗含承诺。"* 因 `date_verdicts` 快照随迁，生产实践中新机重建几乎总命中缓存、不触发首核，发散在稳态下不可见。

---

## H. 论文筛选管线：Agent 作为一种筛选模态（v1.3 决策5，不重写）

**裁决**：**保留**现有论文筛选管线（作者规则 + h-index 门 + LLM-API 标题/摘要打分 + 阈值）不动；**新增** agent 作为一种**可插拔筛选策略**，使筛选不再仅限单次无状态 API 调用，而能做**多步推理 + 工具使用**（读全文 PDF、查引用、对照近期文献、核验是否真匹配细腻的兴趣描述）。这与热点管线"在判断点把 agent 当工具"的哲学同构。

### H.1 统一筛选策略接口（最小改动的关键）
把"如何判断一篇论文该不该留"抽象成一个**策略接口**，现有逻辑与新 agent 各实现之，管线只依赖接口：

```
PaperFilter.judge(paper, criteria) -> {
    keep: bool,
    relevance: float,      # 与现有打分同量纲，复用现排序/阈值
    novelty: float,
    rationale: str,        # 简短理由（可入站点展示）
    evidence: list[url],   # agent 模态可附其查证的引用/对照来源
}
```

- **`ApiScoreFilter`（现状）**：把现有 `prompts/paper/` + 单次 LLM 调用打分包成该接口的一个实现——**零行为变化**，是默认实现。
- **`RuleFilter`（现状）**：作者白名单 + h-index 门，同样包成接口实现（前置硬过滤）。
- **`AgentFilter`（新增）**：在 VPS 的 Claude Code 环境派生一个**无状态短命 subagent**，输入 `(paper 元数据/abstract, 可选全文 PDF, criteria 文本)`，允许其调用工具（WebFetch 全文、Semantic Scholar/OpenAlex 查引用与近期相关工作、arXiv 查版本），输出同一 typed 结构。温度0 + 强制结构化输出 + 后接确定性 verifier（schema 校验 + evidence URL 真实性校验），完全复用 §G 第 3/6 条抗不确定性纪律。

### H.2 编排：级联路由（复用 confidence-aware 思想，控成本）
不要对每篇论文都跑昂贵的 agent。沿用热点管线已验证的 confidence-aware 路由：

```
RuleFilter 硬过滤
  → ApiScoreFilter 廉价打分
      → 高分确信留 / 低分确信弃：直接定（绝大多数）
      → 仅"阈值边界的模糊带" → AgentFilter 深判（读全文+查引用）
```

config 可选三种模式：`api_only`(现状默认) / `agent_only`(全 agent，质量最高成本最高) / `cascade`(推荐，边界带才上 agent)。

### H.3 边界、复用与一致性
- **模块边界**：`AgentFilter` 是 §F.1 模块表外的**独立可插拔模块**，接口 = `judge()`，依赖 = Claude Code subagent + 检索工具；可用 record/replay 缓存响应做确定性单测。
- **复用一等源**：判断"该不该留"时，AgentFilter 可顺带消费 §D 的复用信号（HF 投票、Scholar Inbox 命中、Altmetric 热度）作为佐证，使论文管线也享受"下限=市面并集"。
- **与热点管线共栈**：两条管线共用同一 VPS Claude Code 运行环境、同一抗不确定性宪法（§G）、同一 record/replay 测试范式；论文管线的渲染/归档/双语层**不动**。
- **迁移**：作为独立阶段（在 §F.2 迁移路线图之后或并行）——先抽 `PaperFilter` 接口包住现状（零行为变化、加测试网），再加 `AgentFilter` 并默认 `cascade`，灰度对比 agent 模态与纯 API 的留存差异后再调阈值。
