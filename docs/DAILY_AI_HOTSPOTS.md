# Daily AI Hotspots：一个故事驱动的多源 AI 热点聚合系统

更新日期：2026-04-04

## 摘要

本文档描述本仓库当前实现的 `Daily AI Hotspots` 系统。该系统面向"同一天前沿 AI 领域发生了什么"这一问题，聚合论文、新闻通讯、官方博客、GitHub、社区讨论及 X/Twitter 等 12 类异构来源。核心架构为"故事驱动"（story-centric）：每个原始信号首先经 LLM 批量语义标注获取事件类型、实体和重要性，再通过四轮 Union-Find 将同一真实事件的多条报道合并为统一的 Story。Story 经五因子评分、多样性选择和可选的 LLM 摘要合成，最终输出为静态 JSON 站点。该设计消除了早期基于标题相似度的脆弱聚类，以及 topic 与 source section 之间的结构性重复，同时通过严格的 36 小时新鲜度门控确保输出仅包含当日信息。

## 1. 引言

前沿 AI 信息的传播呈现明显碎片化特征。论文首先出现在 arXiv 或 Hugging Face，产品动态由官方博客和新闻稿发布，开发者工具集中在 GitHub 和社区论坛，而 X/Twitter 承载最快速的讨论与扩散。

早期版本采用"保守聚类 → 14 因子启发式评分 → 置信度分流 → 选择性 LLM 复审"的架构。该设计的核心矛盾在于：(1) topic 和 source section 两条独立数据路径导致同一事件在页面上重复出现；(2) 14 个启发式因子（FRONTIERNESS, TECHNICAL_DEPTH, RESONANCE 等）手工调权，难以解释且常不准确；(3) LLM 仅在最后的 screening 阶段介入，前面全靠正则和关键词匹配；(4) 基于标题 token 重叠的聚类脆弱——同一事件的不同标题措辞可能 Jaccard 值很低而无法合并。

当前版本引入 **Story** 作为核心数据单元。一个 Story 代表一个真实世界事件（产品发布、融资、论文等），多个来源对同一事件的报道被合并进同一个 Story。Topic section 和 source section 均从同一个 Story 列表派生，使得跨 section 重复在结构上不可能出现。

## 2. 系统架构概览

当前 pipeline 由七个串行阶段组成：

```
Stage 1  FETCH        12+ 来源 → raw_items[]
Stage 2  FILTER       URL一致性 + 无日期过滤 + 36h新鲜度门控
Stage 3  ENRICH       LLM 批量标注 or 启发式回退 → EnrichedItem[]
Stage 4  GROUP        四轮 Union-Find → Story[]
Stage 5  SCORE        五因子公式 → 带分数的 Story[]
Stage 6  SELECT       多样性约束选择 → featured / watchlist / categories
Stage 7  SYNTHESIZE   LLM 摘要合成 or 启发式回退 → 最终 report
```

最终产物为结构化 JSON report，由 web data builder 转换为前端 payload，部署为静态站点。

## 3. 多源数据采集

### 3.1 来源分类

系统从以下来源家族采集原始信号，并统一标准化为 `HotspotItem`：

| 来源家族 | 实现模块 | 来源角色 | 典型数据源 |
|---------|---------|---------|----------|
| 官方博客 | `hotspot_official_blogs.py` | `official_news` | OpenAI、Anthropic、Google、Meta、NVIDIA 等 30 个博客 |
| 编辑综述 | `hotspot_roundups.py` | `headline_consensus` / `editorial_depth` | The Rundown AI、The Neuron、The Batch、Import AI 等 10 个 newsletter |
| 分析 Feed | `hotspot_analysis_feeds.py` | `editorial_depth` | 专业 RSS 分析源 |
| 论文趋势 | `hotspot_hf_papers.py` | `paper_trending` | Hugging Face trending papers |
| 本地论文 | `hotspot_local_papers.py` | `research_backbone` | 本仓库每日论文筛选结果 |
| GitHub | `hotspot_github.py` | `github_trend` | GitHub trending repos |
| Hacker News | `hotspot_hn.py` | `hn_discussion` | HN 热门 AI 讨论 |
| Reddit | `hotspot_reddit.py` | `community_heat` | r/MachineLearning、r/LocalLLaMA 等 |
| AINews | `hotspot_ainews.py` | `community_heat` | AINews (smol.ai) RSS |
| X 官方 | `hotspot_x_official.py` | `official_news` | 基于权威账号库的 X API 抓取 |
| X AINews Recap | `hotspot_x_ainews.py` | `community_heat` | AINews 中的 Twitter recap |
| X PaperPulse | `hotspot_x_paperpulse.py` | `paper_trending` | PaperPulse 研究人员 feed |

### 3.2 统一表示层

所有来源的原始信号被映射为 `HotspotItem` 结构：

```python
@dataclass
class HotspotItem:
    source_id: str
    source_name: str
    source_role: str        # official_news, paper_trending, community_heat, ...
    source_type: str        # blog_announcement, paper, discussion, repo, ...
    title: str
    summary: str
    url: str
    canonical_url: str
    published_at: str | None
    tags: list[str]
    authors: list[str]
    metadata: dict[str, Any]
```

系统后续所有操作都基于 `source_role`（信息链中的角色）而非具体来源站点来区分权重，使得添加新来源不需要修改核心逻辑。

### 3.3 日期解析

`parse_datetime()` 函数统一处理所有来源的日期格式：

- ISO 8601（含 `Z` 后缀、时区偏移、毫秒）
- RFC 822（RSS 标准：`Wed, 02 Apr 2026 14:00:00 GMT`）
- 日期字符串（`YYYY-MM-DD`、`Apr 2, 2026`、`April 2, 2026`）
- Unix 时间戳（由各来源预转换为 ISO）

所有结果统一转为 UTC 时区。对于 newsletter 来源，当 HTML 中无法提取 `<time>` 标签时，系统依次尝试页面级 `<meta>`、JSON-LD `datePublished` 和 `target_date` 作为 fallback。

## 4. 过滤层

采集到的原始信号在进入 enrichment 之前经过三层过滤：

### 4.1 URL-标题一致性过滤

`url_title_consistent()` 检测标题与 URL 指向不一致的条目（常见于 newsletter 中链接指向不相关页面的情况），将其移除。

### 4.2 无日期过滤

所有缺少 `published_at` 字段的条目被直接移除。这一策略基于以下判断：无法确认发布日期的条目无法验证其新鲜度，是潜在的过期内容污染源。早期版本允许无日期的 `official_news` 条目通过交叉验证保留，但实践表明这些条目大多是旧产品页面而非当日新闻。

对于 Playwright/LLM 模式采集的 SPA 站点和 LLM 提取的条目，系统使用 `target_date` 作为 `published_at` 的 fallback，确保实时抓取的内容不会因缺少日期而被丢弃。

### 4.3 36 小时新鲜度门控与 `fetched_at` 语义

所有有日期的条目必须满足新鲜度门控，即有效日期在目标日期前 36 小时以内。36 小时窗口的设计考虑了跨时区和深夜发布的情况。

新鲜度判断通过 `get_freshness_date()` 函数统一处理，优先使用 `metadata["fetched_at"]`（采集/trending 日期），回退到 `published_at`（原始发布日期）。这一设计解决了 HF 论文（`publishedAt` 是 arXiv 提交日期而非 trending 日期）和 GitHub 仓库（`created_at` 是创建日期而非 trending 日期）的日期语义问题——"3 天前提交但今天才上 trending"的内容不再被错误过滤。

该过滤统一应用于所有来源——包括论文、GitHub repo、newsletter、官方博客。系统不再对不同来源使用不同的时间窗口。

## 5. LLM 语义标注

过滤后的条目进入 enrichment 阶段，为每个条目补充结构化语义信息。

### 5.1 标注内容

每个条目被标注以下字段，封装为 `EnrichedItem`：

```python
@dataclass
class EnrichedItem:
    item: HotspotItem
    event_type: str       # product_release, funding, acquisition, research_paper,
                          # tooling, industry_move, opinion, tutorial, recap, other
    entities: list[dict]  # [{name: str, type: str}]
    summary: str          # 标准化的两句事实描述
    importance: int       # 1-10
    same_event_as: int | None  # 批内同事件交叉引用
```

### 5.2 LLM 批量标注

在 `openai` 模式下，`enrich_items_batch()` 将条目按批次（默认 20 条/批）提交给 LLM，每批提供 index、标题、摘要前 200 字符、来源名称和来源角色。LLM 返回 JSON 数组，包含每个条目的 `event_type`、`entities`、`summary`、`importance` 和 `same_event_as`。

`same_event_as` 字段是 LLM 标注的核心增值：它允许模型在同一批次内识别"这两条报道说的是同一件事"，为后续 Story 合并提供第一层证据。

单个条目的 LLM 标注失败时，系统 fallback 到启发式标注，确保管线不因个别解析错误中断。LLM 提取的实体与启发式提取的实体会被合并，以提高召回率。

### 5.3 启发式回退

在 `heuristic` 模式下，`enrich_items_heuristic()` 使用正则模式匹配确定 `event_type`，基于 `source_role` 权重和元数据信号计算 `importance`，通过正则提取实体。该路径不产生 `same_event_as` 引用。

## 6. Story 构建：四轮 Union-Find

Enrichment 之后，系统将多个报道同一事件的 `EnrichedItem` 合并为 `Story`。

### 6.1 数据模型

```python
@dataclass
class Story:
    story_id: str              # 基于代表性条目的 SHA1
    canonical_item: EnrichedItem  # 最高权重的代表性条目
    items: list[EnrichedItem]
    event_type: str
    entity_names: set[str]
    category: str              # 从 event_type 映射
    score: float
    headline: str
    summary: str
    why_it_matters: str
    key_takeaways: list[str]
```

### 6.2 合并算法

`group_into_stories()` 使用 Union-Find 数据结构执行四轮合并，任一轮触发即合并：

**第一轮：LLM 交叉引用。** 如果两个条目的 `same_event_as` 指向对方（或指向同一第三方），直接合并。这一轮完全依赖 LLM 标注，是最精确但覆盖最窄的合并条件。

**第二轮：共享 URL/ID。** 如果两个条目共享规范化 URL、arXiv ID 或 GitHub 仓库 URL，合并。这覆盖了"多个来源链接到同一原始资源"的场景。

**第三轮：实体共现。** 如果两个条目至少共享一个非泛化实体且标题有一定重叠，合并。例外：论文-论文对不通过此轮合并，避免将讨论同一模型但本质不同的论文错误合并。

**第四轮：标题相似度。** 三种匹配方式任一触发即合并：(1) 标题包含关系（短标题 token ≥ 80% 出现在长标题中）；(2) Jaccard 相似度 ≥ 0.55；(3) `SequenceMatcher.ratio() ≥ 0.65`（捕获同义词替换，如 "acquires" vs "buys"）。同样排除论文-论文对。LLM enrichment 阶段会按主要实体排序后再分批，使相关条目更可能同批处理，提升 `same_event_as` 跨批覆盖率。

### 6.3 代表性条目选择

每个 Story 的代表性条目（`canonical_item`）选取权重最高的来源：

```python
canonical = max(items, key=lambda ei: (
    SOURCE_ROLE_WEIGHTS[ei.item.source_role],
    ei.importance,
    len(ei.item.summary)
))
```

`SOURCE_ROLE_WEIGHTS` 反映信息链中的角色权威性：

| 角色 | 权重 |
|------|------|
| `official_news` | 6.0 |
| `editorial_depth` | 4.0 |
| `research_backbone` | 3.5 |
| `paper_trending` | 3.0 |
| `github_trend` | 2.5 |
| `builder_momentum` | 2.5 |
| `community_heat` | 2.0 |
| `headline_consensus` | 1.5 |
| `hn_discussion` | 1.2 |

## 7. Story 评分

每个 Story 通过五因子公式计算最终分数。

### 7.1 评分公式

```
raw = (source_weight_sum + evidence_breadth + avg_importance) × event_weight × freshness
score = min(10.0, raw / normalizer)
```

其中：

- **source_weight_sum** = Σ SOURCE_ROLE_WEIGHTS[item.source_role]，上限 25.0
- **evidence_breadth** = 独立来源 ID 数 × 1.5 + 独立来源类型数 × 0.8
- **avg_importance** = 所有条目 importance 的均值（来自 LLM 或启发式）
- **event_weight** = 事件类型权重（product_release: 2.0, funding: 1.8, research_paper: 1.5, tooling: 1.3, opinion: 0.3 等）。对于 opinion 类型，如果关联实体包含行业关键人物（如 Sam Altman、Geoffrey Hinton、Yann LeCun 等 20 位），权重从 0.3 提升到 1.2
- **freshness** = 基于最新条目有效日期（优先 `fetched_at`，回退 `published_at`）的衰减（< 12h: 1.0, 12-24h: 0.8, 24-36h: 0.6, > 36h: 0.4）
- **normalizer** = 三段式动态归一化：P50 映射到 5.0，P95 映射到 9.5，max 映射到 10.0，各段内线性插值。替代了早期硬编码的 `/5.5`，确保全分数范围都有区分度

### 7.2 与早期评分的对比

早期版本使用 14 个启发式因子（FRONTIERNESS, TECHNICAL_DEPTH, RESONANCE, ACTIONABILITY, EVIDENCE_STRENGTH, HYPE_PENALTY, CONFIDENCE 等），手工设定权重和阈值组合。当前五因子公式的每个分量都有明确物理意义：有多少来源报道了它、来源有多权威、LLM 认为它多重要、它是什么类型的事件、它有多新。

## 8. Story 选择与分类

### 8.1 多样性约束选择

`select_and_categorize()` 从评分后的 Story 列表中选出：

- **featured**：默认 5 个，每个 category 最多 3 个，每个来源最多 2 个
- **watchlist**：默认 3 个，应用相同多样性约束
- **categories**：剩余 Story 按 `event_type → category` 映射分组

如果首轮选择未达目标数量，第二轮放宽约束（提高 per-category 上限）。

### 8.2 Category 映射

```
product_release  → Product Release
funding          → Market Signal
acquisition      → Market Signal
research_paper   → Research
tooling          → Tooling
industry_move    → Industry Update
opinion          → Industry Update
tutorial         → Tooling
recap            → Other
other            → Other
```

## 9. 摘要合成

进入 featured 的 Story 通过 `apply_digest_synthesis()` 生成结构化摘要：

- **headline**：一行新闻标题
- **summary_short**：2-3 句核心事实
- **why_it_matters**：为什么值得关注
- **key_takeaways**：3-5 个要点

在 `openai` 模式下由 LLM 生成，`heuristic` 模式下从代表性条目的标题和摘要中提取。

## 10. Web Data 构建与展示

### 10.1 Payload 结构

`build_daily_hotspot_web_payload()` 将 report 转换为前端优化的 JSON payload：

```
payload
├── meta              # 日期、导航链接、统计数字
├── featured_topics   # 今日要闻（CompactTopic[]）
├── category_sections # 分类话题（按 event_type 分组）
├── long_tail_sections # 长尾话题
├── watchlist         # 关注列表
├── paper_spotlight   # 论文聚焦（daily_hot / new_frontier）
├── source_sections   # 信息源条目（6 个来源家族）
├── topic_summary     # 多源话题摘要
├── usage             # API 使用量统计
└── costs             # 成本统计
```

### 10.2 Source Section 去重

Source section 构建时应用多层去重：

1. **日期过滤**：`_is_item_on_date()` 检查 36h 窗口
2. **URL 去重**：全局 canonical URL 去重
3. **标题去重**：Jaccard ≥ 0.55 或 80% 包含关系的标题视为重复
4. **低质量过滤**：28 个正则模式过滤导航文本、CTA、下载链接等
5. **工程讨论过滤**：`engineering_score ≥ 0.45` 的纯工程讨论被移除

六个来源家族各有独立的展示上限：official (10)、market-signals (6)、analysis (8)、papers (12)、github (6)、industry (6)。

### 10.3 前端页面结构

前端使用 React + Vite 构建，页面从上到下展示：

1. **今日要闻**（FeaturedStories）：featured topics 的卡片视图，含标题、摘要、关键要点
2. **论文聚焦**（Paper Spotlight）：按 daily_hot / new_frontier 分组的论文列表
3. **全部话题**（CategoryRadar）：category_sections + long_tail_sections + watchlist 的统一视图
4. **其他动态**（Other Updates）：source section 中未被任何 topic 覆盖且非论文的条目
5. **每日用量**（Usage）：API 调用统计

"其他动态"仅展示不属于任何 topic（无 `topic_refs`）且非论文家族的条目，消除了早期 "Source Feed" 与 "All Topics" 之间的内容重复。

### 10.4 国际化

前端支持中英文切换，通过 `I18nProvider` 和 `useI18n()` hook 实现。UI 标签使用 `t()` 函数翻译，数据字段使用 `tz()` 函数在 `_zh` 后缀变体和原始值之间切换。翻译由 `scripts/translate_hotspot_web_data.py` 批量调用 OpenAI API 生成并写入数据文件。

## 11. X/Twitter 权威账号体系

X 是最容易引入噪声的来源，系统对其采用比其他来源更严格的约束。

### 11.1 权威账号库

系统维护一个可更新的 `authority registry`，种子来自手工整理的官方账号、`follow-the-ai-leaders` 和 PaperPulse researcher authors。运行时种子被刷新为缓存化的权威账号库，作为 X API 抓取的唯一账号集合。

### 11.2 新闻性过滤

即使账号权威，也并非所有内容都值得进入热点系统。系统过滤：招聘/活动安排、webinar/workshop chatter、泛聊天与情绪表达、self-paper announce、低信息量 AI 邻近帖子。

## 12. 构建与可复现性

整个系统是静态可构建的：

1. `scripts/generate_daily_hotspots.py` — 执行 7 阶段 pipeline，输出 report JSON 和 normalized items
2. `scripts/rebuild_hotspot_web_data.py` — 从保存的 report 和 normalized items 重建 web data
3. `scripts/translate_hotspot_web_data.py` — 批量生成中文翻译
4. 前端 `npm run build` — 构建静态站点

所有中间产物保存在 `out/` 目录下，支持完整的历史回放和问题排查。

## 13. 运行时配置

### 13.1 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_topics` | 5 | featured story 数量 |
| `target_watchlist_topics` | 3 | watchlist 数量 |
| `target_category_topics` | 16 | category section 话题数上限 |
| `max_raw_items` | 240 | 采集条目上限 |
| `enrich_batch_size` | 20 | LLM 标注批次大小 |
| `freshness_hours` | 36 | 新鲜度窗口（小时） |
| `paper_spotlight_daily_hot_score_cutoff` | 15 | 论文热度阈值 |

### 13.2 来源配置

- `configs/hotspot/official_blogs.json` — 30 个官方博客（RSS / HTML / Playwright / LLM 四种模式）
- `configs/hotspot/roundup_sites.json` — 10 个 newsletter（daily / weekly 两种节奏）
- `configs/hotspot/source_tiers.json` — 6 级来源可信度分层，40+ 来源 ID 映射
- `configs/hotspot/x_authority_seeds.json` — X 权威账号种子库

## 14. 局限性

### 14.1 LLM 标注质量受限于上下文

每个条目仅提供标题和 200 字符摘要用于 LLM 标注，`importance` 和 `same_event_as` 的准确度受限于这一窄窗口。系统通过启发式后补和实体合并部分缓解此问题。Enrichment 阶段按主要实体排序后分批，提升了批内交叉引用覆盖率，但跨批交叉引用仍依赖后续 Union-Find 的 URL/实体/标题相似度合并。

### 14.2 X 覆盖受限于账号库

权威账号库需要定期更新，否则会遗漏新兴高质量来源。

### 14.3 评分归一化对极端分布的敏感性

动态归一化使用 P90 作为参考点。当数据量极少（<5 条）或分布极度偏斜时，归一化因子可能不稳定。下限保护（normalizer ≥ 0.5）部分缓解此问题。

## 附录 A：实现文件映射

| 文件 | 职责 |
|------|------|
| `arxiv_assistant/hotspots/pipeline.py` | 主 pipeline 流程 |
| `arxiv_assistant/hotspots/enrich.py` | LLM 批量标注与启发式回退 |
| `arxiv_assistant/hotspots/story.py` | Story 数据模型、Union-Find 合并、评分、选择 |
| `arxiv_assistant/filters/filter_hotspots.py` | 工程讨论评分、摘要合成、辅助函数 |
| `arxiv_assistant/utils/hotspot/hotspot_cluster.py` | 标题相似度、实体提取、来源权重 |
| `arxiv_assistant/utils/hotspot/hotspot_web_data.py` | Web payload 构建、source section、新鲜度过滤 |
| `arxiv_assistant/utils/hotspot/hotspot_sources.py` | 日期解析、URL 标准化、新鲜度判断 |
| `arxiv_assistant/utils/hotspot/hotspot_schema.py` | HotspotItem 数据模型 |
| `arxiv_assistant/apis/hotspot/hotspot_official_blogs.py` | 30 个官方博客适配器 |
| `arxiv_assistant/apis/hotspot/hotspot_roundups.py` | Newsletter 爬虫 |
| `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py` | HF trending papers |
| `arxiv_assistant/apis/hotspot/hotspot_github.py` | GitHub trending repos |
| `arxiv_assistant/apis/hotspot/hotspot_reddit.py` | Reddit 抓取 |
| `arxiv_assistant/apis/hotspot/hotspot_hn.py` | Hacker News 抓取 |
| `arxiv_assistant/apis/hotspot/hotspot_x_official.py` | X 权威账号抓取 |

## 附录 B：运行命令

```bash
# 生成当日热点报告（自动选择 openai/heuristic 模式）
python -X utf8 scripts/generate_daily_hotspots.py --mode auto --force

# 从已有 report 重建 web data
python -X utf8 scripts/rebuild_hotspot_web_data.py

# 批量翻译 web data 为中文
python -X utf8 scripts/translate_hotspot_web_data.py

# 启动前端开发服务器
cd web && npm run dev
```
