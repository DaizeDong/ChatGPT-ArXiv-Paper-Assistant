# Daily AI Hotspots：一个面向前沿 AI 信号的多源、成本感知热点聚合系统

更新日期：2026-03-23

## 摘要

本文档描述本仓库当前实现的 `Daily AI Hotspots` 算法系统。该系统面向“同一天内前沿 AI 领域发生了什么”这一问题，聚合论文、新闻通讯、官方博客、GitHub、社区讨论以及 X/Twitter 等异构来源，先将原始信号统一映射为结构化条目，再通过保守聚类、确定性打分、置信度分流和选择性 LLM 复审得到最终热点结果。系统设计的核心矛盾在于：一方面需要尽可能覆盖同日重要动态，另一方面又必须抑制社交噪声、控制 API 成本，并保持结果可解释、可重建。当前实现采用“规则优先、模型补充”的混合架构，通过跨源共振、证据强度、权威来源优先和动态评审预算来提升热点质量。最终产物为一个 source-first 的静态热点站点，重点突出跨来源一致出现的高价值主题，而非单一来源的偶发噪声。

## 1. 引言

前沿 AI 信息的传播呈现明显碎片化特征。研究论文通常首先出现在 arXiv 或 Hugging Face，产品动态和平台变更更常由官方账号、博客或新闻稿发布，开发者工具动向则集中在 GitHub 和社区论坛，而 X/Twitter 往往承载最快速的讨论与扩散。如果简单地把这些来源混合排序，系统通常会退化成两种低质量形态之一：

- 过度偏论文，只能反映研究面，无法覆盖产品、生态和社区层面的关键变化；
- 过度偏社交热度，被低信息量讨论、宣传性内容和单点爆款牵着走。

`Daily AI Hotspots` 的目标并不是构建一个“AI 全网抓取器”，也不是单纯替代 `Personalized Daily Arxiv Paper`。它更像一个面向同日信号的聚合与筛选系统，试图回答下面这个问题：

> 在当天所有高质量 AI 信息源中，哪些事件、发布、研究或讨论真正构成了值得关注的热点？

围绕这个目标，当前算法同时追求以下四点：

1. 尽量覆盖不同来源家族，而不是只服务于单一信息形态；
2. 将“多个不同来源指向同一主题”视为热点成立的重要条件；
3. 将 LLM 用在最需要复审的边界样本上，而不是无差别筛一遍全部候选；
4. 保持产物完全静态化、可回放、可在 GitHub Pages 上复现。

## 2. 系统目标

当前实现围绕以下目标展开：

- 在保证质量的前提下扩大同日覆盖面，而不是把页面压缩成极少数“精选摘要”；
- 优先保留来自官方、公司、研究人员或高信任编辑源的证据，而不是简单追逐热度；
- 允许单源条目在列表中保留，但只有跨源支持的主题才能进入真正的热点摘要层；
- 通过确定性过滤和动态预算降低 OpenAI 调用开销；
- 让结果可以从 `out/` 目录完全重建，而不依赖任何在线状态。

该系统不试图实现实时流式推荐，也不试图覆盖所有 AI 社交噪声。它的定位是一个高质量、可复现的每日热点聚合器。

## 3. 多源数据输入

系统首先从多个来源家族收集原始信号，并将其统一标准化为 `HotspotItem`。

### 3.1 研究与论文源

研究类信号主要来自：

- 本仓库已经筛出的每日论文结果；
- Hugging Face trending papers；
- 官方或研究导向的博客源。

这一层天然提供较强的 `frontierness` 和 `technical_depth`，但如果单独依赖，会导致系统退化为“另一种论文榜单”。

### 3.2 编辑与综述源

编辑类信号主要来自：

- AINews / smol.ai；
- `configs/hotspot/roundup_sites.json` 中登记的 roundup/newsletter 源；
- 官方博客与厂商博客适配器。

这类来源的价值在于，它们往往代表对当天事件的二次整理，因此对“跨源共识”具有较强指示意义。

### 3.3 构建者与讨论源

补充性来源包括：

- GitHub 趋势与工具源；
- Hacker News 讨论源。

它们有助于捕捉开发者动向、工具流行度和社区反馈，但同时也更容易混入单点噪声，因此在排序中并不被直接视为高权威证据。

### 3.4 X/Twitter 相关源

当前系统将 X 视为最重要的前沿动向来源之一，但不再采用“大面积抓取后再清洗”的策略，而是只通过受控高质量入口接入：

- 基于权威账号库的官方 recent-search 抓取；
- AINews 中的 `AI Twitter Recap`；
- PaperPulse 的研究人员 feed。

也就是说，X 不再是一个开放噪声池，而是经过账号质量和新闻性双重约束后的信号源。

## 4. 统一表示层

所有原始信号首先被转换为统一的结构化条目。一个标准化后的 `HotspotItem` 至少包含以下信息：

- 标题；
- 简短摘要；
- 规范化 URL；
- 来源 ID 与来源名称；
- 来源角色 `source_role`；
- 来源类型 `source_type`；
- 发布时间；
- 标签；
- 元数据，例如 stars、upvotes、activity、是否官方等。

这里最重要的抽象不是“它来自哪个网站”，而是“它在信息链中的角色”。当前系统使用的角色包括：

- `research_backbone`
- `paper_trending`
- `official_news`
- `community_heat`
- `headline_consensus`
- `builder_momentum`
- `github_trend`
- `hn_discussion`

后续的聚类、排序、筛选和展示都依赖这一层角色抽象，而不是硬编码每个来源的特殊逻辑。

## 5. 主题构建：保守的相似性聚类

在进入热点筛选之前，系统先将同一事件或同一主题的不同条目聚合为 cluster。

### 5.1 聚类匹配规则

两个条目只有在满足较强证据时才会被合并：

- 规范化 URL 相同；
- arXiv ID 相同；
- GitHub 仓库 URL 相同；
- 或者标题具有足够强的非泛化词重叠。

其中论文聚类采用了更保守的规则。两篇仅仅讨论相似任务的论文不会被简单合并，这可以避免系统把宽泛研究方向误判成一个具体热点。

### 5.2 cluster 的确定性分数

每个 cluster 都会先获得一个确定性分数，用于后续预算分配与候选预排序。其组成主要包括：

- 来源角色权重；
- 不同 source ID 的数量；
- 不同 source type 的数量；
- stars、upvotes、HN score 等源内热度；
- 是否官方；
- 是否关联 GitHub 仓库。

这个分数并不直接等于最终热点分，而是后续候选筛选与资源分配的第一层先验。

## 6. 确定性主题特征

每个 cluster 在被转换为 topic 候选后，会进一步计算一组可解释的中间特征。当前实现中，最重要的特征包括：

- `FRONTIERNESS`
- `TECHNICAL_DEPTH`
- `CROSS_SOURCE_RESONANCE`
- `ACTIONABILITY`
- `EVIDENCE_STRENGTH`
- `HYPE_PENALTY`
- `CONFIDENCE`

这些中间变量再被压缩成用户更容易理解的分数：

- `QUALITY`
- `HEAT`
- `IMPORTANCE`
- `FINAL_SCORE`

其逻辑可概括为：

```text
QUALITY    <- frontierness, technical_depth, importance, evidence
HEAT       <- cross-source resonance + item-level activity
IMPORTANCE <- importance, evidence, frontierness
FINAL      <- quality + heat + importance + actionability + evidence - hype_penalty
```

这种拆分的价值在于，系统显式地区分了三类常被混淆的概念：

- 技术价值；
- 社区或市场关注度；
- 证据是否充分。

一个主题可以“很热但证据不足”，也可以“技术上强但传播不足”，还可以“很重要但当前充满噪声”。只有把这些维度拆开，系统才不会被单一指标劫持。

## 7. 置信度感知筛选与 API 成本控制

当前系统最关键的成本控制机制，是将 LLM 只用于必要的边界案例，而不是对所有候选一视同仁。

### 7.1 四路预分流

在任何 OpenAI 调用之前，候选 topic 会先被划分到以下几类：

- `auto_keep`
- `auto_watch`
- `auto_drop`
- `review`

此外，系统还保留一个实际很重要的 `heuristic_only` 路径，用于那些值得出现在广覆盖层里、但不值得消耗 LLM 成本复审的尾部候选。

该分流主要依据以下量：

- `FINAL_SCORE`
- `CONFIDENCE`
- `EVIDENCE_STRENGTH`
- `CROSS_SOURCE_RESONANCE`
- 来源数量；
- 是否包含官方来源；
- 是否只是孤立的 repo / discussion / 单篇 paper。

### 7.2 这一设计的意义

如果不做这一步，系统会在大量明显样本上浪费成本，例如：

- 弱单源 GitHub repo；
- 孤立社区讨论片段；
- 低证据 paper-only 尾部主题；
- 已经足够明确的多源官方发布。

当前设计只把真正模糊、但又可能有价值的候选送入模型审核。

### 7.3 动态复审预算

当前 review 队列不是固定大小，而是根据当天 `featured/watchlist` 的目标数量动态计算。这意味着：

- 热点较少的日期不会无意义消耗大预算；
- 热点较多的日期仍能给高价值候选足够复审空间；
- API 成本更接近“当天的真实不确定性”，而不是“候选总量”。

## 8. LLM 辅助筛选

进入 `review` 阶段的候选会通过 OpenAI 接受结构化筛选。模型输出包括：

- 类别；
- 是否进入当日热点；
- 是否进入 watchlist；
- 简短摘要；
- 为什么值得关注；
- 与排序兼容的数值判断。

随后，这些结果会被重新规范化回统一的 topic 结构，因此下游逻辑无需区分一个主题是“纯规则产生”还是“LLM 复审产生”。

为了提高 token 利用率，当前实现还做了两项额外优化：

- 每个 cluster 只放少量代表性 evidence；
- digest synthesis 阶段去掉不必要的 URL，并限制每个主题可携带的证据条数。

因此，LLM 在系统中的角色不是“全面替代规则”，而是“补足规则在边界案例上的不确定性”。

## 9. 最终主题选择

在 heuristic 与 LLM 结果合并之后，系统会进入最后的 trimming 阶段，以形成：

- `featured_topics`
- `watchlist`
- `category_sections`
- `long_tail_sections`

这一阶段并不是简单按分排序，而是明确考虑组合质量，避免：

- featured 被过多孤立论文占满；
- featured 被弱 GitHub-only 项目占满；
- 某一个来源家族过度刷屏。

因此，最终首页更像一个“有结构的编辑前台”，而不是一张单纯的分数排行榜。

## 10. X 权威账号库与新闻性过滤

X 是当前系统中最敏感也最容易失控的一层，因此这里采取了比其他来源更严格的质量约束。

### 10.1 权威账号库

当前系统维护一个动态 `authority registry`，其种子来自：

- 手工整理的官方账号、公司账号和研究人员账号；
- `follow-the-ai-leaders`；
- `PaperPulse` 的 researcher authors。

运行时，这些种子会被刷新为缓存化的权威账号库，再作为 direct X 抓取的唯一账号集合。这样，系统就从：

> 先抓一个很大的 AI 推文面，再靠文本过滤

转变为：

> 先限定账号质量，再抓取内容

这是 X 层质量提升的关键原因。

### 10.2 新闻性过滤

即使账号本身权威，也不代表它发布的所有内容都值得进入热点系统。当前实现因此会显式过滤以下内容：

- 招聘、报名、活动安排；
- webinar、workshop、conference chatter；
- 泛聊天、情绪表达、无信息量对话；
- 单篇 work/paper 的自我宣传；
- 低信息量、低实质性的 AI 邻近帖子。

对官方/公司账号，系统要求同时满足：

- 文本确实与 AI 相关；
- 并且具有明确新闻模式，或者具有足够强的产品/平台/研究实质性。

对研究人员账号，要求更严格：

- 禁止 self-paper announce；
- 必须带外链证据；
- 低 activity 帖子直接过滤；
- 文本要更像评论、评测、政策、安全、发布解读或研究判断。

因此，X 层现在优化的不是“数量”，而是“有新闻意义的权威信号”。

## 11. Daily Topics 的语义

`Daily Topics` 不应该被理解为“页面里所有 tag 的集合”。在当前算法中，它表示的是跨源一致出现的主题摘要。

具体来说，当前实现只保留：

- 至少由两个来源支持的 topic。

这意味着 `Daily Topics` 是对“当天到底围绕哪些东西形成了热点”的抽象，而不是一个由前端展示反推出的标签索引。

## 12. Web Data 与展示层

当前系统输出的是结构化 web data，而不是把 markdown 当作最终展示结构。

每日 payload 至少包含：

- `meta`
- `totals`
- `costs`
- `source_stats`
- `source_sections`
- `topic_summary`
- `featured_topics`
- `category_sections`
- `long_tail_sections`
- `watchlist`
- `x_buzz`

随后前端以 source-first 的紧凑表格视图来渲染这些数据。这里的职责划分是刻意为之：

- Python 决定什么值得保留、如何聚合、如何排序；
- 前端负责把这些结果以高密度、低废话的方式展示出来。

## 13. 构建与可复现性

整个热点系统是静态可构建的。当前的端到端流程为：

1. 采集并标准化原始信号；
2. 构建 cluster；
3. 计算确定性分数并进行候选分流；
4. 对边界案例做 LLM 复审；
5. 写出 report JSON 与 web-data JSON；
6. 构建前端静态资源；
7. 合并到最终静态站点并部署。

当前仓库仍然保持代码与数据分支分离：

- `main` 只放代码；
- `auto_update` 承载 `out/` 数据产物。

这使得历史回放、问题排查与前端重建都相对稳定。

## 14. 当前系统的经验性表现

结合最近多次本地运行，当前实现已经表现出几个明确特征：

- X 官方抓取在具备正确 X API 权限时已经可正常运行；
- 通过 authority registry 与 newsworthiness 过滤，低质量 X 杂讯已经显著下降；
- 在“安静日”上，系统会宁缺毋滥，不再为了凑数把弱单源主题硬塞进 featured；
- 在“热闹日”上，跨源官方/产品/研究主题能够正确进入主层；
- OpenAI 成本相比早期实现已有显著下降，主要得益于前置分流与动态 review 预算，而不是单纯换模型。

这说明当前质量提升的核心，并不是“让模型做更多事”，而是“让模型只做该做的事”。

## 15. 局限性

尽管当前版本已经明显优于早期实现，但仍有几个结构性局限。

### 15.1 X 覆盖仍受账号库约束

authority registry 提升了质量，但也意味着系统对新出现的研究者、团队或账号不够敏感。若不定期更新种子库，就可能漏掉新兴高质量来源。

### 15.2 跨源约束可能压制早期弱信号

要求多个来源共同支持有助于抑制噪声，但也会导致一些“刚刚出现、还未扩散”的早期重要信号被延后识别。

### 15.3 排序先验仍然部分依赖手工阈值

当前不少权重与阈值是可解释的启发式设计，而不是通过长期标注数据学习出来的。因此它们具备工程实用性，但仍有进一步量化评估与校准空间。

### 15.4 展示层与算法层故意解耦

这有利于维护，但也意味着视觉层的优化并不自动代表聚合质量的提升。真正的质量评估必须回到 report 层，而不能只看页面是否更美观。

## 16. 结论

当前 `Daily AI Hotspots` 更适合被理解为一个“面向前沿 AI 日信号的多源聚合与筛选系统”，而不是一个普通的 AI 新闻抓取器。它的主要贡献在于：

- 将异构来源统一映射到可比较的信号表示；
- 采用保守聚类来避免虚假主题合并；
- 将排序拆解为多个可解释维度；
- 通过置信度感知分流显著降低 API 成本；
- 通过权威账号库与新闻性过滤显著提升 X 层质量；
- 保持整个系统可静态部署、可历史重建。

在当前数据源和工程约束下，这套设计在质量、覆盖、成本之间取得了一个较强的平衡。

## 附录 A：实现映射

当前热点算法的核心实现分布在以下文件中：

- `arxiv_assistant/hotspots/pipeline.py`
- `arxiv_assistant/filters/filter_hotspots.py`
- `arxiv_assistant/utils/hotspot_cluster.py`
- `arxiv_assistant/utils/hotspot_web_data.py`
- `arxiv_assistant/utils/x_authority_registry.py`
- `arxiv_assistant/apis/hotspot/hotspot_x_common.py`
- `arxiv_assistant/apis/hotspot/hotspot_x_official.py`
- `arxiv_assistant/apis/hotspot/hotspot_x_ainews.py`
- `arxiv_assistant/apis/hotspot/hotspot_x_paperpulse.py`

## 附录 B：运行时配置

当前与热点系统最相关的运行时配置包括：

- `configs/config.ini`
- `configs/hotspot/roundup_sites.json`
- `configs/hotspot/x_authority_seeds.json`

本地调试可通过以下环境文件配置：

- `.env`
- `.env.local`

常用变量包括：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `X_BEARER_TOKEN`
- `GITHUB_TOKEN`

