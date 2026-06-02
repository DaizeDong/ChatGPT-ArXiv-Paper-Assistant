# 竞品全景调研：每日 AI 论文 + 热点聚合赛道

> **快照日期**：2026-06-02（所有 star/粉丝/价格均以此日为准，标注 `[fetched 2026-06-02]`）
> **调研方法**：market-intel skill（deep 档）→ 8 个并行 subagent 真实抓取 GitHub REST API、官网/Pricing 页、twitterapi.io 实时数据、技术博客与论文 → 结构化证据单元 → 交叉核验。
> **对比基准（我们）**：`ChatGPT-ArXiv-Paper-Assistant`（DaizeDong fork of tatsu-lab/gpt_paper_assistant）。双管线：① 个性化 arXiv 论文筛选（作者规则 + LLM 标题/摘要打分）② 每日 AI 热点聚合（28+ 源、归一化 → 链接去重 → 聚类成话题 → 确定性打分 → confidence-aware 路由 → LLM 复核 → 中英双语静态站点，日/月/年归档）。
> **来源分级**：L1 一手/官方 · L2 独立第三方 · L3 利益相关方 · L4 UGC/匿名 · L5 兜底推断。

---

## 1. 执行摘要：我们到底落后在哪、领先在哪

你的直觉"很多都比我们强很多"——**在单一维度上成立，在整体组合上不成立**。诚实结论分三层：

**A. 我们明确落后（且差距巨大）的维度**
- **分发/触达**：这是最硬的差距。我们零主动分发，只产静态站。对比：The Rundown AI ~200 万邮件订阅、AK(@_akhaliq) 50 万 X 粉、TLDR AI 50万–125万、Scholar Inbox/HF Daily Papers 数千–上万订阅。我们 7★ vs 头部仓库 2,800–5,400★。
- **社区/网络效应**：HF Daily Papers 的投票、alphaXiv 的逐行讨论带来自增长；我们没有任何社交层。
- **个性化模型**：Scholar Inbox（学习型推荐，ACL 2025 demo）、zotero-arxiv-daily（Zotero 嵌入个性化）、Semantic Scholar Research Feeds 都有"按用户学习"的推荐；我们只有静态 prompt + 阈值。
- **深度能力**：Elicit/SciSpace（系统综述抽取）、Consensus（证据合成）、Undermind（agentic 深检索）、scite（引用核查）在"读懂一篇/一组论文"上远超我们——但这是**互补赛道，不是同一件事**。

**B. 我们出乎意料地领先 / 几乎无人重合的维度**
- **没有任何一个竞品同时具备这 5 件事**：① 开源 + 自托管（无锁定/无配额）② 中英**并排**多语言静态站 ③ arXiv 个性化筛选 ④ 多源 AI 热点聚合（含 X/社交）⑤ 语义聚类 + 多时间维度（日/月/年）归档。
- 最接近的 `agents-radar`(785★) 和 `Horizon`(5,392★) 也各缺一块：agents-radar **无语义聚类**（按源分栏摘要）、Horizon 是**通用新闻雷达**（论文是顺带、去重是 URL 级而非话题聚类、无周/月归档）。
- 唯一"真正规模化自动"的英文对标 smol-ai/AINews 核心 pipeline **闭源 + 纯英文 + 依赖爬社区平台**（Discord 已于 2026-03 被切断，暴露脆弱性——而我们用 API/托管源更稳）。

**C. 一句话定位**
> 我们不是"做得比别人差的论文 bot"，而是**整个赛道里唯一把"开源自托管 + 双语 + arXiv 筛选 + 多源热点 + 语义聚类 + 多时间归档"打包成一个零后端方案的工具**。真正的短板是**最后一公里的分发与人格化包装**，而非产品能力本身。

---

## 2. 竞品全景（按赛道分六类）

### 2.1 类目 A — arXiv LLM 筛选/推荐 开源仓库（我们的直接赛道）

| 仓库 | ★ `[2026-06-02]` | 活跃 | 机制 | 输出 | 多语言 | 热点 | 相对我们 |
|---|---|---|---|---|---|---|---|
| **TideDra/zotero-arxiv-daily** | **5,432** | 很活跃 | Zotero 库嵌入相似度 + LLM TL;DR | 邮件 | TL;DR 可配 | ✗ | 个性化最强；但单源、邮件、无站点/聚类/热点 |
| **dw-dengwei/daily-arXiv-ai-enhanced** | **2,805** | 很活跃 | 关键词/作者 + LLM 摘要(DeepSeek) + 兴趣高亮 | GitHub Pages | 默认中文 | ✗ | 最接近的开源对标；但单源 arXiv、无热点、单语言 |
| Vincentqyw/cv-arxiv-daily | 1,472 | 活跃 | Actions 关键词爬取 + PwC 链接 | Markdown 列表 | ✗ | ✗ | 模板级、无 LLM/聚类 |
| karpathy/arxiv-sanity-lite | 1,620 | 停更(2023) | SVM over tf-idf（无 LLM） | 自托管 Web | ✗ | ✗ | 架构祖宗；经典但已停 |
| ziwenhahaha/daily-paper-reader | 620 | 活跃 | 多源 + 嵌入召回 + Qwen rerank + LLM + 站内问答 | Pages 交互阅读站 | 中文为主 | ✗ | 检索栈最强；但 ID 去重非语义、无周月归档 |
| AutoLLM/ArxivDigest | 425 | 停更(2024) | LLM(1-10 打分) | 邮件 + Gradio | ✗ | ✗ | 早期名作、已停 |
| yuandong-tian/arXiv_recbot | 307 | 活跃 | 个性化推荐 | Telegram | ✗ | ✗ | Telegram 分发 |
| tatsu-lab/gpt_paper_assistant（母仓库） | 549 | **停更(2024-03)** | 作者白名单 + LLM 打分 | Markdown/Slack | ✗ | ✗ | 我们的起点；原作已弃管 |
| Xuchen-Li/llm-arxiv-daily | 143 | 很活跃 | 关键词/分类 Actions | README | ✗ | ✗ | 零基建但仅关键词 |
| freemty/no-more-fomo | 6 | 活跃 | 16 X KOL+lab+播客+arXiv+HF+HN，去重+分类 | 本地 HTML | 含中文 | ✓ | 概念最像热点；但极早期、CLI |

> **母仓库 fork 网络**：tatsu-lab/gpt_paper_assistant 的 fork 几乎全是低星休眠克隆（最高非我们：stanford-oval 12★）。**我们 7★、2026-06-02 仍在 push，是这一血脉中维护最活跃的增强 fork。**

### 2.2 类目 B — 学术论文发现平台（SaaS/网站）

| 平台 | 定位 | 机制 | 价格 `[2026-06-02]` | 相对我们 |
|---|---|---|---|---|
| **HF Daily Papers** | 论文社交中心（PwC 后继） | 社区投票 + 作者自提 + 代码链接 | 免费 | 网络效应碾压；但靠热度、非个性化、纯英文 |
| **alphaXiv** | arXiv 之上的讨论/AI 分析层 | 逐行评论 + AI Ask + 语义检索 | 免费（$7M 种子轮 Menlo，2025-11） | 资金+势头强；偏单篇深读非日推 |
| **Papers with Code** | 曾经的 代码/SOTA 榜单霸主 | benchmark/leaderboard | **已关停**（Meta，2025-07-24/25） | ⚠ **市场空缺**：HF 替代品缺榜单深度 |
| **Scholar Inbox** | 免费个性化论文推荐器 | **学习型推荐模型**(用户评分) + 邮件 | 免费（ACL 2025 demo） | **最直接竞品**；强在个性化，但闭源 SaaS |
| Cool Papers (papers.cool, 苏剑林) | 沉浸式刷论文 | 分类聚合 + Kimi 中文摘要 + RSS | 免费、开源 | UX 最像；无个性化/热点（与我们一样靠浏览） |
| Emergent Mind | arXiv 研究助手 | 社媒热度排序 + 多 LLM + 讲解视频 | 免费10篇/周·Pro $12/mo | 多 LLM+讲解；有配额、闭源 |
| Connected Papers | 可视化文献图谱 | 共被引相似度图 | 免费5图/月·学术$5/mo | 探索工具非日推 |
| Zeta Alpha | 企业级神经发现 | 语义检索 + 推荐 | €20/座/月起 | 企业向、过重 |
| Semantic Scholar | 免费学术搜索引擎(AI2) | Research Feeds 自适应推荐 + TLDR | 免费（开放 API） | 免费且像我们；但是站内 feed、无热点层。**其 API 很可能是我们这类工具的上游数据源** |
| 42papers | 趋势论文 + 个性化 feed | 不透明 | unknown（主页拒连） | 低调、机制不明 |

### 2.3 类目 C — 英文 AI 新闻/热点日报（对标热点管线）

| 产品 | 人工/自动 | 规模 `[2026-06-02]` | 开源 | 相对我们 |
|---|---|---|---|---|
| **smol-ai / AINews** | **自动**(99% agent) | ~8 万订阅（站点仍标 150k，冲突见 §5） | **核心闭源**（仅前端开源） | **最像我们的自动化对标**；但纯英文、爬社区平台脆弱、跳过 arXiv/blog/GitHub 正式源 |
| The Rundown AI | 人工编辑 | ~200 万订阅 | ✗ | 触达冠军；但全人工、消费向、纯英文 |
| TLDR AI | 人工编辑 | 50万–125万（冲突） | ✗ | 规模标杆；无聚类/去重技术 |
| Import AI (Jack Clark) | 人工单作者 | ~12.9 万 | ✗ | 深度权威；周更、不可规模化 |
| Ben's Bites | 人工 | ~11.5 万 | ✗ | 已降频(2→周)，创始人转 VC，相关性下滑 |
| The Batch (吴恩达) | 人工编辑 | 母体 700万学习者 | ✗ | 权威质量标杆；周更、非速度竞品 |
| Last Week in AI | 人工 | unknown | ✗ | 周更、爱好者规模 |
| **finaldie/auto-news** | 自动 | 883★（**2024-07 停更**） | ✓ MIT | 真·多源+LLM 降噪；但自托管、停更、无聚类 |
| **Thysrael/Horizon** | 自动 | **5,392★**（很活跃） | ✓ | 见类目 E，跨类目最强通用新闻雷达 |

### 2.4 类目 D — 中文 AI 资讯/论文平台

| 平台 | 类型 | 生产方式 | 开源 | 相对我们 |
|---|---|---|---|---|
| 机器之心 | 媒体+工具 | 编辑 + ArXiv Weekly Radiostation(人工精选) + SOTA!模型站 | ✗ | 内容标杆非工具对标；人工筛选、纯中文 |
| 量子位 | 媒体 | 纯编辑、日更多篇 | ✗ | 不做论文筛选 |
| 新智元 | 媒体 | 编辑、研究导向 | ✗ | ~350万用户、决策层为主；不做工具 |
| PaperWeekly | 媒体+学术社区 | 众包(100+志愿者)人工筛选 | ✗ | 学术深度好；依赖人工、规模受限 |
| **AMiner**（智谱/清华） | 工具 | 知识图谱 + GLM + 个性化推荐 + 邮件订阅 | ✗ | 最强中文工具对标；偏严肃检索、闭源 |
| Cool Papers | 工具 | 见类目 B | partial | UX 强、无个性化/热点 |

> **结论**：中文媒体本质是编辑/众包内容平台（非算法自动化、不开源、纯中文、无双语并排），是**内容质量标杆而非产品对标**。中文赛道里真正的工具对标只有 AMiner（闭源商业）和开源的 daily-arXiv-ai-enhanced。

### 2.5 类目 E — GitHub 开源多源聚合器（最同质竞品层）

| 仓库 | ★ `[2026-06-02]` | 多源 | 去重/聚类 | 双语 | 多时间归档 | 输出渠道 | 相对我们 |
|---|---|---|---|---|---|---|---|
| **Thysrael/Horizon** | **5,392** | ✓ 7类源 | URL/story 去重（非语义） | ✓ EN/ZH | ✗ | Pages+邮件+Webhook+MCP | **最强通用新闻雷达**；论文顺带、去重 URL 级、无周月归档 |
| **duanyytop/agents-radar** | **785** | ✓ 10源 | **无语义聚类**(按源分栏) | ✓ ZH/EN | ✓ 日/周/月 | Pages+Issues+TG+飞书+RSS+MCP | **最直接对标、广度最全**；缺语义聚类（=我们的差异化） |
| SuYxh/ai-news-aggregator | 257 | ✓ 156+源 | 源级去重、无聚类 | 标题翻译 | 45天归档、2h更新 | Pages | 源最多；但无 LLM 摘要/聚类 |
| ziwenhahaha/daily-paper-reader | 620 | ✓ 论文多源 | ID 去重 + rerank | 中文为主 | ✗ | 交互阅读站 | 论文向多源最强；非语义聚类 |
| anuj0456/AiLert | 28 | ✓ 150+源 | 无 | ✗ | 日/周 | 邮件 | 广覆盖；需 AWS、无 LLM |
| gabrielchua/daily-ai-papers | 218 | ✗(HF镜像) | ✗ | ✗ | ✗ | TG 音频摘要 | 新颖音频输出 |

### 2.6 类目 F — X/Twitter 论文/热点账号（既是信息源也是分发竞品，twitterapi.io 实拉）

| 账号 | 粉丝 `[2026-06-02]` | 形态 | 与聚合关系 |
|---|---|---|---|
| **@_akhaliq (AK)** | **502,465** | 半自动(模板+人工RT) | HF Daily Papers 的人格化前端，逐条推论文给 50万人 |
| @TheRundownAI | 221,258 | 人工编辑、定时班次 | newsletter↔X 双渠道规模化模板 |
| @rohanpaul_ai | 149,406 | 人工长文解读 | 论文→通俗解读+商业语境（增值层） |
| @dair_ai (Elvis) | 124,965 | 人工策展 + 每周 Top Papers 榜 | 与我们最像的"周榜"模式，导流自家 AI Academy |
| @tldrnewsletter | 122,238 | **X 已休眠**(2025-12起) | 大号也会弃 X 回归邮件 |
| @Arxiv_Daily (DeepAI) | 49,854 | **已失活** bot | 纯 bot 无维护会衰减（反面教材） |
| @HuggingPapers | 18,196 | **纯 bot** 全自动 | 与我们产品形态最直接对标 |

> **关键数字**：官方全自动 @HuggingPapers 仅 1.8万粉，人格化的 AK 有 50万粉——**27 倍差距**。证明纯 bot 转推触达极低，护城河在"选品 + 解读 + 人格信任"，不是"有没有 bot"。

---

## 3. 实际效果对比（量级一览）

| 维度 | 我们 | 头部竞品 | 差距 |
|---|---|---|---|
| GitHub Stars | 7 | zotero-arxiv-daily 5,432 / Horizon 5,392 / dw-dengwei 2,805 | ~400–780× |
| 订阅/触达 | 0 主动分发 | Rundown ~200万 / AINews ~8万 / HF Daily ~1.2万+ | 极大 |
| X 粉丝 | 无账号 | AK 50万 / Rundown 22万 | 极大 |
| 自动化程度 | **全自动 Actions** | AINews 全自动；其余英文头部多为人工 | **我们领先于多数** |
| 多源聚合 | ✓ 28+源 | agents-radar 10 / Horizon 7类 / SuYxh 156 | 中游偏上 |
| 语义聚类 | ✓ | 几乎无人做（多为 URL/源级去重） | **我们领先** |
| 双语并排 | ✓ 中英 | Horizon/agents-radar 双语、SuYxh 仅标题翻译 | 少数同级 |
| 多时间归档 | ✓ 日/月/年 | 仅 agents-radar 日/周/月、SuYxh 45天 | **我们领先** |
| 个性化模型 | ✗(静态prompt) | Scholar Inbox/zotero/S2 有学习型 | **我们落后** |

---

## 4. 交叉核验与冲突矩阵

| 声明 | 来源 A | 来源 B | 判定 |
|---|---|---|---|
| AINews 订阅数 | smol.ai 自报 ~8万 (2026) | 站点营销文案 150k+ | **disputed**：取较新自报 8万为准，150k 视为营销旧值 (L3) |
| TLDR AI 触达 | 部分源 50万 | 部分源 92万–125万(网络整体) | **unresolved**：AI 垂类精确值 unknown |
| alphaXiv 融资 | $7M 种子轮（Yahoo Finance, 2025-11-19, L2 ✓verified） | 某 Medium 称 ~$70M | **disputed**：$7M 已核实，$70M 无一手源，标 unknown |
| The Batch / Import AI "2025-06 上线" | 搜索返回 | — | 判为**搜索伪影**（两者均早于该日），未采纳 |
| Consensus Pro 价 | 二手聚合 $10–15/mo | 官网 403 无法核 | **unverified**：标 unknown/secondary |

所有 GitHub star（GitHub REST API 实拉）与 X 粉丝（twitterapi.io 实拉）均为 ✓verified，非估算。

---

## 5. 风险与反面证据（主动反向检索）

- **平台依赖风险（已被验证为真实）**：smol-ai/AINews 的 Discord 抓取于 ~2026-03-30 被切断且未恢复——印证"爬社区平台"的脆弱性。我们用 arXiv/blog/GitHub/HN API + 托管 X provider，结构上更稳，但 §6 显示我们的 X 实际产出≈0，等于优势未兑现。
- **渠道生命周期风险**：@Arxiv_Daily（4.9万粉）已失活、@tldrnewsletter 弃用 X。纯 bot 渠道无维护会衰减——若我们做 X 分发，必须叠加人格/解读层，否则重蹈覆辙。
- **赛道拥挤**：arXiv 日报这一窄赛道已有 5,000★+ 的成熟玩家（zotero-arxiv-daily），纯做"又一个 arXiv 日报"无差异化空间。我们的生路在**热点聚合 + 双语 + 聚类**的组合，不在单管线比拼。
- **执行摩擦**：我们 docs 已自查出三大真实缺陷（见 §6），在补分发之前应先修内功，否则放大分发只会放大缺陷。
- 反向检索未发现针对本赛道开源工具的"骗局/封禁"类风险——但这不证明无风险。

---

## 6. 可落地改进路线图（融合竞品最佳实践 + 我们 docs 自查痛点）

我们 `docs/` 已诚实记录三大痛点：**① 时效性 bug（HF 老论文绕过新鲜度门）② X 覆盖≈0（架构有 546 账号但日产 0–6 条）③ 跨天重复（相邻两天证据 URL 重叠 ~60%）**。竞品调研给出了对应的成熟解法：

| 优先级 | 改进 | 借鉴来源 | 难度 | 解决我们的 |
|---|---|---|---|---|
| **P0** | **持久化 story-centric 状态**（story ID + 质心嵌入存 sqlite/JSON，跨天匹配，标 NEW vs ONGOING） | TDT 经典范式 / 多日 story 追踪 | 高 | **跨天重复 + 时效性**（一次性治本） |
| **P0** | **HN 式 gravity 时间衰减**：`score=(P-1)/(T+2)^1.8`，T=自**发布**(非抓取)的小时数 | Hacker News 排名公式 | 中 | **时效性**（老论文必然下沉） |
| **P0** | 修 HF papers `fetched_at` 绕过 + 加 `is_fresh()`（已在 docs 设计） | 自查 | 低(1–5行) | **时效性** |
| **P1** | **语义去重层**：多语言句向量嵌入(title+lede)，日内 cosine>0.9 去重，跨语言（中英同一发布）合并为一话题 | 跨语言 dedup($100/mo 实践)、Matryoshka 多粒度嵌入 | 中 | **跨天/跨语言重复**（MinHash 仅作 exact 预筛） |
| **P1** | **Altmetric 式信号增强**：分级源权重(news=8…tweet=1/0.25)对齐量级 + **按作者去重提及**(抑制刷屏) + 引入 GitHub star **增速** & HF 投票作 evidence | Altmetric 公式 / HF trending | 低 | **热度打分质量 + X 覆盖质量** |
| **P1** | **map-reduce "Summary of Summaries"** + 分层模型（廉价模型做聚类/合并，强模型只做最终合成与模糊带复核——契合我们已有 confidence-aware 路由） | smol-ai/AINews 公开披露 | 中 | **摘要质量 + 成本** |
| **P2** | X 分发底座：用现有管线 + `_zh` 双语资产自动生成"每日/每周论文 X 线程"（类 @HuggingPapers），叠加固定栏目化(类 dair_ai 周榜)与轻解读 | AK / dair_ai / HuggingPapers | 中 | **分发短板**（最高 ROI 补强，因产出端已就绪） |
| **P2** | 先修 `is_newsworthy_x_text` 过滤器（官方账号豁免 SELF_WORK_PATTERNS）+ 改用 User Timeline 端点 | 自查 docs_x_coverage | 低–中 | **X 覆盖≈0** |

**战略次序建议**：先 P0/P1 修内功（时效+去重+打分），再 P2 补分发。理由见 §5——在缺陷未修前放大分发只会放大缺陷。

---

## 7. 覆盖缺口（诚实声明）

- **42papers** 主页拒绝程序化连接（ECONNREFUSED），机制/价格 unknown。
- **alphaXiv $70M** 融资说法无一手源，未采纳。
- **Consensus** 官网 Pricing 403，价格取二手、Pro 月费 $10–15 区间未定。
- **AINews 内部算法**仅部分公开（swyx 未开源完整 pipeline），细节来自 Buttondown 存档/Latent Space（权威但非 spec）。
- **The Batch / Last Week in AI** 未披露订阅数，标 unknown。
- 未单独覆盖：小语种(日/韩/欧陆)论文日报、Discord/Slack 内部机器人生态、企业内网知识雷达（多为私有）。

---

## 8. 完整来源清单（节选，均 `[fetched 2026-06-02]`）

- GitHub REST API（所有 star/last-push，L1 ✓verified）
- twitterapi.io MCP（所有 X 粉丝/发帖模式，L1 ✓verified）
- 官网/Pricing：elicit.com, undermind.ai, researchrabbit.ai, scholarcy.com, scispace.com, scite.ai, connectedpapers.com, scholar-inbox.com, emergentmind.com, zeta-alpha.com（L1，部分 403 已标）
- smol-ai/AINews：buttondown.com/ainews 存档 + Latent Space（L1/L2）
- Altmetric 权重：help.altmetric.com（L1）；HN 排名：righto.com（L2）
- 跨语言 dedup：yingjiezhao.com（L2）；TDT/FSD：Springer/arXiv（L1）
- 平台状态：Papers with Code 关停（HF CTO 公告，2025-07，L1）；alphaXiv $7M（Yahoo Finance，L2）
- 中文平台：jiqizhixin.com, qbitai.com, aiera.com.cn, paperweekly.site, papers.cool, aminer.cn（L1）
