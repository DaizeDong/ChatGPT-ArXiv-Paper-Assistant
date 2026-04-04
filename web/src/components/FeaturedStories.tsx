import { useI18n, tCategory } from "../lib/i18n";
import type { CompactTopic } from "../types/hotspot";

const CATEGORY_COLORS: Record<string, string> = {
  "Product Release": "var(--cat-release)",
  "Market Signal": "var(--cat-market)",
  Research: "var(--cat-research)",
  Tooling: "var(--cat-tooling)",
  "Industry Update": "var(--cat-industry)",
};

function categoryColor(category: string) {
  return CATEGORY_COLORS[category] ?? "var(--muted)";
}

function StoryCard({ topic }: { topic: CompactTopic }) {
  const { lang, t, tz } = useI18n();
  const color = categoryColor(topic.category);
  const any = topic as Record<string, unknown>;
  return (
    <div className="story-card">
      <div className="story-card-head">
        <span className="story-category-badge" style={{ borderColor: color, color }}>
          {tCategory(topic.category, lang)}
        </span>
        <span className="story-score">{topic.scores.final.toFixed(1)}</span>
      </div>
      <h3 className="story-headline">{tz(topic.headline, any.headline_zh as string)}</h3>
      {topic.summary_short ? <p className="story-summary story-summary-full">{tz(topic.summary_short, any.summary_short_zh as string)}</p> : null}
      {topic.key_takeaways?.length ? (
        <ul className="story-takeaways">
          {((any.key_takeaways_zh as string[] | undefined) && lang === "zh"
            ? (any.key_takeaways_zh as string[])
            : topic.key_takeaways
          ).map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      ) : null}
      <div className="story-evidence">
        {topic.evidence.map((ev, i) => (
          <a key={i} className="story-evidence-link" href={ev.url} target="_blank" rel="noreferrer">
            <span className="story-evidence-source">{ev.source_name}</span>
            <span className="story-evidence-title">{tz(ev.title, (ev as Record<string, unknown>).title_zh as string)}</span>
          </a>
        ))}
      </div>
      <div className="story-card-footer">
        <span className="story-sources">{topic.source_names.length} {t("label.sources")}</span>
        {topic.source_tier !== "community" ? <span className="story-tier">{topic.source_tier.replace("_", " ")}</span> : null}
      </div>
    </div>
  );
}

export function FeaturedStories({ topics, searchQuery }: { topics: CompactTopic[]; searchQuery: string }) {
  const { t } = useI18n();
  const filtered = topics.filter((tp) => {
    if (!searchQuery) return true;
    const haystack = [tp.headline, tp.summary_short, tp.why_it_matters, tp.category, ...tp.source_names, ...tp.evidence.map((e) => e.title)].join(" ").toLowerCase();
    return haystack.includes(searchQuery);
  });

  if (!filtered.length) return null;

  return (
    <section className="panel compact-panel">
      <div className="section-header table-header">
        <h2>{t("section.featured")} ({filtered.length})</h2>
      </div>
      <div className="story-grid">
        {filtered.map((topic) => (
          <StoryCard key={topic.topic_id} topic={topic} />
        ))}
      </div>
    </section>
  );
}
