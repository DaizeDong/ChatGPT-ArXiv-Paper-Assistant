import { useI18n, tCategory } from "../lib/i18n";
import type { CompactTopic } from "../types/hotspot";

type CategorySection = {
  category: string;
  total_candidates: number;
  topics: CompactTopic[];
};

function matchesTopic(topic: CompactTopic, query: string) {
  if (!query) return true;
  const haystack = [topic.headline, topic.summary_short, topic.why_it_matters, topic.category, ...topic.source_names, ...topic.evidence.map((e) => e.title)].join(" ").toLowerCase();
  return haystack.includes(query);
}

function TopicItem({ topic }: { topic: CompactTopic }) {
  const { lang, t, tz } = useI18n();
  const any = topic as Record<string, unknown>;
  const firstUrl = topic.evidence.length > 0 ? topic.evidence[0].url : undefined;
  const headline = tz(topic.headline, any.headline_zh as string);
  return (
    <div className="category-topic-item">
      <div className="category-topic-head">
        <div className="category-topic-headline-row">
          {firstUrl ? (
            <a className="category-topic-headline" href={firstUrl} target="_blank" rel="noreferrer">
              {headline}
            </a>
          ) : (
            <span className="category-topic-headline">{headline}</span>
          )}
          <span className="category-topic-score">{topic.scores.final.toFixed(1)}</span>
          <span className="category-topic-sources">{topic.source_names.length} {t("label.sources")}</span>
        </div>
      </div>
      {topic.summary_short ? <p className="category-topic-summary">{tz(topic.summary_short, any.summary_short_zh as string)}</p> : null}
      {topic.why_it_matters ? (
        <p className="category-topic-why">
          <strong>{t("label.why")}</strong> {tz(topic.why_it_matters, any.why_it_matters_zh as string)}
        </p>
      ) : null}
      {topic.evidence.length > 0 ? (
        <div className="category-topic-evidence">
          {topic.evidence.map((ev, i) => (
            <a key={i} className="category-topic-evidence-link" href={ev.url} target="_blank" rel="noreferrer">
              <span className="category-topic-evidence-source">{ev.source_name}</span>
              <span className="category-topic-evidence-title">{tz(ev.title, (ev as Record<string, unknown>).title_zh as string)}</span>
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Merge sections sharing the same category into one entry per category. */
function mergeSections(sections: CategorySection[]): CategorySection[] {
  const map = new Map<string, CategorySection>();
  for (const sec of sections) {
    const existing = map.get(sec.category);
    if (existing) {
      const seen = new Set(existing.topics.map((tp) => tp.topic_id));
      for (const tp of sec.topics) {
        if (!seen.has(tp.topic_id)) {
          existing.topics.push(tp);
          seen.add(tp.topic_id);
        }
      }
      existing.total_candidates += sec.total_candidates;
    } else {
      map.set(sec.category, { ...sec, topics: [...sec.topics] });
    }
  }
  return Array.from(map.values());
}

export function CategoryRadar({
  title,
  sections,
  searchQuery,
}: {
  title: string;
  sections: CategorySection[];
  searchQuery: string;
}) {
  const { lang } = useI18n();
  const merged = mergeSections(sections);
  const filteredSections = merged
    .map((section) => ({
      ...section,
      topics: section.topics.filter((tp) => matchesTopic(tp, searchQuery)),
    }))
    .filter((section) => section.topics.length > 0);

  const totalTopics = filteredSections.reduce((sum, s) => sum + s.topics.length, 0);

  if (!totalTopics) return null;

  return (
    <section className="panel compact-panel">
      <div className="section-header table-header">
        <h2>{title} ({totalTopics})</h2>
      </div>
      <div className="category-sections">
        {filteredSections.map((section) => (
          <div className="category-section" key={section.category}>
            <div className="category-section-header">
              <span className="category-section-name">{tCategory(section.category, lang)}</span>
              <span className="category-section-count">{section.topics.length} / {section.total_candidates}</span>
            </div>
            <div className="category-topic-list">
              {section.topics.map((topic) => (
                <TopicItem key={topic.topic_id} topic={topic} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
