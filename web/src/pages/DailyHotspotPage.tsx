import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadDailyHotspot } from "../lib/data";
import type { DailyHotspotPayload, SourceSection, SourceSectionItem } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

type DensityMode = "compact" | "comfortable";

function compactNumber(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
  }
  return `${value}`;
}

function itemSignals(item: SourceSectionItem) {
  const signals: string[] = [];
  if (item.signals.activity > 0) {
    signals.push(`activity ${compactNumber(item.signals.activity)}`);
  }
  if (item.signals.github_stars > 0) {
    signals.push(`stars ${compactNumber(item.signals.github_stars)}`);
  }
  if (item.signals.hn_score > 0) {
    signals.push(`HN ${compactNumber(item.signals.hn_score)}`);
  }
  if (item.signals.upvotes > 0) {
    signals.push(`upvotes ${compactNumber(item.signals.upvotes)}`);
  }
  if (item.signals.daily_score > 0) {
    signals.push(`daily ${compactNumber(item.signals.daily_score)}`);
  }
  return signals.slice(0, 3);
}

function matchesSearch(item: SourceSectionItem, query: string) {
  if (!query) {
    return true;
  }
  const haystack = [
    item.title,
    item.summary_short,
    item.source_name,
    item.source_role,
    item.source_type,
    item.tags.join(" "),
    item.topic_refs.map((topic) => topic.slug).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

function filteredSections(sections: SourceSection[], searchQuery: string) {
  const query = searchQuery.trim().toLowerCase();
  return sections
    .map((section) => {
      const items = section.items.filter((item) => matchesSearch(item, query));
      return { ...section, filteredItems: items };
    })
    .filter((section) => section.filteredItems.length > 0);
}

function defaultVisibleCount(section: SourceSection, density: DensityMode) {
  const boosted = section.slug === "x-buzz" || section.slug === "blogs" ? 12 : 10;
  return density === "compact" ? boosted : Math.max(8, boosted - 2);
}

function topicSummaryLimit(density: DensityMode) {
  return density === "compact" ? 18 : 12;
}

function sectionTitle(section: SourceSection) {
  return `${section.label} (${section.count})`;
}

function SignalRow({
  date,
  density,
  item,
}: {
  date: string;
  density: DensityMode;
  item: SourceSectionItem;
}) {
  const signals = itemSignals(item);
  const topicRefs = item.topic_refs.slice(0, density === "compact" ? 2 : 3);

  return (
    <article className={`signal-row ${density}`}>
      <div className="signal-main">
        <a className="signal-title" href={item.url} target="_blank" rel="noreferrer">
          {item.title}
        </a>
        <div className="signal-meta">
          <span>{item.source_name}</span>
          <span>{item.source_role.replaceAll("_", " ")}</span>
          <span>score {item.signal_score.toFixed(1)}</span>
          {signals.map((signal) => (
            <span key={signal}>{signal}</span>
          ))}
        </div>
        {density === "comfortable" ? <p className="signal-summary">{item.summary_short}</p> : null}
      </div>
      {topicRefs.length > 0 ? (
        <div className="signal-topics">
          {topicRefs.map((topic) => (
            <Link key={topic.topic_id} className="signal-topic-pill" to={`/hot/${date}/topic/${topic.slug}`}>
              {topic.headline}
            </Link>
          ))}
        </div>
      ) : null}
    </article>
  );
}

export function DailyHotspotPage({ date }: { date: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [density, setDensity] = useState<DensityMode>("compact");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedCounts, setExpandedCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    let active = true;
    loadDailyHotspot(date)
      .then((payload) => {
        if (active) {
          setState({ status: "ready", payload });
          setExpandedCounts({});
        }
      })
      .catch((error: Error) => {
        if (active) {
          setState({ status: "error", message: error.message });
        }
      });
    return () => {
      active = false;
    };
  }, [date]);

  if (state.status === "loading") {
    return <section className="panel">Loading daily hotspot payload...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load {date}: {state.message}</section>;
  }

  const { payload } = state;
  const visibleSections = filteredSections(payload.source_sections, searchQuery);
  const visibleTopicSummary = payload.topic_summary.slice(0, topicSummaryLimit(density));

  return (
    <div className="stack">
      <section className="panel day-hero">
        <div className="day-nav">
          <div className="day-nav-edge">
            {payload.meta.previous_date ? <Link to={`/hot/${payload.meta.previous_date}`}>Previous day</Link> : <span>Previous day</span>}
          </div>
          <Link to="/hot" className="day-nav-center">
            Hotspot index
          </Link>
          <div className="day-nav-edge right">
            {payload.meta.next_date ? <Link to={`/hot/${payload.meta.next_date}`}>Next day</Link> : <span>Next day</span>}
          </div>
        </div>

        <div className="hero-grid">
          <div>
            <p className="eyebrow">{payload.meta.mode} generation</p>
            <h1>Daily AI Hotspots {payload.meta.date}</h1>
            <p className="lede">{payload.meta.summary}</p>
          </div>
          <div className="hero-actions">
            <Link className="primary-link" to={`/hot/${payload.meta.month}`}>
              Month archive
            </Link>
            <Link className="inline-link" to={`/hot/${payload.meta.year}`}>
              Year archive
            </Link>
          </div>
        </div>

        <div className="summary-band">
          <div>
            <p className="stat-label">Featured</p>
            <p className="stat-value">{payload.meta.counts.featured_topics}</p>
          </div>
          <div>
            <p className="stat-label">Source items</p>
            <p className="stat-value">{payload.meta.counts.source_items}</p>
          </div>
          <div>
            <p className="stat-label">Topic strip</p>
            <p className="stat-value">{payload.meta.counts.topic_summary}</p>
          </div>
          <div>
            <p className="stat-label">Long tail</p>
            <p className="stat-value">{payload.meta.counts.long_tail}</p>
          </div>
        </div>

        <div className="section-chip-row">
          {visibleSections.map((section) => (
            <a key={section.slug} className="section-chip" href={`#section-${section.slug}`}>
              <span>{section.label}</span>
              <strong>{section.filteredItems.length}</strong>
            </a>
          ))}
        </div>
      </section>

      <section className="panel controls-panel">
        <div className="controls-row">
          <label className="search-field">
            <span>Search today's signals</span>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="model releases, agents, Claude, OCR..."
            />
          </label>
          <div className="toggle-group" role="tablist" aria-label="Display density">
            <button
              className={density === "compact" ? "active" : ""}
              onClick={() => setDensity("compact")}
              type="button"
            >
              Compact
            </button>
            <button
              className={density === "comfortable" ? "active" : ""}
              onClick={() => setDensity("comfortable")}
              type="button"
            >
              Comfortable
            </button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>Today at a glance</h2>
            <p>Short featured topics for the strongest same-day stories before the denser source scan below.</p>
          </div>
        </div>
        <div className="featured-grid">
          {payload.featured_topics.map((topic) => (
            <Link className="featured-card" key={topic.topic_id} to={`/hot/${payload.meta.date}/topic/${topic.slug}`}>
              <span className="card-kicker">{topic.category}</span>
              <strong>{topic.headline}</strong>
              <p>{topic.summary_short || topic.why_it_matters}</p>
              <div className="card-metrics">
                <span>{topic.source_names.length} sources</span>
                <span>final {topic.scores.final.toFixed(1)}</span>
                <span>heat {topic.scores.heat.toFixed(1)}</span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>Topic summary strip</h2>
            <p>Clustered navigation for repeated themes, kept secondary so the page still prioritizes broad source coverage.</p>
          </div>
        </div>
        <div className="topic-strip">
          {visibleTopicSummary.map((topic) => (
            <Link className="topic-pill" key={topic.topic_id} to={`/hot/${payload.meta.date}/topic/${topic.slug}`}>
              <span>{topic.headline}</span>
              <small>
                {topic.source_count} sources | {topic.category}
              </small>
            </Link>
          ))}
        </div>
      </section>

      {visibleSections.map((section) => {
        const initial = defaultVisibleCount(section, density);
        const expanded = expandedCounts[section.slug] ?? initial;
        const visibleItems = section.filteredItems.slice(0, expanded);
        const remaining = Math.max(section.filteredItems.length - visibleItems.length, 0);

        return (
          <section className="panel source-family-panel" id={`section-${section.slug}`} key={section.slug}>
            <div className="section-header">
              <div>
                <h2>{sectionTitle(section)}</h2>
                <p>{section.description}</p>
              </div>
              <div className="section-header-actions">
                <Link className="inline-link" to={`/hot/${payload.meta.date}/source/${section.slug}`}>
                  Open detail page
                </Link>
              </div>
            </div>

            <div className="signal-list">
              {visibleItems.map((item) => (
                <SignalRow date={payload.meta.date} density={density} item={item} key={item.id} />
              ))}
            </div>

            {remaining > 0 ? (
              <button
                className="show-more-button"
                onClick={() =>
                  setExpandedCounts((current) => ({
                    ...current,
                    [section.slug]: expanded + defaultVisibleCount(section, density),
                  }))
                }
                type="button"
              >
                Show {Math.min(defaultVisibleCount(section, density), remaining)} more from {section.label}
              </button>
            ) : null}
          </section>
        );
      })}

      {!visibleSections.length ? (
        <section className="panel">
          <h2>No matching signals</h2>
          <p>Try a broader search query or switch back to compact mode for denser scanning.</p>
        </section>
      ) : null}
    </div>
  );
}
