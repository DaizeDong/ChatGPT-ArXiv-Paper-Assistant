import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { SignalRow } from "../components/SignalRow";
import { loadDailyHotspot } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
import { defaultVisibleCount, filterSectionsBySearch } from "../lib/hotspotView";
import type { DailyHotspotPayload, SourceSection } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

function topicSummaryLimit() {
  return 8;
}

function sectionTitle(section: SourceSection) {
  return `${section.label} (${section.count})`;
}

export function DailyHotspotPage({ date }: { date: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });
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
  const visibleSections = filterSectionsBySearch(payload.source_sections, searchQuery);
  const crossSourceTopics = payload.topic_summary.filter((topic) => topic.source_count > 1);
  const visibleTopicSummary = (crossSourceTopics.length ? crossSourceTopics : payload.topic_summary).slice(0, topicSummaryLimit());
  const paperDayRoute = bestPaperRoute(payload.meta.paper_routes, ["day", "month", "year", "home"]);
  const paperArchiveRoute = bestPaperRoute(payload.meta.paper_routes, ["month", "year", "home"]);

  return (
    <div className="stack">
      <section className="panel day-hero dense-day-hero">
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

        <div className="hero-grid dense-hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Daily feed</p>
            <h1>Daily AI Hotspots {payload.meta.date}</h1>
            <div className="jump-row">
              <a className="inline-link" href={paperDayRoute}>Paper digest</a>
              <a className="inline-link" href={paperArchiveRoute}>Paper archive</a>
              <Link className="inline-link" to={`/hot/${payload.meta.month}`}>Hot month</Link>
              <Link className="inline-link" to={`/hot/${payload.meta.year}`}>Hot year</Link>
            </div>
            <div className="summary-band compact-summary-band">
              <span className="summary-chip">items {payload.meta.counts.source_items}</span>
              <span className="summary-chip">featured {payload.meta.counts.featured_topics}</span>
              <span className="summary-chip">source groups {visibleSections.length}</span>
              <span className="summary-chip">cross-source {crossSourceTopics.length}</span>
            </div>
          </div>
          <div className="hero-tools">
            <label className="search-field compact-search">
              <span>Search</span>
              <input
                type="search"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search titles, sources, tags..."
              />
            </label>
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

      <section className="panel compact-panel">
        <div className="section-header">
          <div>
            <h2>Cross-source topics</h2>
          </div>
        </div>
        <div className="topic-strip compact-topic-strip">
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
        const initial = defaultVisibleCount(section, "compact");
        const expanded = expandedCounts[section.slug] ?? initial;
        const visibleItems = section.filteredItems.slice(0, expanded);
        const remaining = Math.max(section.filteredItems.length - visibleItems.length, 0);

        return (
          <section className="panel source-family-panel" id={`section-${section.slug}`} key={section.slug}>
            <div className="section-header">
              <div>
                <h2>{sectionTitle(section)}</h2>
              </div>
              <div className="section-header-actions">
                <Link className="inline-link" to={`/hot/${payload.meta.date}/source/${section.slug}`}>
                  Open detail page
                </Link>
              </div>
            </div>

            <div className="signal-list">
              {visibleItems.map((item) => (
                <SignalRow date={payload.meta.date} density="compact" item={item} key={item.id} />
              ))}
            </div>

            {remaining > 0 ? (
              <button
                className="show-more-button"
                onClick={() =>
                  setExpandedCounts((current) => ({
                    ...current,
                    [section.slug]: expanded + defaultVisibleCount(section, "compact"),
                  }))
                }
                type="button"
              >
                Show {Math.min(defaultVisibleCount(section, "compact"), remaining)} more from {section.label}
              </button>
            ) : null}
          </section>
        );
      })}

      {!visibleSections.length ? (
        <section className="panel">
          <h2>No matching signals</h2>
          <p>Try a broader search query.</p>
        </section>
      ) : null}
    </div>
  );
}
