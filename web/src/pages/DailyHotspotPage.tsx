import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { SignalTable } from "../components/SignalTable";
import { TopicSummaryTable } from "../components/TopicSummaryTable";
import { loadDailyHotspot } from "../lib/data";
import { filterSectionsBySearch } from "../lib/hotspotView";
import type { DailyHotspotPayload, SourceSection } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

function sectionTitle(section: SourceSection) {
  return `${section.label} (${section.count})`;
}

function shouldShowDailyTopic(topic: DailyHotspotPayload["topic_summary"][number]) {
  return topic.source_count > 1 || topic.llm_status === "featured";
}

export function DailyHotspotPage({ date }: { date: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [searchParams] = useSearchParams();

  useEffect(() => {
    let active = true;
    loadDailyHotspot(date)
      .then((payload) => {
        if (active) {
          setState({ status: "ready", payload });
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
  const searchQuery = (searchParams.get("q") ?? "").trim().toLowerCase();
  const visibleSections = filterSectionsBySearch(payload.source_sections, searchQuery);
  const referencedTopicIds = new Set(
    visibleSections.flatMap((section) => section.filteredItems.flatMap((item) => item.topic_refs.map((topic) => topic.topic_id))),
  );
  const visibleTopicSummary = payload.topic_summary.filter((topic) => {
    if (!referencedTopicIds.has(topic.topic_id)) {
      return false;
    }
    if (!searchQuery) {
      return shouldShowDailyTopic(topic);
    }
    return `${topic.headline} ${topic.category} ${topic.llm_status}`.toLowerCase().includes(searchQuery);
  });

  return (
    <div className="stack">
      <section className="daily-summary-strip">
        <div className="day-nav large">
          <div className="day-nav-edge">
            {payload.meta.previous_date ? <Link to={`/hot/${payload.meta.previous_date}`}>Previous Day</Link> : <span>Previous Day</span>}
          </div>
          <div className="day-nav-center">{payload.meta.date}</div>
          <div className="day-nav-edge right">
            {payload.meta.next_date ? <Link to={`/hot/${payload.meta.next_date}`}>Next Day</Link> : <span>Next Day</span>}
          </div>
        </div>
        <div className="feed-toolbar">
          <div className="feed-links">
            <Link to={`/hot/${payload.meta.month}`}>Month</Link>
            <Link to={`/hot/${payload.meta.year}`}>Year</Link>
          </div>
          <div className="feed-stats">
            <span>{payload.meta.counts.source_items} items</span>
            <span>{visibleTopicSummary.length} topics</span>
            <span>{visibleSections.length} sections</span>
          </div>
        </div>
      </section>

      <section className="panel compact-panel">
        <div className="section-header table-header">
          <h2>Daily Topics</h2>
          <div className="section-header-actions">
            <span className="plain-meta">{visibleTopicSummary.length}</span>
          </div>
        </div>
        <TopicSummaryTable topics={visibleTopicSummary} />
      </section>

      {visibleSections.map((section) => (
        <section className="panel source-family-panel compact-panel" id={`section-${section.slug}`} key={section.slug}>
          <div className="section-header table-header">
            <h2>
              <Link className="section-header-link" to={`/hot/${payload.meta.date}/source/${section.slug}`}>
                {sectionTitle(section)}
              </Link>
            </h2>
          </div>
          <SignalTable items={section.filteredItems} />
        </section>
      ))}

      {!visibleSections.length ? (
        <section className="panel">
          <h2>No matching signals</h2>
        </section>
      ) : null}
    </div>
  );
}
