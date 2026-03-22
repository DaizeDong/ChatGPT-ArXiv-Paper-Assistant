import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { SignalRow } from "../components/SignalRow";
import { loadDailyHotspot } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
import { findSourceSection } from "../lib/hotspotSelectors";
import { defaultVisibleCount } from "../lib/hotspotView";
import type { DailyHotspotPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

export function SourceDetailPage() {
  const { date = "", sourceSlug = "" } = useParams();
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [visibleCount, setVisibleCount] = useState(12);

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

  useEffect(() => {
    setVisibleCount(12);
  }, [sourceSlug]);

  if (state.status === "loading") {
    return <section className="panel">Loading source detail...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load source detail: {state.message}</section>;
  }

  const section = findSourceSection(state.payload, sourceSlug);
  if (!section) {
    return (
      <section className="panel">
        <h1>Unknown source section</h1>
        <p>No section named <code>{sourceSlug}</code> exists for {date}.</p>
      </section>
    );
  }

  const linkedTopicCounts = new Map<string, { headline: string; count: number; slug: string }>();
  for (const item of section.items) {
    for (const topic of item.topic_refs) {
      const existing = linkedTopicCounts.get(topic.topic_id);
      if (existing) {
        existing.count += 1;
      } else {
        linkedTopicCounts.set(topic.topic_id, { headline: topic.headline, count: 1, slug: topic.slug });
      }
    }
  }
  const topTopics = Array.from(linkedTopicCounts.values())
    .sort((left, right) => right.count - left.count || left.headline.localeCompare(right.headline))
    .slice(0, 12);

  const displayed = section.items.slice(0, visibleCount);
  const remaining = Math.max(section.items.length - displayed.length, 0);
  const paperDayRoute = bestPaperRoute(state.payload.meta.paper_routes, ["day", "month", "year", "home"]);
  const paperArchiveRoute = bestPaperRoute(state.payload.meta.paper_routes, ["month", "year", "home"]);

  return (
    <div className="stack">
      <section className="panel day-hero compact-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to={`/hot/${date}`}>Back to day</Link>
          </div>
          <Link className="day-nav-center" to="/hot">
            Hotspot index
          </Link>
          <div className="day-nav-edge right">
            <a href={paperDayRoute}>Paper digest</a>
          </div>
        </div>
        <div className="hero-grid dense-hero-grid">
          <div>
            <p className="eyebrow">Source view</p>
            <h1>{section.label}</h1>
            <div className="summary-band compact-summary-band">
              <span className="summary-chip">items {section.count}</span>
              <span className="summary-chip">linked topics {linkedTopicCounts.size}</span>
              <span className="summary-chip">day {date}</span>
            </div>
          </div>
          <div className="hero-actions">
            <a className="inline-link" href={paperArchiveRoute}>Paper archive</a>
          </div>
        </div>
      </section>

      {topTopics.length > 0 ? (
        <section className="panel compact-panel">
          <div className="section-header">
            <div>
              <h2>Linked topics</h2>
            </div>
          </div>
          <div className="topic-strip">
            {topTopics.map((topic) => (
              <Link key={topic.slug} className="topic-pill" to={`/hot/${date}/topic/${topic.slug}`}>
                <span>{topic.headline}</span>
                <small>{topic.count} linked items</small>
              </Link>
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel source-family-panel compact-panel">
        <div className="section-header">
          <div>
            <h2>{section.label} signals</h2>
          </div>
        </div>
          <div className="signal-list">
            {displayed.map((item) => (
              <SignalRow date={date} density="compact" item={item} key={item.id} />
            ))}
          </div>
        {remaining > 0 ? (
          <button className="show-more-button" onClick={() => setVisibleCount(visibleCount + defaultVisibleCount(section, "compact"))} type="button">
            Show {Math.min(defaultVisibleCount(section, "compact"), remaining)} more from {section.label}
          </button>
        ) : null}
      </section>
    </div>
  );
}
