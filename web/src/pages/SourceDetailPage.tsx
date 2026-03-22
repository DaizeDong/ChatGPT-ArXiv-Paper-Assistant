import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { SignalRow } from "../components/SignalRow";
import { loadDailyHotspot } from "../lib/data";
import { findSourceSection } from "../lib/hotspotSelectors";
import { defaultVisibleCount, type DensityMode } from "../lib/hotspotView";
import type { DailyHotspotPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

export function SourceDetailPage() {
  const { date = "", sourceSlug = "" } = useParams();
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [density, setDensity] = useState<DensityMode>("compact");
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
    setVisibleCount(density === "compact" ? 12 : 10);
  }, [density, sourceSlug]);

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

  return (
    <div className="stack">
      <section className="panel day-hero">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to={`/hot/${date}`}>Back to day</Link>
          </div>
          <Link className="day-nav-center" to="/hot">
            Hotspot index
          </Link>
          <div className="day-nav-edge right">
            <Link to={`/hot/${state.payload.meta.month}`}>Month archive</Link>
          </div>
        </div>
        <div className="hero-grid">
          <div>
            <p className="eyebrow">Source family detail</p>
            <h1>{section.label}</h1>
            <p className="lede">{section.description}</p>
          </div>
          <div className="hero-actions">
            <div className="toggle-group" role="tablist" aria-label="Detail density">
              <button className={density === "compact" ? "active" : ""} onClick={() => setDensity("compact")} type="button">
                Compact
              </button>
              <button className={density === "comfortable" ? "active" : ""} onClick={() => setDensity("comfortable")} type="button">
                Comfortable
              </button>
            </div>
          </div>
        </div>
        <div className="summary-band">
          <div>
            <p className="stat-label">Items</p>
            <p className="stat-value">{section.count}</p>
          </div>
          <div>
            <p className="stat-label">Linked topics</p>
            <p className="stat-value">{linkedTopicCounts.size}</p>
          </div>
          <div>
            <p className="stat-label">Daily route</p>
            <p className="stat-value">{date}</p>
          </div>
        </div>
      </section>

      {topTopics.length > 0 ? (
        <section className="panel">
          <div className="section-header">
            <div>
              <h2>Most-linked topics in this source family</h2>
              <p>These are the repeated topic references that appear most often inside {section.label}.</p>
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

      <section className="panel source-family-panel">
        <div className="section-header">
          <div>
            <h2>All {section.label} signals</h2>
            <p>Dense view of every retained item for this source family on {date}.</p>
          </div>
        </div>
        <div className="signal-list">
          {displayed.map((item) => (
            <SignalRow date={date} density={density} item={item} key={item.id} />
          ))}
        </div>
        {remaining > 0 ? (
          <button className="show-more-button" onClick={() => setVisibleCount(visibleCount + defaultVisibleCount(section, density))} type="button">
            Show {Math.min(defaultVisibleCount(section, density), remaining)} more from {section.label}
          </button>
        ) : null}
      </section>
    </div>
  );
}
