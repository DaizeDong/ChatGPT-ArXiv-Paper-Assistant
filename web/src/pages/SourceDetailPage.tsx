import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { SignalTable } from "../components/SignalTable";
import { loadDailyHotspot } from "../lib/data";
import { findSourceSection } from "../lib/hotspotSelectors";
import type { DailyHotspotPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

export function SourceDetailPage() {
  const { date = "", sourceSlug = "" } = useParams();
  const [state, setState] = useState<AsyncState>({ status: "loading" });

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

  return (
    <div className="stack">
      <section className="panel compact-panel feed-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to={`/hot/${date}`}>Back to day</Link>
          </div>
          <div className="day-nav-center">{section.label}</div>
          <div className="day-nav-edge right">{date}</div>
        </div>
        <div className="feed-toolbar">
          <div className="feed-links">
            <strong>{section.label}</strong>
          </div>
          <div className="feed-stats">
            <span>{section.count} items</span>
            <span>{linkedTopicCounts.size} topics</span>
          </div>
        </div>
      </section>

      {topTopics.length > 0 ? (
        <section className="panel compact-panel">
          <div className="topic-inline-row">
            {topTopics.map((topic) => (
              <Link key={topic.slug} className="topic-inline-link" to={`/hot/${date}/topic/${topic.slug}`}>
                <strong>{topic.headline}</strong>
                <span>{topic.count} items</span>
              </Link>
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel source-family-panel compact-panel">
        <div className="section-header table-header">
          <h2>{section.label} signals</h2>
        </div>
        <SignalTable items={section.items} />
      </section>
    </div>
  );
}
