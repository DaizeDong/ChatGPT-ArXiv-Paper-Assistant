import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadMonthIndex } from "../lib/data";
import { SOURCE_FAMILY_LABELS } from "../lib/hotspotView";
import type { MonthIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: MonthIndexPayload };

export function MonthArchivePage({ month }: { month: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadMonthIndex(month)
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
  }, [month]);

  if (state.status === "loading") {
    return <section className="panel">Loading month archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load month archive: {state.message}</section>;
  }

  const { payload } = state;
  const sourceMix = Object.entries(payload.source_section_totals)
    .sort((left, right) => right[1] - left[1])
    .filter(([, count]) => count > 0);

  return (
    <div className="stack">
      <section className="panel compact-panel feed-panel">
        <div className="day-nav">
          <div className="day-nav-edge" />
          <div className="day-nav-center">{payload.month}</div>
          <div className="day-nav-edge right">Month</div>
        </div>
        <div className="feed-toolbar">
          <div className="feed-links">
            <Link to={`/hot/${payload.year}`}>Year</Link>
          </div>
          <div className="feed-stats">
            <span>{payload.totals.days} days</span>
            <span>{payload.totals.source_items} items</span>
            <span>{payload.totals.featured_topics} featured</span>
          </div>
        </div>
        <div className="section-chip-row compact">
          {sourceMix.map(([slug, count]) => (
            <span className="section-chip" key={slug}>
              <span>{SOURCE_FAMILY_LABELS[slug] ?? slug} {count}</span>
            </span>
          ))}
        </div>
      </section>

      <section className="panel compact-panel">
        <div className="section-header">
          <h2>Daily archive</h2>
        </div>
        <div className="archive-row-list">
          {payload.days.map((day) => (
            <Link className="archive-row-link" key={day.date} to={`/hot/${day.date}`}>
              <div className="archive-row-main">
                <div className="archive-row-header">
                  <strong className="archive-row-title">{day.date}</strong>
                  <span>{day.source_items} items</span>
                </div>
                <p className="archive-row-summary">{day.summary}</p>
                <div className="archive-row-meta">
                  <span>{day.featured_topics} featured</span>
                  <span>{day.topic_summary} topics</span>
                </div>
                <div className="section-chip-row compact">
                  {Object.entries(day.source_section_counts)
                    .filter(([, count]) => count > 0)
                    .slice(0, 4)
                    .map(([slug, count]) => (
                      <span className="section-chip" key={`${day.date}-${slug}`}>
                        <span>{SOURCE_FAMILY_LABELS[slug] ?? slug} {count}</span>
                      </span>
                    ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
