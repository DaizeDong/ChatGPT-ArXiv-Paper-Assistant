import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadMonthIndex } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
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
  const paperMonthRoute = bestPaperRoute(payload.paper_routes, ["month", "year", "home"]);
  const paperYearRoute = bestPaperRoute(payload.paper_routes, ["year", "home"]);

  return (
    <div className="stack">
      <section className="panel day-hero compact-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to="/hot">Hotspot index</Link>
          </div>
          <div className="day-nav-center">Monthly archive</div>
          <div className="day-nav-edge right">
            <a href={paperMonthRoute}>Paper month</a>
          </div>
        </div>
        <div className="hero-grid dense-hero-grid">
          <div>
            <p className="eyebrow">Monthly hotspot archive</p>
            <h1>{payload.month}</h1>
            <div className="summary-band compact-summary-band">
              <span className="summary-chip">days {payload.totals.days}</span>
              <span className="summary-chip">items {payload.totals.source_items}</span>
              <span className="summary-chip">featured {payload.totals.featured_topics}</span>
              <span className="summary-chip">topics {payload.totals.topic_summary}</span>
            </div>
          </div>
          <div className="hero-actions">
            <Link className="inline-link" to={`/hot/${payload.year}`}>Hot year</Link>
            <a className="inline-link" href={paperYearRoute}>Paper year</a>
          </div>
        </div>
        <div className="section-chip-row">
          {sourceMix.map(([slug, count]) => (
            <span className="section-chip" key={slug}>
              <span>{SOURCE_FAMILY_LABELS[slug] ?? slug}</span>
              <strong>{count}</strong>
            </span>
          ))}
        </div>
      </section>

      <section className="panel compact-panel">
        <div className="section-header">
          <div>
            <h2>Daily archive</h2>
          </div>
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
                        <span>{SOURCE_FAMILY_LABELS[slug] ?? slug}</span>
                        <strong>{count}</strong>
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
