import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadYearIndex } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
import { SOURCE_FAMILY_LABELS } from "../lib/hotspotView";
import type { YearIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: YearIndexPayload };

export function YearArchivePage({ year }: { year: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadYearIndex(year)
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
  }, [year]);

  if (state.status === "loading") {
    return <section className="panel">Loading year archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load year archive: {state.message}</section>;
  }

  const { payload } = state;
  const sourceMix = Object.entries(payload.source_section_totals)
    .sort((left, right) => right[1] - left[1])
    .filter(([, count]) => count > 0);
  const paperYearRoute = bestPaperRoute(payload.paper_routes, ["year", "home"]);

  return (
    <div className="stack">
      <section className="panel day-hero compact-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to="/hot">Hotspot index</Link>
          </div>
          <div className="day-nav-center">Yearly archive</div>
          <div className="day-nav-edge right">
            <a href={paperYearRoute}>Paper year</a>
          </div>
        </div>
        <div className="hero-grid dense-hero-grid">
          <div>
            <p className="eyebrow">Yearly hotspot archive</p>
            <h1>{payload.year}</h1>
            <div className="summary-band compact-summary-band">
              <span className="summary-chip">months {payload.months.length}</span>
              <span className="summary-chip">days {payload.totals.days}</span>
              <span className="summary-chip">items {payload.totals.source_items}</span>
              <span className="summary-chip">featured {payload.totals.featured_topics}</span>
            </div>
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
            <h2>Monthly archive</h2>
          </div>
        </div>
        <div className="archive-row-list">
          {payload.months.map((monthRow) => (
            <Link className="archive-row-link" key={monthRow.month} to={`/hot/${monthRow.month}`}>
              <div className="archive-row-main">
                <div className="archive-row-header">
                  <strong className="archive-row-title">{monthRow.month}</strong>
                  <span>{monthRow.days} days</span>
                </div>
                <div className="archive-row-meta">
                  <span>{monthRow.source_items} items</span>
                  <span>{monthRow.featured_topics} featured</span>
                  <span>{monthRow.topic_summary} topics</span>
                </div>
                <div className="section-chip-row compact">
                  {Object.entries(monthRow.source_section_totals)
                    .filter(([, count]) => count > 0)
                    .slice(0, 4)
                    .map(([slug, count]) => (
                      <span className="section-chip" key={`${monthRow.month}-${slug}`}>
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
