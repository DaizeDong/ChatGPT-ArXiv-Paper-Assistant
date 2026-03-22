import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadYearIndex } from "../lib/data";
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

  return (
    <div className="stack">
      <section className="panel day-hero">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to="/hot">Hotspot index</Link>
          </div>
          <div className="day-nav-center">Yearly archive</div>
          <div className="day-nav-edge right">
            <span>{payload.months.length} active months</span>
          </div>
        </div>
        <div className="hero-grid">
          <div>
            <p className="eyebrow">Yearly hotspot archive</p>
            <h1>{payload.year}</h1>
            <p className="lede">Month-level overview of how the hotspot system distributed attention across the year.</p>
          </div>
        </div>
        <div className="summary-band">
          <div>
            <p className="stat-label">Days</p>
            <p className="stat-value">{payload.totals.days}</p>
          </div>
          <div>
            <p className="stat-label">Source items</p>
            <p className="stat-value">{payload.totals.source_items}</p>
          </div>
          <div>
            <p className="stat-label">Featured topics</p>
            <p className="stat-value">{payload.totals.featured_topics}</p>
          </div>
          <div>
            <p className="stat-label">Topic strip items</p>
            <p className="stat-value">{payload.totals.topic_summary}</p>
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

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>Monthly archive</h2>
            <p>Each month retains its own aggregate mix so you can see how the hotspot surface shifted over time.</p>
          </div>
        </div>
        <div className="archive-grid">
          {payload.months.map((monthRow) => (
            <Link className="archive-card" key={monthRow.month} to={`/hot/${monthRow.month}`}>
              <div className="archive-card-header">
                <strong>{monthRow.month}</strong>
                <span>{monthRow.days} days</span>
              </div>
              <div className="card-metrics">
                <span>{monthRow.source_items} items</span>
                <span>{monthRow.featured_topics} featured</span>
                <span>{monthRow.topic_summary} topic-strip</span>
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
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
