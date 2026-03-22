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
  const busiestDays = [...payload.days]
    .sort((left, right) => right.source_items - left.source_items || right.featured_topics - left.featured_topics)
    .slice(0, 6);

  return (
    <div className="stack">
      <section className="panel day-hero">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to="/hot">Hotspot index</Link>
          </div>
          <div className="day-nav-center">Monthly archive</div>
          <div className="day-nav-edge right">
            <Link to={`/hot/${payload.year}`}>Year archive</Link>
          </div>
        </div>
        <div className="hero-grid">
          <div>
            <p className="eyebrow">Monthly hotspot archive</p>
            <h1>{payload.month}</h1>
            <p className="lede">Daily hotspot coverage, source-family mix, and the busiest days retained for this month.</p>
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
            <h2>Busiest days</h2>
            <p>Highest-volume daily hotspot pages in this month.</p>
          </div>
        </div>
        <div className="featured-grid">
          {busiestDays.map((day) => (
            <Link className="featured-card" key={day.date} to={`/hot/${day.date}`}>
              <span className="card-kicker">{day.date}</span>
              <strong>{day.summary}</strong>
              <div className="card-metrics">
                <span>{day.source_items} items</span>
                <span>{day.featured_topics} featured</span>
                <span>{day.topic_summary} topic-strip</span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-header">
          <div>
            <h2>Daily archive</h2>
            <p>Every hotspot day captured for {payload.month}, with compact source mix for fast backtracking.</p>
          </div>
        </div>
        <div className="archive-grid">
          {payload.days.map((day) => (
            <Link className="archive-card" key={day.date} to={`/hot/${day.date}`}>
              <div className="archive-card-header">
                <strong>{day.date}</strong>
                <span>{day.source_items} items</span>
              </div>
              <p>{day.summary}</p>
              <div className="card-metrics">
                <span>{day.featured_topics} featured</span>
                <span>{day.topic_summary} topic-strip</span>
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
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
