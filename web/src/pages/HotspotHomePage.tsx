import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadRootIndex } from "../lib/data";
import type { RootIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: RootIndexPayload };

export function HotspotHomePage() {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadRootIndex()
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
  }, []);

  if (state.status === "loading") {
    return <section className="panel">Loading hotspot index...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load hotspot index: {state.message}</section>;
  }

  const { payload } = state;
  const latest = payload.dates[payload.dates.length - 1];

  return (
    <div className="stack">
      <section className="panel compact-panel">
        <div className="section-header">
          <h1>Daily AI Hotspots</h1>
          <div className="section-header-actions">
            {latest ? (
              <Link className="primary-link" to={`/hot/${latest.date}`}>
                Latest day
              </Link>
            ) : null}
          </div>
        </div>
        {latest ? (
          <div className="summary-band compact-summary-band">
            <Link className="summary-chip" to={`/hot/${latest.date}`}>
              date {latest.date}
            </Link>
            <span className="summary-chip">featured {latest.featured_topics}</span>
            <span className="summary-chip">items {latest.source_items}</span>
          </div>
        ) : (
          <p>No hotspot data has been generated yet.</p>
        )}
      </section>

      <section className="panel compact-panel">
        <h2>Recent daily payloads</h2>
        <ul className="compact-list">
          {payload.dates.slice(-8).reverse().map((entry) => (
            <li key={entry.date}>
              <Link to={`/hot/${entry.date}`}>{entry.date}</Link>
              <span>{entry.featured_topics} featured</span>
              <span>{entry.source_items} source items</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
