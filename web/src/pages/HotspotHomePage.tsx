import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadRootIndex } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
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
  const latestPaperRoute = bestPaperRoute(latest?.paper_routes, ["day", "month", "year", "home"]);

  return (
    <div className="stack">
      <section className="hero panel compact-panel">
        <div className="hero-grid dense-hero-grid">
          <div>
            <p className="eyebrow">Daily AI Hotspots</p>
            <h1>Dense AI signal radar</h1>
            <p className="lede tight-lede">Daily source-first aggregation across buzz, blogs, official updates, papers, GitHub, and discussions.</p>
          </div>
          <div className="hero-actions">
            {latest ? (
              <Link className="primary-link" to={`/hot/${latest.date}`}>
                Latest hotspots
              </Link>
            ) : null}
            <a className="inline-link" href={latestPaperRoute}>Paper digest</a>
          </div>
        </div>
      </section>

      <section className="panel compact-panel">
        <h2>Latest day</h2>
        {latest ? (
          <div className="stat-row">
            <div>
              <p className="stat-label">Date</p>
              <Link className="stat-value" to={`/hot/${latest.date}`}>
                {latest.date}
              </Link>
            </div>
            <div>
              <p className="stat-label">Featured topics</p>
              <p className="stat-value">{latest.featured_topics}</p>
            </div>
            <div>
              <p className="stat-label">Source items</p>
              <p className="stat-value">{latest.source_items}</p>
            </div>
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
              <a href={bestPaperRoute(entry.paper_routes, ["day", "month", "year", "home"])}>paper</a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
