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
      <section className="hero panel">
        <div>
          <p className="eyebrow">Daily AI Hotspots</p>
          <h1>Source-first AI news radar.</h1>
          <p className="lede">
            The hotspot frontend now reads dense JSON feeds from <code>out/web_data</code> and exposes daily, monthly,
            yearly, source, and topic routes without depending on markdown rendering.
          </p>
        </div>
        {latest ? (
          <Link className="primary-link" to={`/hot/${latest.date}`}>
            Open latest hotspot day
          </Link>
        ) : null}
      </section>

      <section className="panel">
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

      <section className="panel">
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
