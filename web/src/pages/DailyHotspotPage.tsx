import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadDailyHotspot } from "../lib/data";
import type { DailyHotspotPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

export function DailyHotspotPage({ date }: { date: string }) {
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
    return <section className="panel">Loading daily hotspot payload...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load {date}: {state.message}</section>;
  }

  const { payload } = state;

  return (
    <div className="stack">
      <section className="panel hero">
        <div className="nav-strip">
          <Link to="/hot">Back to index</Link>
          <Link to={`/hot/${payload.meta.month}`}>Month archive</Link>
        </div>
        <h1>Daily AI Hotspots {payload.meta.date}</h1>
        <p className="lede">{payload.meta.summary}</p>
        <div className="stat-row">
          <div>
            <p className="stat-label">Featured topics</p>
            <p className="stat-value">{payload.meta.counts.featured_topics}</p>
          </div>
          <div>
            <p className="stat-label">Source items</p>
            <p className="stat-value">{payload.meta.counts.source_items}</p>
          </div>
          <div>
            <p className="stat-label">Topic summary</p>
            <p className="stat-value">{payload.meta.counts.topic_summary}</p>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Source families</h2>
        <div className="grid two-up">
          {payload.source_sections.map((section) => (
            <article className="subpanel" key={section.slug}>
              <div className="subpanel-header">
                <div>
                  <h3>{section.label}</h3>
                  <p>{section.description}</p>
                </div>
                <span className="chip">{section.count}</span>
              </div>
              <ul className="compact-list">
                {section.items.slice(0, 4).map((item) => (
                  <li key={item.id}>
                    <a href={item.url} target="_blank" rel="noreferrer">
                      {item.title}
                    </a>
                    <span>{item.source_name}</span>
                  </li>
                ))}
              </ul>
              <Link className="inline-link" to={`/hot/${payload.meta.date}/source/${section.slug}`}>
                Open source detail
              </Link>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Topic strip</h2>
        <div className="topic-strip">
          {payload.topic_summary.slice(0, 12).map((topic) => (
            <Link className="topic-pill" key={topic.topic_id} to={`/hot/${payload.meta.date}/topic/${topic.slug}`}>
              <span>{topic.headline}</span>
              <small>{topic.source_count} sources</small>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
