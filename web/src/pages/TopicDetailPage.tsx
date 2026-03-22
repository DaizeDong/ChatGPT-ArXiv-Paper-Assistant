import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { SignalRow } from "../components/SignalRow";
import { loadDailyHotspot } from "../lib/data";
import { bestPaperRoute } from "../lib/routes";
import { findSupportingItemsForTopic, findTopicBySlug, relatedTopicsByCategory } from "../lib/hotspotSelectors";
import type { DailyHotspotPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

export function TopicDetailPage() {
  const { date = "", topicSlug = "" } = useParams();
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
    return <section className="panel">Loading topic detail...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load topic detail: {state.message}</section>;
  }

  const topic = findTopicBySlug(state.payload, topicSlug);
  if (!topic) {
    return (
      <section className="panel">
        <h1>Unknown topic</h1>
        <p>No topic with slug <code>{topicSlug}</code> exists for {date}.</p>
      </section>
    );
  }

  const supportingGroups = findSupportingItemsForTopic(state.payload, topic.topic_id);
  const relatedTopics = relatedTopicsByCategory(state.payload, topic, 10);
  const paperDayRoute = bestPaperRoute(state.payload.meta.paper_routes, ["day", "month", "year", "home"]);
  const paperArchiveRoute = bestPaperRoute(state.payload.meta.paper_routes, ["month", "year", "home"]);

  return (
    <div className="stack">
      <section className="panel day-hero compact-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to={`/hot/${date}`}>Back to day</Link>
          </div>
          <Link className="day-nav-center" to="/hot">
            Hotspot index
          </Link>
          <div className="day-nav-edge right">
            <a href={paperDayRoute}>Paper digest</a>
          </div>
        </div>

        <div className="hero-grid dense-hero-grid">
          <div>
            <p className="eyebrow">{topic.category}</p>
            <h1>{topic.headline}</h1>
            <p className="lede tight-lede">{topic.why_it_matters || topic.summary_short}</p>
          </div>
          <div className="hero-actions">
            <a className="inline-link" href={paperArchiveRoute}>Paper archive</a>
            <Link className="inline-link" to={`/hot/${date}/source/${supportingGroups[0]?.section.slug ?? "blogs"}`}>
              Source detail
            </Link>
            <Link className="primary-link" to={`/hot/${date}`}>
              Return to daily view
            </Link>
          </div>
        </div>

        <div className="summary-band">
          <div>
            <p className="stat-label">Final score</p>
            <p className="stat-value">{topic.scores.final.toFixed(1)}</p>
          </div>
          <div>
            <p className="stat-label">Heat</p>
            <p className="stat-value">{topic.scores.heat.toFixed(1)}</p>
          </div>
          <div>
            <p className="stat-label">Sources</p>
            <p className="stat-value">{topic.source_names.length}</p>
          </div>
          <div>
            <p className="stat-label">Evidence items</p>
            <p className="stat-value">{supportingGroups.reduce((total, group) => total + group.items.length, 0)}</p>
          </div>
        </div>

        <div className="section-chip-row">
          {topic.source_names.map((sourceName) => (
            <span className="section-chip" key={sourceName}>
              <span>{sourceName}</span>
            </span>
          ))}
        </div>
      </section>

      {topic.evidence.length > 0 ? (
        <section className="panel compact-panel">
          <div className="section-header">
            <div>
              <h2>Representative evidence</h2>
            </div>
          </div>
          <div className="featured-grid">
            {topic.evidence.map((evidence) => (
              <a className="featured-card" href={evidence.url} key={evidence.url || evidence.title} target="_blank" rel="noreferrer">
                <span className="card-kicker">{evidence.source_name}</span>
                <strong>{evidence.title}</strong>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      {supportingGroups.map((group) => (
        <section className="panel source-family-panel compact-panel" key={group.section.slug}>
          <div className="section-header">
            <div>
              <h2>{group.section.label}</h2>
            </div>
            <div className="section-header-actions">
              <Link className="inline-link" to={`/hot/${date}/source/${group.section.slug}`}>
                Open full source page
              </Link>
            </div>
          </div>
          <div className="signal-list">
            {group.items.map((item) => (
              <SignalRow date={date} density="comfortable" item={item} key={item.id} />
            ))}
          </div>
        </section>
      ))}

      {relatedTopics.length > 0 ? (
        <section className="panel compact-panel">
          <div className="section-header">
            <div>
              <h2>Related topics in the same category</h2>
            </div>
          </div>
          <div className="topic-strip">
            {relatedTopics.map((related) => (
              <Link className="topic-pill" key={related.topic_id} to={`/hot/${date}/topic/${related.slug}`}>
                <span>{related.headline}</span>
                <small>{related.source_names.length} sources</small>
              </Link>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
