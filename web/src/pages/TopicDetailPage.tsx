import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { SignalTable } from "../components/SignalTable";
import { loadDailyHotspot } from "../lib/data";
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

  return (
    <div className="stack">
      <section className="panel compact-panel feed-panel">
        <div className="day-nav">
          <div className="day-nav-edge">
            <Link to={`/hot/${date}`}>Back to day</Link>
          </div>
          <div className="day-nav-center">Topic</div>
          <div className="day-nav-edge right">{date}</div>
        </div>

        <div className="feed-toolbar">
          <div className="feed-links">
            <strong>{topic.headline}</strong>
          </div>
          <div className="feed-stats">
            <span>{topic.category}</span>
            <span>{topic.source_names.length} sources</span>
            <span>{supportingGroups.reduce((total, group) => total + group.items.length, 0)} items</span>
          </div>
        </div>
        <div className="section-chip-row compact">
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
            <h2>Evidence</h2>
          </div>
          <div className="archive-row-list">
            {topic.evidence.map((evidence) => (
              <a className="archive-row-link" href={evidence.url} key={evidence.url || evidence.title} target="_blank" rel="noreferrer">
                <div className="archive-row-main">
                  <strong className="archive-row-title">{evidence.title}</strong>
                  <div className="archive-row-meta">
                    <span>{evidence.source_name}</span>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      {supportingGroups.map((group) => (
        <section className="panel source-family-panel compact-panel" key={group.section.slug}>
          <div className="section-header table-header">
            <h2>{group.section.label}</h2>
            <div className="section-header-actions">
              <Link className="inline-link compact-link" to={`/hot/${date}/source/${group.section.slug}`}>
                Detail
              </Link>
            </div>
          </div>
          <SignalTable items={group.items} />
        </section>
      ))}

      {relatedTopics.length > 0 ? (
        <section className="panel compact-panel">
          <div className="section-header">
            <h2>Related topics</h2>
          </div>
          <div className="topic-inline-row">
            {relatedTopics.map((related) => (
              <Link className="topic-inline-link" key={related.topic_id} to={`/hot/${date}/topic/${related.slug}`}>
                <strong>{related.headline}</strong>
                <span>{related.source_names.length} sources</span>
              </Link>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
