import { Link } from "react-router-dom";

import { itemSignals, type DensityMode } from "../lib/hotspotView";
import type { SourceSectionItem } from "../types/hotspot";

export function SignalRow({
  date,
  density,
  item,
}: {
  date: string;
  density: DensityMode;
  item: SourceSectionItem;
}) {
  const signals = itemSignals(item);
  const topicRefs = item.topic_refs.slice(0, density === "compact" ? 1 : 2);

  return (
    <article className={`signal-row ${density}`}>
      <div className="signal-main">
        <a className="signal-title" href={item.url} target="_blank" rel="noreferrer">
          {item.title}
        </a>
        <div className="signal-meta">
          <span>{item.source_name}</span>
          <span>{item.source_role.replaceAll("_", " ")}</span>
          <span>score {item.signal_score.toFixed(1)}</span>
          {signals.map((signal) => (
            <span key={signal}>{signal}</span>
          ))}
        </div>
        {density === "comfortable" ? <p className="signal-summary">{item.summary_short}</p> : null}
      </div>
      {topicRefs.length > 0 ? (
        <div className="signal-topics">
          {topicRefs.map((topic) => (
            <Link key={topic.topic_id} className="signal-topic-pill" to={`/hot/${date}/topic/${topic.slug}`}>
              {topic.headline}
            </Link>
          ))}
        </div>
      ) : null}
    </article>
  );
}
