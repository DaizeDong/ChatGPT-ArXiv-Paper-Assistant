import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ArchiveNav } from "../components/ArchiveNav";
import { SignalTable } from "../components/SignalTable";
import { TopicSummaryTable, type TopicSummaryRow } from "../components/TopicSummaryTable";
import { loadDailyHotspot } from "../lib/data";
import { filterSectionsBySearch } from "../lib/hotspotView";
import type { DailyHotspotPayload, SourceSection } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

function sectionTitle(section: SourceSection) {
  return `${section.label} (${section.count})`;
}

function buildVisibleTopicSummary(payload: DailyHotspotPayload, searchQuery: string) {
  return payload.topic_summary
    .filter((topic) => topic.source_count >= 2)
    .filter((topic) => !searchQuery || `${topic.headline} ${topic.category} ${topic.llm_status}`.toLowerCase().includes(searchQuery))
    .map<TopicSummaryRow>((topic) => ({
      topic_id: topic.topic_id,
      headline: topic.headline,
      final_score: topic.final_score,
      heat: topic.heat,
      display_priority: topic.display_priority,
      source_count: topic.source_count,
      item_count: topic.item_count,
    }))
    .sort((left, right) => {
      if (right.source_count !== left.source_count) {
        return right.source_count - left.source_count;
      }
      if ((right.item_count ?? 0) !== (left.item_count ?? 0)) {
        return (right.item_count ?? 0) - (left.item_count ?? 0);
      }
      if ((right.display_priority ?? right.final_score) !== (left.display_priority ?? left.final_score)) {
        return (right.display_priority ?? right.final_score) - (left.display_priority ?? left.final_score);
      }
      if (right.heat !== left.heat) {
        return right.heat - left.heat;
      }
      return left.headline.localeCompare(right.headline, undefined, { sensitivity: "base" });
    });
}

export function DailyHotspotPage({ date }: { date: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [searchParams] = useSearchParams();

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
  const searchQuery = (searchParams.get("q") ?? "").trim().toLowerCase();
  const visibleSections = filterSectionsBySearch(payload.source_sections, searchQuery);
  const visibleTopicSummary = buildVisibleTopicSummary(payload, searchQuery);

  return (
    <div className="stack hotspot-stack">
      <section className="archive-head">
        <ArchiveNav
          previous={
            payload.meta.previous_date
              ? {
                  href: `/hot/${payload.meta.previous_date}`,
                  title: "Previous Day",
                  subtitle: payload.meta.previous_date,
                  arrow: "left",
                }
              : null
          }
          center={{
            href: `/hot/${payload.meta.month}`,
            title: "Monthly Overview",
            subtitle: payload.meta.month,
          }}
          next={
            payload.meta.next_date
              ? {
                  href: `/hot/${payload.meta.next_date}`,
                  title: "Next Day",
                  subtitle: payload.meta.next_date,
                  arrow: "right",
                }
              : null
          }
        />
        <div className="archive-head-meta">
          <span>{payload.meta.date}</span>
          <span>{payload.meta.counts.source_items} items</span>
          <span>{visibleTopicSummary.length} topics</span>
          <span>{visibleSections.length} sections</span>
        </div>
      </section>

      <section className="panel compact-panel">
        <div className="section-header table-header">
          <h2>Daily Topics</h2>
          <div className="section-header-actions">
            <span className="plain-meta">{visibleTopicSummary.length}</span>
          </div>
        </div>
        <TopicSummaryTable topics={visibleTopicSummary} />
      </section>

      {visibleSections.map((section) => (
        <section className="panel source-family-panel compact-panel" id={`section-${section.slug}`} key={section.slug}>
          <div className="section-header table-header">
            <h2>{sectionTitle(section)}</h2>
          </div>
          <SignalTable items={section.filteredItems} />
        </section>
      ))}

      {!visibleSections.length ? (
        <section className="panel">
          <h2>No matching signals</h2>
        </section>
      ) : null}
    </div>
  );
}
