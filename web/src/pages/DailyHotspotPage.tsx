import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ArchiveNav } from "../components/ArchiveNav";
import { CrossSiteSwitch } from "../components/CrossSiteSwitch";
import { SignalTable } from "../components/SignalTable";
import { TopicSummaryTable, type TopicSummaryRow } from "../components/TopicSummaryTable";
import { loadDailyHotspot } from "../lib/data";
import { filterSectionsBySearch } from "../lib/hotspotView";
import { bestPaperRoute } from "../lib/routes";
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

type UsageRow = {
  id: string;
  api: string;
  mode: string;
  requests: number | null;
  items: number | null;
  prompt: number | null;
  completion: number | null;
  cost: number;
};

function formatCount(value: number | null) {
  if (value === null) {
    return "—";
  }
  return value.toLocaleString();
}

function formatCost(value: number) {
  return `$${value.toFixed(6)}`;
}

function buildUsageRows(payload: DailyHotspotPayload): UsageRow[] {
  const rows: UsageRow[] = [];
  const llm = payload.usage?.llm;
  if (llm) {
    rows.push({
      id: "openai-llm",
      api: "OpenAI LLM",
      mode: [llm.screen_model, llm.summary_model].filter(Boolean).join(" / ") || (llm.billing_model ?? "quota"),
      requests: llm.requests ?? 0,
      items: null,
      prompt: llm.prompt_tokens ?? 0,
      completion: llm.completion_tokens ?? 0,
      cost: llm.total_cost ?? 0,
    });
  }

  const externalEntries = Object.entries(payload.usage?.external ?? {})
    .filter(([, row]) => (row.requests ?? 0) > 0 || row.billing_model === "quota" || (row.items ?? 0) > 0)
    .sort((left, right) => {
      const leftQuota = left[1].billing_model === "quota" ? 1 : 0;
      const rightQuota = right[1].billing_model === "quota" ? 1 : 0;
      if (rightQuota !== leftQuota) {
        return rightQuota - leftQuota;
      }
      if ((right[1].requests ?? 0) !== (left[1].requests ?? 0)) {
        return (right[1].requests ?? 0) - (left[1].requests ?? 0);
      }
      return left[0].localeCompare(right[0], undefined, { sensitivity: "base" });
    });

  for (const [sourceId, row] of externalEntries) {
    rows.push({
      id: sourceId,
      api: row.provider || sourceId,
      mode: row.billing_model || "unknown",
      requests: row.requests ?? 0,
      items: row.items ?? 0,
      prompt: null,
      completion: null,
      cost: row.estimated_cost ?? 0,
    });
  }

  return rows;
}

export function DailyHotspotPage({ date }: { date: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });
  const [searchParams, setSearchParams] = useSearchParams();

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
  const paperHref = bestPaperRoute(payload.meta.paper_routes, ["day", "month", "year", "home"]);
  const usageRows = buildUsageRows(payload);

  return (
    <div className="stack hotspot-stack">
      <section className="archive-head">
        <div className="archive-titlebar">
          <h1>Daily AI Hotspots {payload.meta.date}</h1>
          <label className="nav-search archive-search">
            <input
              type="search"
              value={searchParams.get("q") ?? ""}
              onChange={(event) => {
                const next = new URLSearchParams(searchParams);
                const value = event.target.value.trimStart();
                if (value) {
                  next.set("q", value);
                } else {
                  next.delete("q");
                }
                setSearchParams(next, { replace: true });
              }}
              placeholder="Search"
            />
          </label>
        </div>
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
        <CrossSiteSwitch href={paperHref} label="Personalized Daily Arxiv Paper" />
        <div className="archive-head-meta">
          <span>{payload.meta.date}</span>
          <span>{payload.meta.counts.source_items} items</span>
          <span>{visibleTopicSummary.length} topics</span>
          <span>{visibleSections.length} sections</span>
        </div>
      </section>

      {usageRows.length ? (
        <section className="panel compact-panel">
          <div className="section-header table-header">
            <h2>Daily Usage</h2>
          </div>
          <div className="table-wrap">
            <table className="signal-table usage-table">
              <thead>
                <tr>
                  <th className="col-title">API</th>
                  <th className="col-signal align-left">Mode</th>
                  <th className="col-score align-center">Requests</th>
                  <th className="col-score align-center">Items</th>
                  <th className="col-score align-center">Prompt</th>
                  <th className="col-score align-center">Completion</th>
                  <th className="col-heat align-center">Cost</th>
                </tr>
              </thead>
              <tbody>
                {usageRows.map((row) => (
                  <tr key={row.id}>
                    <td className="col-title">
                      <span className="title-link">{row.api}</span>
                    </td>
                    <td className="col-signal">{row.mode}</td>
                    <td className="col-score align-center">{formatCount(row.requests)}</td>
                    <td className="col-score align-center">{formatCount(row.items)}</td>
                    <td className="col-score align-center">{formatCount(row.prompt)}</td>
                    <td className="col-score align-center">{formatCount(row.completion)}</td>
                    <td className="col-heat align-center">{formatCost(row.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

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
