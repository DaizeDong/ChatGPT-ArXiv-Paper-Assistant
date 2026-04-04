import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ArchiveNav } from "../components/ArchiveNav";
import { CategoryRadar } from "../components/CategoryRadar";
import { CrossSiteSwitch } from "../components/CrossSiteSwitch";
import { FeaturedStories } from "../components/FeaturedStories";
import { SignalTable } from "../components/SignalTable";
import { isNotFoundError, loadDailyHotspot } from "../lib/data";
import { filterSectionsBySearch } from "../lib/hotspotView";
import { useI18n } from "../lib/i18n";
import { bestPaperRoute } from "../lib/routes";
import type { CompactTopic, DailyHotspotPayload, SourceSection } from "../types/hotspot";
import { NotFoundPage } from "./NotFoundPage";

type AsyncState =
  | { status: "loading" }
  | { status: "notFound" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: DailyHotspotPayload };

function sectionTitle(section: SourceSection) {
  return `${section.label} (${section.count})`;
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
    return "\u2014";
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
  const { t } = useI18n();

  useEffect(() => {
    let active = true;
    loadDailyHotspot(date)
      .then((payload) => {
        if (active) {
          setState({ status: "ready", payload });
        }
      })
      .catch((error: unknown) => {
        if (active) {
          if (isNotFoundError(error)) {
            setState({ status: "notFound" });
            return;
          }
          setState({ status: "error", message: error instanceof Error ? error.message : String(error) });
        }
      });
    return () => {
      active = false;
    };
  }, [date]);

  if (state.status === "loading") {
    return <section className="panel">Loading daily hotspot payload...</section>;
  }
  if (state.status === "notFound") {
    return <NotFoundPage />;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load {date}: {state.message}</section>;
  }

  const { payload } = state;
  const searchQuery = (searchParams.get("q") ?? "").trim().toLowerCase();
  const visiblePaperSpotlight = filterSectionsBySearch(payload.paper_spotlight ?? [], searchQuery);
  // "Other Updates": source items not covered by any topic AND not papers (papers live in Paper Spotlight)
  const otherSections = filterSectionsBySearch(
    (payload.source_sections ?? []).filter((sec) => sec.slug !== "papers"),
    searchQuery,
    true,
  );
  const paperHref = bestPaperRoute(payload.meta.paper_routes, ["day", "month", "year", "home"]);
  const usageRows = buildUsageRows(payload);

  // Merge category_sections, long_tail_sections, and watchlist into a unified list
  const watchlistByCategory = new Map<string, CompactTopic[]>();
  for (const tp of payload.watchlist ?? []) {
    const cat = tp.category || "Other";
    const list = watchlistByCategory.get(cat);
    if (list) list.push(tp);
    else watchlistByCategory.set(cat, [tp]);
  }
  const watchlistSections = Array.from(watchlistByCategory.entries()).map(([category, topics]) => ({
    category,
    total_candidates: topics.length,
    topics,
  }));
  const allCategorySections = [...(payload.category_sections ?? []), ...(payload.long_tail_sections ?? []), ...watchlistSections];

  const counts = payload.meta.counts;

  return (
    <div className="stack hotspot-stack">
      <section className="archive-head">
        <div className="archive-titlebar">
          <h1>{t("site.title")} {payload.meta.date}</h1>
          <div className="archive-titlebar-controls">
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
                placeholder={t("label.search")}
              />
            </label>
          </div>
        </div>
        <ArchiveNav
          previous={
            payload.meta.previous_date
              ? {
                  href: `/hot/${payload.meta.previous_date}`,
                  title: t("nav.prev"),
                  subtitle: payload.meta.previous_date,
                  arrow: "left",
                }
              : null
          }
          center={{
            href: `/hot/${payload.meta.month}`,
            title: t("nav.monthly"),
            subtitle: payload.meta.month,
          }}
          next={
            payload.meta.next_date
              ? {
                  href: `/hot/${payload.meta.next_date}`,
                  title: t("nav.next"),
                  subtitle: payload.meta.next_date,
                  arrow: "right",
                }
              : null
          }
        />
        <CrossSiteSwitch href={paperHref} label={t("nav.paper")} />
        <div className="archive-head-meta">
          <span>{payload.meta.date}</span>
          <span>{counts.featured_topics ?? 0} {t("label.featured")}</span>
          <span>{counts.category_radar ?? 0} {t("label.categorized")}</span>
          <span>{counts.paper_spotlight ?? 0} {t("label.spotlight")}</span>
          <span>{counts.source_items ?? 0} {t("label.otherItems")}</span>
        </div>
      </section>

      {/* Featured Stories */}
      <FeaturedStories topics={payload.featured_topics} searchQuery={searchQuery} />

      {/* Paper Spotlight */}
      {visiblePaperSpotlight.map((section) => (
        <section className="panel source-family-panel compact-panel" id={`section-${section.slug}`} key={section.slug}>
          <div className="section-header table-header">
            <h2>{sectionTitle(section)}</h2>
          </div>
          <SignalTable items={section.filteredItems} />
        </section>
      ))}

      {/* All Topics */}
      <CategoryRadar title={t("section.topics")} sections={allCategorySections} searchQuery={searchQuery} />

      {/* Other Updates — non-paper items not covered by any topic */}
      {otherSections.length > 0 ? (() => {
        const otherItems = otherSections.flatMap((sec) => sec.filteredItems);
        return (
          <section className="panel source-family-panel compact-panel" id="other-updates">
            <div className="section-header table-header">
              <h2>{t("section.other")} ({otherItems.length})</h2>
            </div>
            <SignalTable items={otherItems} />
          </section>
        );
      })() : null}

      {!otherSections.length && !visiblePaperSpotlight.length && !payload.featured_topics.length ? (
        <section className="panel">
          <h2>{t("label.noSignals")}</h2>
        </section>
      ) : null}

      {/* Usage Table */}
      {usageRows.length ? (
        <section className="panel compact-panel">
          <div className="section-header table-header">
            <h2>{t("section.usage")}</h2>
          </div>
          <div className="table-wrap">
            <table className="signal-table usage-table">
              <thead>
                <tr>
                  <th className="col-title">{t("usage.api")}</th>
                  <th className="col-signal align-left">{t("usage.mode")}</th>
                  <th className="col-score align-center">{t("usage.requests")}</th>
                  <th className="col-score align-center">{t("usage.items")}</th>
                  <th className="col-score align-center">{t("usage.prompt")}</th>
                  <th className="col-score align-center">{t("usage.completion")}</th>
                  <th className="col-heat align-center">{t("usage.cost")}</th>
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
    </div>
  );
}
