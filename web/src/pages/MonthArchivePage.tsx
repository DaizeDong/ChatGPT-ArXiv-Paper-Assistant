import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ArchiveNav } from "../components/ArchiveNav";
import { CrossSiteSwitch } from "../components/CrossSiteSwitch";
import { loadMonthIndex, loadRootIndex } from "../lib/data";
import { SOURCE_FAMILY_LABELS } from "../lib/hotspotView";
import { bestPaperRoute } from "../lib/routes";
import type { MonthIndexPayload, RootIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: MonthIndexPayload; rootIndex: RootIndexPayload };

export function MonthArchivePage({ month }: { month: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    Promise.all([loadMonthIndex(month), loadRootIndex()])
      .then(([payload, rootIndex]) => {
        if (active) {
          setState({ status: "ready", payload, rootIndex });
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
  }, [month]);

  if (state.status === "loading") {
    return <section className="panel">Loading month archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load month archive: {state.message}</section>;
  }

  const { payload, rootIndex } = state;
  const sourceMix = Object.entries(payload.source_section_totals)
    .sort((left, right) => right[1] - left[1])
    .filter(([, count]) => count > 0);
  const monthIndex = rootIndex.months.findIndex((entry) => entry.month === payload.month);
  const previousMonth = monthIndex > 0 ? rootIndex.months[monthIndex - 1] : null;
  const nextMonth = monthIndex >= 0 && monthIndex < rootIndex.months.length - 1 ? rootIndex.months[monthIndex + 1] : null;
  const paperHref = bestPaperRoute(payload.paper_routes, ["month", "year", "home"]);

  return (
    <div className="stack hotspot-stack">
      <section className="archive-head">
        <ArchiveNav
          previous={
            previousMonth
              ? {
                  href: `/hot/${previousMonth.month}`,
                  title: "Previous Month",
                  subtitle: previousMonth.month,
                  arrow: "left",
                }
              : null
          }
          center={{
            href: `/hot/${payload.year}`,
            title: "Yearly Overview",
            subtitle: payload.year,
          }}
          next={
            nextMonth
              ? {
                  href: `/hot/${nextMonth.month}`,
                  title: "Next Month",
                  subtitle: nextMonth.month,
                  arrow: "right",
                }
              : null
          }
        />
        <CrossSiteSwitch href={paperHref} label="Personalized Daily Arxiv Paper" />
        <div className="archive-head-meta">
          <span>{payload.month}</span>
          <span>{payload.totals.days} days</span>
          <span>{payload.totals.source_items} items</span>
          <span>{payload.totals.featured_topics} featured</span>
        </div>
        <div className="section-chip-row compact">
          {sourceMix.map(([slug, count]) => (
            <span className="section-chip" key={slug}>
              <span>{SOURCE_FAMILY_LABELS[slug] ?? slug} {count}</span>
            </span>
          ))}
        </div>
      </section>

      <section className="panel compact-panel">
        <div className="section-header">
          <h2>Daily archive</h2>
        </div>
        <div className="archive-row-list">
          {payload.days.map((day) => (
            <Link className="archive-row-link" key={day.date} to={`/hot/${day.date}`}>
              <div className="archive-row-main">
                <div className="archive-row-header">
                  <strong className="archive-row-title">{day.date}</strong>
                  <span>{day.source_items} items</span>
                </div>
                <p className="archive-row-summary">{day.summary}</p>
                <div className="archive-row-meta">
                  <span>{day.featured_topics} featured</span>
                  <span>{day.topic_summary} topics</span>
                </div>
                <div className="section-chip-row compact">
                  {Object.entries(day.source_section_counts)
                    .filter(([, count]) => count > 0)
                    .slice(0, 4)
                    .map(([slug, count]) => (
                      <span className="section-chip" key={`${day.date}-${slug}`}>
                        <span>{SOURCE_FAMILY_LABELS[slug] ?? slug} {count}</span>
                      </span>
                    ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
