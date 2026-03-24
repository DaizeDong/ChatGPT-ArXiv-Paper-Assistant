import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ArchiveNav } from "../components/ArchiveNav";
import { CrossSiteSwitch } from "../components/CrossSiteSwitch";
import { loadRootIndex, loadYearIndex } from "../lib/data";
import { SOURCE_FAMILY_LABELS } from "../lib/hotspotView";
import { bestPaperRoute } from "../lib/routes";
import type { RootIndexPayload, YearIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: YearIndexPayload; rootIndex: RootIndexPayload };

export function YearArchivePage({ year }: { year: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    Promise.all([loadYearIndex(year), loadRootIndex()])
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
  }, [year]);

  if (state.status === "loading") {
    return <section className="panel">Loading year archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load year archive: {state.message}</section>;
  }

  const { payload, rootIndex } = state;
  const sourceMix = Object.entries(payload.source_section_totals)
    .sort((left, right) => right[1] - left[1])
    .filter(([, count]) => count > 0);
  const yearIndex = rootIndex.years.findIndex((entry) => entry.year === payload.year);
  const previousYear = yearIndex > 0 ? rootIndex.years[yearIndex - 1] : null;
  const nextYear = yearIndex >= 0 && yearIndex < rootIndex.years.length - 1 ? rootIndex.years[yearIndex + 1] : null;
  const paperHref = bestPaperRoute(payload.paper_routes, ["year", "home"]);

  return (
    <div className="stack hotspot-stack">
      <section className="archive-head">
        <ArchiveNav
          previous={
            previousYear
              ? {
                  href: `/hot/${previousYear.year}`,
                  title: "Previous Year",
                  subtitle: previousYear.year,
                  arrow: "left",
                }
              : null
          }
          center={{
            href: null,
            title: "Yearly Overview",
            subtitle: payload.year,
          }}
          next={
            nextYear
              ? {
                  href: `/hot/${nextYear.year}`,
                  title: "Next Year",
                  subtitle: nextYear.year,
                  arrow: "right",
                }
              : null
          }
        />
        <CrossSiteSwitch href={paperHref} label="Personalized Daily Arxiv Paper" />
        <div className="archive-head-meta">
          <span>{payload.year}</span>
          <span>{payload.months.length} months</span>
          <span>{payload.totals.days} days</span>
          <span>{payload.totals.source_items} items</span>
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
          <h2>Monthly archive</h2>
        </div>
        <div className="archive-row-list">
          {payload.months.map((monthRow) => (
            <Link className="archive-row-link" key={monthRow.month} to={`/hot/${monthRow.month}`}>
              <div className="archive-row-main">
                <div className="archive-row-header">
                  <strong className="archive-row-title">{monthRow.month}</strong>
                  <span>{monthRow.days} days</span>
                </div>
                <div className="archive-row-meta">
                  <span>{monthRow.source_items} items</span>
                  <span>{monthRow.featured_topics} featured</span>
                  <span>{monthRow.topic_summary} topics</span>
                </div>
                <div className="section-chip-row compact">
                  {Object.entries(monthRow.source_section_totals)
                    .filter(([, count]) => count > 0)
                    .slice(0, 4)
                    .map(([slug, count]) => (
                      <span className="section-chip" key={`${monthRow.month}-${slug}`}>
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
