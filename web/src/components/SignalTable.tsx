import { useI18n } from "../lib/i18n";
import { displayTopicRef, formatSourceRole, itemHeat } from "../lib/hotspotView";
import type { SourceSectionItem } from "../types/hotspot";
import { SortableTable, type TableColumn } from "./SortableTable";

function emptyCell() {
  return <span className="cell-empty" />;
}

function summaryOverlaps(summary: string, comment: string): boolean {
  if (!summary || !comment) return false;
  if (summary === comment) return true;
  const normS = summary.replace(/\.{3}$/, "").replace(/…$/, "");
  const normC = comment.replace(/\.{3}$/, "").replace(/…$/, "");
  return normS.startsWith(normC) || normC.startsWith(normS);
}

export function SignalTable({ items }: { items: SourceSectionItem[] }) {
  const { t, tz } = useI18n();

  const columns: Array<TableColumn<SourceSectionItem>> = [
    {
      key: "title",
      label: t("table.title"),
      className: "col-title col-title-dense",
      defaultDirection: "asc",
      sortValue: (item) => item.title,
      render: (item) => {
        const any = item as Record<string, unknown>;
        const topic = displayTopicRef(item);
        const spotlightComment = (item.spotlight_comment ?? "").trim();
        const summary = (item.summary_short ?? "").trim();
        const showComment = !!spotlightComment;
        const showSummary = !!summary && summary !== item.title && !summaryOverlaps(summary, spotlightComment);
        return (
          <div className="title-cell">
            <a className="table-link title-link" href={item.url} target="_blank" rel="noreferrer">
              {tz(item.title, any.title_zh as string)}
            </a>
            {showComment ? <div className="title-subline">{tz(spotlightComment, any.spotlight_comment_zh as string)}</div> : null}
            {showSummary ? <div className="title-summary">{tz(summary, any.summary_short_zh as string)}</div> : null}
            <div className="title-tag-row">
              {topic ? <span className="title-tag topic-tag">{tz(topic.headline, (topic as Record<string, unknown>).headline_zh as string)}</span> : null}
              {item.spotlight_primary_label ? <span className="title-tag topic-tag">{item.spotlight_primary_label}</span> : null}
              <span className="title-tag">{item.source_name}</span>
              <span className="title-tag">{formatSourceRole(item.source_role)}</span>
            </div>
          </div>
        );
      },
    },
    {
      key: "score",
      label: t("table.score"),
      className: "col-score",
      align: "center",
      defaultDirection: "desc",
      sortValue: (item) => item.signal_score,
      render: (item) => <span className="number-cell">{item.signal_score.toFixed(1)}</span>,
    },
    {
      key: "heat",
      label: t("table.heat"),
      className: "col-heat",
      align: "center",
      defaultDirection: "desc",
      sortValue: (item) => itemHeat(item).value,
      render: (item) => {
        const heat = itemHeat(item);
        return heat.label ? <span className="heat-cell">{heat.label}</span> : emptyCell();
      },
    },
  ];

  return <SortableTable columns={columns} rows={items} rowKey={(item) => item.id} defaultSortKey="score" emptyText={t("label.noMatch")} />;
}
