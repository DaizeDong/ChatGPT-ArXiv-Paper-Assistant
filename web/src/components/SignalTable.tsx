import { displayTopicRef, formatSourceRole, itemHeat } from "../lib/hotspotView";
import type { SourceSectionItem } from "../types/hotspot";
import { SortableTable, type TableColumn } from "./SortableTable";

function emptyCell() {
  return <span className="cell-empty" />;
}

function renderTitleCell(item: SourceSectionItem) {
  const topic = displayTopicRef(item);
  const spotlightComment = (item.spotlight_comment ?? "").trim();
  return (
    <div className="title-cell">
      <a className="table-link title-link" href={item.url} target="_blank" rel="noreferrer">
        {item.title}
      </a>
      {spotlightComment ? <div className="title-subline">{spotlightComment}</div> : null}
      <div className="title-tag-row">
        {topic ? <span className="title-tag topic-tag">{topic.headline}</span> : null}
        {item.spotlight_primary_label ? <span className="title-tag topic-tag">{item.spotlight_primary_label}</span> : null}
        <span className="title-tag">{item.source_name}</span>
        <span className="title-tag">{formatSourceRole(item.source_role)}</span>
      </div>
    </div>
  );
}

const columns: Array<TableColumn<SourceSectionItem>> = [
  {
    key: "title",
    label: "Title",
    className: "col-title col-title-dense",
    defaultDirection: "asc",
    sortValue: (item) => item.title,
    render: renderTitleCell,
  },
  {
    key: "score",
    label: "Score",
    className: "col-score",
    align: "center",
    defaultDirection: "desc",
    sortValue: (item) => item.signal_score,
    render: (item) => <span className="number-cell">{item.signal_score.toFixed(1)}</span>,
  },
  {
    key: "heat",
    label: "Heat",
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

export function SignalTable({ items }: { items: SourceSectionItem[] }) {
  return <SortableTable columns={columns} rows={items} rowKey={(item) => item.id} defaultSortKey="score" emptyText="No matching items" />;
}
