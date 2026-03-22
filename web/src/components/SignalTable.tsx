import { Link } from "react-router-dom";

import { formatSourceRole, itemHeat, primaryTopicRef } from "../lib/hotspotView";
import type { SourceSectionItem } from "../types/hotspot";
import { SortableTable, type TableColumn } from "./SortableTable";

function emptyCell() {
  return <span className="cell-empty" />;
}

function normalize(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

const columns: Array<TableColumn<SourceSectionItem>> = [
  {
    key: "title",
    label: "Title",
    className: "col-title",
    defaultDirection: "asc",
    sortValue: (item) => item.title,
    render: (item) => (
      <a className="table-link title-link" href={item.url} target="_blank" rel="noreferrer">
        {item.title}
      </a>
    ),
  },
  {
    key: "topic",
    label: "Topic",
    className: "col-topic",
    defaultDirection: "asc",
    sortValue: (item) => {
      const topic = primaryTopicRef(item);
      if (!topic || normalize(topic.headline) === normalize(item.title)) {
        return "";
      }
      return topic.headline;
    },
    render: (item) => {
      const topic = primaryTopicRef(item);
      return topic && normalize(topic.headline) !== normalize(item.title) ? (
        <Link className="table-link topic-link" to={topic.daily_route}>
          {topic.headline}
        </Link>
      ) : (
        emptyCell()
      );
    },
  },
  {
    key: "source",
    label: "Source",
    className: "col-source",
    defaultDirection: "asc",
    sortValue: (item) => item.source_name,
    render: (item) => <span>{item.source_name}</span>,
  },
  {
    key: "signal",
    label: "Signal",
    className: "col-signal",
    defaultDirection: "asc",
    sortValue: (item) => formatSourceRole(item.source_role),
    render: (item) => <span>{formatSourceRole(item.source_role)}</span>,
  },
  {
    key: "score",
    label: "Score",
    className: "col-score",
    align: "right",
    defaultDirection: "desc",
    sortValue: (item) => item.signal_score,
    render: (item) => <span className="number-cell">{item.signal_score.toFixed(1)}</span>,
  },
  {
    key: "heat",
    label: "Heat",
    className: "col-heat",
    align: "right",
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
