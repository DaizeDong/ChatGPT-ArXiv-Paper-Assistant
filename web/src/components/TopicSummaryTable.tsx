import { Link } from "react-router-dom";

import type { DailyHotspotPayload } from "../types/hotspot";
import { SortableTable, type TableColumn } from "./SortableTable";

type TopicRow = DailyHotspotPayload["topic_summary"][number];

function statusLabel(status: string) {
  const normalized = status.replaceAll("_", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function emptyCell() {
  return <span className="cell-empty" />;
}

const columns: Array<TableColumn<TopicRow>> = [
  {
    key: "topic",
    label: "Topic",
    className: "col-title",
    defaultDirection: "asc",
    sortValue: (topic) => topic.headline,
    render: (topic) => (
      <Link className="table-link title-link" to={topic.route}>
        {topic.headline}
      </Link>
    ),
  },
  {
    key: "category",
    label: "Category",
    className: "col-topic",
    defaultDirection: "asc",
    sortValue: (topic) => topic.category,
    render: (topic) => <span>{topic.category || emptyCell()}</span>,
  },
  {
    key: "sources",
    label: "Sources",
    className: "col-source",
    align: "right",
    defaultDirection: "desc",
    sortValue: (topic) => topic.source_count,
    render: (topic) => <span className="number-cell">{topic.source_count}</span>,
  },
  {
    key: "signal",
    label: "Signal",
    className: "col-signal",
    defaultDirection: "asc",
    sortValue: (topic) => statusLabel(topic.llm_status),
    render: (topic) => <span>{statusLabel(topic.llm_status)}</span>,
  },
  {
    key: "score",
    label: "Score",
    className: "col-score",
    align: "right",
    defaultDirection: "desc",
    sortValue: (topic) => topic.display_priority || topic.final_score,
    render: (topic) => <span className="number-cell">{topic.final_score.toFixed(1)}</span>,
  },
  {
    key: "heat",
    label: "Heat",
    className: "col-heat",
    align: "right",
    defaultDirection: "desc",
    sortValue: (topic) => topic.heat,
    render: (topic) => <span className="number-cell">{topic.heat.toFixed(1)}</span>,
  },
];

export function TopicSummaryTable({ topics }: { topics: TopicRow[] }) {
  return <SortableTable columns={columns} rows={topics} rowKey={(topic) => topic.topic_id} defaultSortKey="score" emptyText="No matching topics" />;
}
