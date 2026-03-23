import { SortableTable, type TableColumn } from "./SortableTable";

export type TopicSummaryRow = {
  topic_id: string;
  headline: string;
  final_score: number;
  heat: number;
  display_priority?: number;
  source_count: number;
  item_count?: number;
};

const columns: Array<TableColumn<TopicSummaryRow>> = [
  {
    key: "topic",
    label: "Topic",
    className: "col-title",
    defaultDirection: "asc",
    sortValue: (topic) => topic.headline,
    render: (topic) => <span className="title-link">{topic.headline}</span>,
  },
  {
    key: "sources",
    label: "Sources",
    className: "col-source",
    align: "center",
    defaultDirection: "desc",
    sortValue: (topic) => topic.source_count,
    compare: (left, right, direction) => {
      const factor = direction === "asc" ? 1 : -1;
      if (left.source_count !== right.source_count) {
        return factor * (left.source_count - right.source_count);
      }
      if ((left.display_priority || left.final_score) !== (right.display_priority || right.final_score)) {
        return factor * ((left.display_priority || left.final_score) - (right.display_priority || right.final_score));
      }
      return factor * (left.heat - right.heat);
    },
    render: (topic) => <span className="number-cell">{topic.source_count}</span>,
  },
  {
    key: "score",
    label: "Score",
    className: "col-score",
    align: "center",
    defaultDirection: "desc",
    sortValue: (topic) => topic.display_priority || topic.final_score,
    render: (topic) => <span className="number-cell">{topic.final_score.toFixed(1)}</span>,
  },
  {
    key: "heat",
    label: "Heat",
    className: "col-heat",
    align: "center",
    defaultDirection: "desc",
    sortValue: (topic) => topic.heat,
    render: (topic) => <span className="number-cell">{topic.heat.toFixed(1)}</span>,
  },
];

export function TopicSummaryTable({ topics }: { topics: TopicSummaryRow[] }) {
  return <SortableTable columns={columns} rows={topics} rowKey={(topic) => topic.topic_id} defaultSortKey="sources" emptyText="No matching topics" />;
}
