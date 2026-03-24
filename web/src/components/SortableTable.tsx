import { useMemo, useState } from "react";
import type { ReactNode } from "react";

type SortDirection = "asc" | "desc";

export type TableColumn<T> = {
  key: string;
  label: string;
  render: (row: T) => ReactNode;
  sortValue: (row: T) => string | number | null | undefined;
  compare?: (left: T, right: T, direction: SortDirection) => number;
  defaultDirection?: SortDirection;
  align?: "left" | "right" | "center";
  className?: string;
};

function compareValues(left: string | number | null | undefined, right: string | number | null | undefined) {
  if (typeof left === "number" || typeof right === "number") {
    return Number(left ?? 0) - Number(right ?? 0);
  }
  return `${left ?? ""}`.localeCompare(`${right ?? ""}`, undefined, { sensitivity: "base" });
}

function sortIndicator(isActive: boolean, direction: SortDirection) {
  if (!isActive) {
    return "\u2195";
  }
  return direction === "desc" ? "\u2193" : "\u2191";
}

export function SortableTable<T>({
  columns,
  rows,
  rowKey,
  defaultSortKey,
  emptyText = "No rows",
}: {
  columns: Array<TableColumn<T>>;
  rows: T[];
  rowKey: (row: T) => string;
  defaultSortKey?: string;
  emptyText?: string;
}) {
  const initialColumn = columns.find((column) => column.key === defaultSortKey) ?? columns[0];
  const [sortKey, setSortKey] = useState(initialColumn.key);
  const [sortDirection, setSortDirection] = useState<SortDirection>(initialColumn.defaultDirection ?? "desc");

  const activeColumn = columns.find((column) => column.key === sortKey) ?? columns[0];
  const sortedRows = useMemo(() => {
    const copied = [...rows];
    copied.sort((left, right) => {
      const base = activeColumn.compare
        ? activeColumn.compare(left, right, sortDirection)
        : sortDirection === "asc"
          ? compareValues(activeColumn.sortValue(left), activeColumn.sortValue(right))
          : -compareValues(activeColumn.sortValue(left), activeColumn.sortValue(right));
      if (base === 0) {
        return compareValues(rowKey(left), rowKey(right));
      }
      return base;
    });
    return copied;
  }, [activeColumn, rowKey, rows, sortDirection]);

  if (!rows.length) {
    return <div className="table-empty">{emptyText}</div>;
  }

  return (
    <div className="table-wrap">
      <table className="signal-table">
        <thead>
          <tr>
            {columns.map((column) => {
              const isActive = column.key === sortKey;
              const nextDirection: SortDirection = isActive && sortDirection === "desc" ? "asc" : "desc";
              const alignClass = column.align ? `align-${column.align}` : "align-left";
              return (
                <th className={`${column.className ?? ""} ${alignClass}`.trim()} key={column.key}>
                  <button
                    className={`sort-button ${alignClass} ${isActive ? "active" : ""}`.trim()}
                    onClick={() => {
                      if (isActive) {
                        setSortDirection(nextDirection);
                      } else {
                        setSortKey(column.key);
                        setSortDirection(column.defaultDirection ?? "desc");
                      }
                    }}
                    type="button"
                  >
                    <span>{column.label}</span>
                    <span className="sort-indicator">{sortIndicator(isActive, sortDirection)}</span>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((column) => (
                <td className={`${column.className ?? ""} ${column.align ? `align-${column.align}` : ""}`.trim()} key={column.key}>
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
