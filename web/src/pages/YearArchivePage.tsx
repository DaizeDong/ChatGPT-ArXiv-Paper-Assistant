import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadYearIndex } from "../lib/data";
import type { YearIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: YearIndexPayload };

export function YearArchivePage({ year }: { year: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadYearIndex(year)
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
  }, [year]);

  if (state.status === "loading") {
    return <section className="panel">Loading year archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load year archive: {state.message}</section>;
  }

  return (
    <section className="panel">
      <h1>{state.payload.year} hotspot archive</h1>
      <ul className="compact-list">
        {state.payload.months.map((month) => (
          <li key={month.month}>
            <Link to={`/hot/${month.month}`}>{month.month}</Link>
            <span>{month.days} active days</span>
            <span>{month.source_items} source items</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
