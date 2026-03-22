import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { loadMonthIndex } from "../lib/data";
import type { MonthIndexPayload } from "../types/hotspot";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: MonthIndexPayload };

export function MonthArchivePage({ month }: { month: string }) {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadMonthIndex(month)
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
  }, [month]);

  if (state.status === "loading") {
    return <section className="panel">Loading month archive...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load month archive: {state.message}</section>;
  }

  return (
    <section className="panel">
      <h1>{state.payload.month} hotspot archive</h1>
      <ul className="compact-list">
        {state.payload.days.map((day) => (
          <li key={day.date}>
            <Link to={`/hot/${day.date}`}>{day.date}</Link>
            <span>{day.featured_topics} featured</span>
            <span>{day.source_items} source items</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
