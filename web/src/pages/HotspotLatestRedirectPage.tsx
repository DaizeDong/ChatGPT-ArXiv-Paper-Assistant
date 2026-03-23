import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";

import { loadRootIndex } from "../lib/data";

type AsyncState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; latestDate: string | null };

export function HotspotLatestRedirectPage() {
  const [state, setState] = useState<AsyncState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadRootIndex()
      .then((payload) => {
        if (active) {
          setState({ status: "ready", latestDate: payload.latest_date });
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
  }, []);

  if (state.status === "loading") {
    return <section className="panel">Loading hotspot index...</section>;
  }
  if (state.status === "error") {
    return <section className="panel">Failed to load hotspot index: {state.message}</section>;
  }
  if (!state.latestDate) {
    return <section className="panel">No hotspot data has been generated yet.</section>;
  }
  return <Navigate replace to={`/hot/${state.latestDate}`} />;
}
