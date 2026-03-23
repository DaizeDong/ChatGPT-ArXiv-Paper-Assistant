import { Link, useLocation, useSearchParams } from "react-router-dom";
import type { PropsWithChildren } from "react";
import { useEffect, useState } from "react";

const MODE_KEY = "arxiv_site_color_mode";

export function PageShell({ children }: PropsWithChildren) {
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const [colorMode, setColorMode] = useState<"light" | "dark">("light");
  const isDailyHotspotRoute = /^\/hot\/\d{4}-\d{2}-\d{2}\/?$/.test(location.pathname);
  const searchValue = searchParams.get("q") ?? "";

  useEffect(() => {
    let nextMode = localStorage.getItem(MODE_KEY) as "light" | "dark" | null;
    if (nextMode !== "light" && nextMode !== "dark") {
      nextMode = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    document.documentElement.setAttribute("data-color-mode", nextMode);
    setColorMode(nextMode);
  }, []);

  function toggleColorMode() {
    const nextMode = colorMode === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-color-mode", nextMode);
    localStorage.setItem(MODE_KEY, nextMode);
    setColorMode(nextMode);
  }

  return (
    <div className="app-shell">
      <button className="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode" onClick={toggleColorMode}>
        <span className="theme-toggle-track" />
        <span className="theme-toggle-thumb">{colorMode === "dark" ? "☀" : "◐"}</span>
      </button>
      <header className="topbar">
        <Link className="brand" to="/hot">
          Daily AI Hotspots
        </Link>
        <div className="topbar-right">
          {isDailyHotspotRoute ? (
            <label className="nav-search">
              <input
                type="search"
                value={searchValue}
                onChange={(event) => {
                  const next = new URLSearchParams(searchParams);
                  const value = event.target.value.trimStart();
                  if (value) {
                    next.set("q", value);
                  } else {
                    next.delete("q");
                  }
                  setSearchParams(next, { replace: true });
                }}
                placeholder="Search"
              />
            </label>
          ) : null}
        </div>
      </header>
      <main className="page-content">{children}</main>
    </div>
  );
}
