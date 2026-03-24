import type { PropsWithChildren } from "react";
import { useEffect, useState } from "react";

const MODE_KEY = "arxiv_site_color_mode";

export function PageShell({ children }: PropsWithChildren) {
  const [colorMode, setColorMode] = useState<"light" | "dark">("light");

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
        <span className="theme-toggle-thumb">{colorMode === "dark" ? "\u263E" : "\u2600"}</span>
      </button>
      <main className="page-content">{children}</main>
    </div>
  );
}
