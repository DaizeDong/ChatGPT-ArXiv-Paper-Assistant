import { Link } from "react-router-dom";
import type { PropsWithChildren } from "react";

export function PageShell({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI research radar</p>
          <Link className="brand" to="/">
            Daily AI Hotspots
          </Link>
        </div>
        <nav className="topnav">
          <Link to="/hot">Hotspots</Link>
          <a href="/archive/">Paper Digest</a>
        </nav>
      </header>
      <main className="page-content">{children}</main>
    </div>
  );
}
