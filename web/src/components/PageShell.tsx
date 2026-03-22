import { Link } from "react-router-dom";
import type { PropsWithChildren } from "react";

function repoPath(relativePath: string) {
  const basePath = import.meta.env.BASE_URL || "/";
  const cleanedBase = basePath.endsWith("/") ? basePath : `${basePath}/`;
  const cleanedRelative = relativePath.startsWith("/") ? relativePath.slice(1) : relativePath;
  return `${cleanedBase}${cleanedRelative}`;
}

export function PageShell({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI research radar</p>
          <Link className="brand" to="/hot">
            Daily AI Hotspots
          </Link>
        </div>
        <nav className="topnav">
          <Link to="/hot">Hotspots</Link>
          <a href={repoPath("")}>Paper Digest</a>
        </nav>
      </header>
      <main className="page-content">{children}</main>
    </div>
  );
}
