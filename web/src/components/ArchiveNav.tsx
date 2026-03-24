import { Link } from "react-router-dom";

type NavEntry = {
  href?: string | null;
  title: string;
  subtitle: string;
  arrow?: "left" | "right";
};

function NavPlaceholder({ align }: { align: "left" | "center" | "right" }) {
  return (
    <div className={`archive-nav-slot ${align}`} aria-hidden="true">
      <span className="archive-nav-placeholder" />
    </div>
  );
}

function NavEdge({ align, entry }: { align: "left" | "right"; entry?: NavEntry | null }) {
  if (!entry?.href) {
    return <NavPlaceholder align={align} />;
  }
  return (
    <div className={`archive-nav-slot ${align}`}>
      <Link className={`archive-nav-link ${align}`} to={entry.href}>
        {entry.arrow === "left" ? <span className="archive-nav-arrow">{"\u2190"}</span> : null}
        <span className="archive-nav-copy">
          <span className="archive-nav-title">{entry.title}</span>
          <span className="archive-nav-subtitle">{entry.subtitle}</span>
        </span>
        {entry.arrow === "right" ? <span className="archive-nav-arrow">{"\u2192"}</span> : null}
      </Link>
    </div>
  );
}

function NavCenter({ entry }: { entry: NavEntry }) {
  return (
    <div className="archive-nav-slot center">
      {entry.href ? (
        <Link className="archive-nav-link center" to={entry.href}>
          <span className="archive-nav-copy">
            <span className="archive-nav-title">{entry.title}</span>
            <span className="archive-nav-subtitle">{entry.subtitle}</span>
          </span>
        </Link>
      ) : (
        <span className="archive-nav-link center static">
          <span className="archive-nav-copy">
            <span className="archive-nav-title">{entry.title}</span>
            <span className="archive-nav-subtitle">{entry.subtitle}</span>
          </span>
        </span>
      )}
    </div>
  );
}

export function ArchiveNav({
  previous,
  center,
  next,
}: {
  previous?: NavEntry | null;
  center: NavEntry;
  next?: NavEntry | null;
}) {
  return (
    <nav className="archive-nav" aria-label="Archive navigation">
      <NavEdge align="left" entry={previous} />
      <NavCenter entry={center} />
      <NavEdge align="right" entry={next} />
    </nav>
  );
}
