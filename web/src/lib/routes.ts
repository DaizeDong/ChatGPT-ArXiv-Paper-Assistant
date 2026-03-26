import type { PaperRoutes } from "../types/hotspot";

export function bestPaperRoute(paperRoutes?: PaperRoutes | null, preferred: Array<keyof PaperRoutes> = ["day", "month", "year", "home"]) {
  if (!paperRoutes) {
    return "/";
  }
  for (const key of preferred) {
    const value = paperRoutes[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return "/";
}

function inferredBasePath() {
  if (typeof window === "undefined") {
    return "";
  }
  const marker = "/hot";
  const pathname = window.location.pathname || "";
  const index = pathname.indexOf(marker);
  return index > 0 ? pathname.slice(0, index) : "";
}

function resolveRelativeHref(href: string) {
  if (typeof window === "undefined") {
    return href;
  }
  const pathname = window.location.pathname || "/";
  const directoryPath = pathname.endsWith("/") ? pathname : `${pathname}/`;
  const resolved = new URL(href, `${window.location.origin}${directoryPath}`);
  return `${resolved.pathname}${resolved.search}${resolved.hash}`;
}

export function withBasePath(href: string) {
  if (!href || /^[a-z]+:/i.test(href) || href.startsWith("//")) {
    return href;
  }
  if (!href.startsWith("/")) {
    return resolveRelativeHref(href);
  }
  const base = inferredBasePath() || import.meta.env.BASE_URL || "/";
  if (base === "/") {
    return href;
  }
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;
  if (!normalizedBase) {
    return href;
  }
  if (href === normalizedBase || href.startsWith(`${normalizedBase}/`)) {
    return href;
  }
  return `${normalizedBase}${href}`;
}
