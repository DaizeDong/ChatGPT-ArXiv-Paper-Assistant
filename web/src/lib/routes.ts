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

export function withBasePath(href: string) {
  if (!href || /^[a-z]+:/i.test(href) || href.startsWith("//")) {
    return href;
  }
  const inferredBase =
    typeof window !== "undefined"
      ? (() => {
          const marker = "/hot";
          const pathname = window.location.pathname || "";
          const index = pathname.indexOf(marker);
          return index > 0 ? pathname.slice(0, index) : "";
        })()
      : "";
  const base = inferredBase || import.meta.env.BASE_URL || "/";
  if (!href.startsWith("/")) {
    return href;
  }
  if (base === "/") {
    return href;
  }
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;
  if (href === normalizedBase || href.startsWith(`${normalizedBase}/`)) {
    return href;
  }
  return `${normalizedBase}${href}`;
}
