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

