import type { DailyHotspotPayload, MonthIndexPayload, RootIndexPayload, YearIndexPayload } from "../types/hotspot";

function assetPath(relativePath: string) {
  const basePath = import.meta.env.BASE_URL || "/";
  const cleanedBase = basePath.endsWith("/") ? basePath : `${basePath}/`;
  const cleanedRelative = relativePath.startsWith("/") ? relativePath.slice(1) : relativePath;
  return `${cleanedBase}${cleanedRelative}`;
}

async function fetchJson<T>(relativePath: string): Promise<T> {
  const response = await fetch(assetPath(relativePath));
  if (!response.ok) {
    throw new Error(`Failed to fetch ${relativePath}: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function loadRootIndex() {
  return fetchJson<RootIndexPayload>("web_data/hot/index.json");
}

export function loadDailyHotspot(date: string) {
  return fetchJson<DailyHotspotPayload>(`web_data/hot/${date}.json`);
}

export function loadMonthIndex(month: string) {
  return fetchJson<MonthIndexPayload>(`web_data/hot/${month}/index.json`);
}

export function loadYearIndex(year: string) {
  return fetchJson<YearIndexPayload>(`web_data/hot/${year}/index.json`);
}
