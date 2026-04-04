import type { SourceSection, SourceSectionItem } from "../types/hotspot";

export type DensityMode = "compact" | "comfortable";
export const SOURCE_FAMILY_LABELS: Record<string, string> = {
  official: "Official Updates",
  "market-signals": "Market Signals",
  analysis: "Expert Analysis",
  papers: "Research Papers",
  github: "GitHub / Tools",
  industry: "Industry News",
};

const SOURCE_ROLE_LABELS: Record<string, string> = {
  builder_momentum: "Builder momentum",
  editorial_depth: "Editorial depth",
  headline_consensus: "Headline consensus",
  product_launch: "Product launch",
  paper_trending: "Paper trending",
  github_trend: "GitHub trend",
  hn_discussion: "HN discussion",
  community_heat: "Community heat",
  official_update: "Official update",
};

export function compactNumber(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
  }
  return `${value}`;
}

export function formatSourceRole(sourceRole: string) {
  return SOURCE_ROLE_LABELS[sourceRole] ?? sourceRole.replaceAll("_", " ");
}

export function primaryTopicRef(item: SourceSectionItem) {
  return item.topic_refs[0] ?? null;
}

function normalizeTopicValue(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

export function displayTopicRef(item: SourceSectionItem) {
  const topic = primaryTopicRef(item);
  if (!topic) {
    return null;
  }
  return normalizeTopicValue(topic.headline) === normalizeTopicValue(item.title) ? null : topic;
}

export function itemHeat(item: SourceSectionItem) {
  if (item.signals.activity > 0) {
    return { label: `activity ${compactNumber(item.signals.activity)}`, value: item.signals.activity };
  }
  if (item.signals.github_stars > 0) {
    return { label: `stars ${compactNumber(item.signals.github_stars)}`, value: item.signals.github_stars };
  }
  if (item.signals.hn_score > 0) {
    return { label: `HN ${compactNumber(item.signals.hn_score)}`, value: item.signals.hn_score };
  }
  if (item.signals.upvotes > 0) {
    return { label: `upvotes ${compactNumber(item.signals.upvotes)}`, value: item.signals.upvotes };
  }
  if (item.signals.daily_score > 0) {
    return { label: `daily ${compactNumber(item.signals.daily_score)}`, value: item.signals.daily_score };
  }
  return { label: "", value: 0 };
}

export function matchesSearch(item: SourceSectionItem, query: string) {
  if (!query) {
    return true;
  }
  const haystack = [
    item.title,
    item.summary_short,
    item.source_name,
    item.source_role,
    item.source_type,
    item.spotlight_primary_label ?? "",
    item.spotlight_comment ?? "",
    item.tags.join(" "),
    item.topic_refs.map((topic) => topic.headline).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

export function filterSectionsBySearch(sections: SourceSection[], searchQuery: string, excludeTopicCovered = false) {
  const query = searchQuery.trim().toLowerCase();
  return sections
    .map((section) => {
      let filteredItems = section.items.filter((item) => matchesSearch(item, query));
      if (excludeTopicCovered) {
        filteredItems = filteredItems.filter((item) => !item.topic_refs.length);
      }
      return { ...section, filteredItems };
    })
    .filter((section) => section.filteredItems.length > 0);
}

export function defaultVisibleCount(section: SourceSection, density: DensityMode) {
  const compactMap: Record<string, number> = {
    official: 10,
    "market-signals": 6,
    analysis: 8,
    papers: 12,
    github: 6,
    industry: 6,
  };
  const compactDefault = compactMap[section.slug] ?? 12;
  return density === "compact" ? compactDefault : Math.max(8, compactDefault - 4);
}
