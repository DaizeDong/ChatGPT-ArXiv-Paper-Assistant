import type { SourceSection, SourceSectionItem } from "../types/hotspot";

export type DensityMode = "compact" | "comfortable";
export const SOURCE_FAMILY_LABELS: Record<string, string> = {
  "x-buzz": "X / Buzz",
  blogs: "Blogs / Newsletters",
  official: "Official Updates",
  papers: "Papers",
  github: "GitHub / Tools",
  discussions: "Discussions",
};

export function compactNumber(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
  }
  return `${value}`;
}

export function itemSignals(item: SourceSectionItem) {
  const signals: string[] = [];
  if (item.signals.activity > 0) {
    signals.push(`activity ${compactNumber(item.signals.activity)}`);
  }
  if (item.signals.github_stars > 0) {
    signals.push(`stars ${compactNumber(item.signals.github_stars)}`);
  }
  if (item.signals.hn_score > 0) {
    signals.push(`HN ${compactNumber(item.signals.hn_score)}`);
  }
  if (item.signals.upvotes > 0) {
    signals.push(`upvotes ${compactNumber(item.signals.upvotes)}`);
  }
  if (item.signals.daily_score > 0) {
    signals.push(`daily ${compactNumber(item.signals.daily_score)}`);
  }
  return signals.slice(0, 3);
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
    item.tags.join(" "),
    item.topic_refs.map((topic) => topic.headline).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

export function filterSectionsBySearch(sections: SourceSection[], searchQuery: string) {
  const query = searchQuery.trim().toLowerCase();
  return sections
    .map((section) => {
      const filteredItems = section.items.filter((item) => matchesSearch(item, query));
      return { ...section, filteredItems };
    })
    .filter((section) => section.filteredItems.length > 0);
}

export function defaultVisibleCount(section: SourceSection, density: DensityMode) {
  const boosted = section.slug === "x-buzz" || section.slug === "blogs" ? 12 : 10;
  return density === "compact" ? boosted : Math.max(8, boosted - 2);
}
