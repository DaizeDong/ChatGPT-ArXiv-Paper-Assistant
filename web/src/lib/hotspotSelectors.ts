import type { CompactTopic, DailyHotspotPayload, SourceSection, SourceSectionItem } from "../types/hotspot";

export function flattenTopics(payload: DailyHotspotPayload): CompactTopic[] {
  const seen = new Set<string>();
  const ordered: CompactTopic[] = [];
  const sources = [
    ...payload.featured_topics,
    ...payload.category_sections.flatMap((section) => section.topics),
    ...payload.long_tail_sections.flatMap((section) => section.topics),
    ...payload.watchlist,
  ];
  for (const topic of sources) {
    if (!seen.has(topic.topic_id)) {
      seen.add(topic.topic_id);
      ordered.push(topic);
    }
  }
  return ordered;
}

export function findTopicBySlug(payload: DailyHotspotPayload, slug: string): CompactTopic | undefined {
  return flattenTopics(payload).find((topic) => topic.slug === slug);
}

export function findSourceSection(payload: DailyHotspotPayload, slug: string): SourceSection | undefined {
  return payload.source_sections.find((section) => section.slug === slug);
}

export function findSupportingItemsForTopic(payload: DailyHotspotPayload, topicId: string) {
  const groups: Array<{ section: SourceSection; items: SourceSectionItem[] }> = [];
  for (const section of payload.source_sections) {
    const items = section.items.filter((item) => item.topic_refs.some((topic) => topic.topic_id === topicId));
    if (items.length > 0) {
      groups.push({ section, items });
    }
  }
  return groups;
}

export function relatedTopicsByCategory(payload: DailyHotspotPayload, topic: CompactTopic, limit: number) {
  return flattenTopics(payload)
    .filter((candidate) => candidate.topic_id !== topic.topic_id && candidate.category === topic.category)
    .slice(0, limit);
}
