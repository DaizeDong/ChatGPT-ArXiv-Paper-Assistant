export type RootIndexEntry = {
  date: string;
  month: string;
  year: string;
  summary: string;
  featured_topics: number;
  source_items: number;
  topic_summary: number;
  source_section_counts: Record<string, number>;
  route: string;
};

export type RootIndexPayload = {
  schema_version: number;
  latest_date: string | null;
  dates: RootIndexEntry[];
  months: Array<{ month: string; days: number; featured_topics: number; source_items: number; route?: string }>;
  years: Array<{ year: string; days: number; featured_topics: number; source_items: number; route?: string }>;
};

export type TopicRef = {
  topic_id: string;
  slug: string;
  headline: string;
  daily_route: string;
};

export type SourceSectionItem = {
  id: string;
  title: string;
  summary_short: string;
  url: string;
  canonical_url: string;
  source_id: string;
  source_name: string;
  source_role: string;
  source_type: string;
  source_family: string;
  published_at: string | null;
  tags: string[];
  authors: string[];
  signal_score: number;
  signals: {
    activity: number;
    github_stars: number;
    hn_score: number;
    upvotes: number;
    daily_score: number;
  };
  topic_refs: TopicRef[];
};

export type SourceSection = {
  slug: string;
  label: string;
  description: string;
  count: number;
  items: SourceSectionItem[];
};

export type CompactTopic = {
  topic_id: string;
  slug: string;
  headline: string;
  category: string;
  summary_short: string;
  why_it_matters: string;
  scores: {
    final: number;
    quality: number;
    heat: number;
    importance: number;
    occurrence: number;
    display_priority: number;
  };
  source_names: string[];
  source_roles: string[];
  source_types: string[];
  llm_status: string;
  evidence: Array<{ title: string; url: string; source_name: string }>;
  route: string;
};

export type DailyHotspotPayload = {
  schema_version: number;
  meta: {
    date: string;
    month: string;
    year: string;
    generated_at: string;
    mode: string;
    summary: string;
    previous_date: string | null;
    next_date: string | null;
    counts: Record<string, number>;
  };
  totals: Record<string, number>;
  costs: Record<string, number>;
  source_stats: Record<string, number>;
  source_section_counts: Record<string, number>;
  source_sections: SourceSection[];
  topic_summary: Array<{
    topic_id: string;
    slug: string;
    headline: string;
    category: string;
    final_score: number;
    heat: number;
    occurrence_score: number;
    display_priority: number;
    llm_status: string;
    source_count: number;
    item_count: number;
    route: string;
  }>;
  featured_topics: CompactTopic[];
  category_sections: Array<{ category: string; total_candidates: number; topics: CompactTopic[] }>;
  long_tail_sections: Array<{ category: string; total_candidates: number; topics: CompactTopic[] }>;
  watchlist: CompactTopic[];
  x_buzz: Array<Record<string, unknown>>;
};

export type MonthIndexPayload = {
  schema_version: number;
  month: string;
  year: string;
  days: Array<{
    date: string;
    summary: string;
    featured_topics: number;
    source_items: number;
    topic_summary: number;
    source_section_counts: Record<string, number>;
    route: string;
  }>;
};

export type YearIndexPayload = {
  schema_version: number;
  year: string;
  months: Array<{
    month: string;
    days: number;
    featured_topics: number;
    source_items: number;
    topic_summary: number;
    route: string;
  }>;
};
