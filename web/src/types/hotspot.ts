export type PaperRoutes = {
  home: string;
  day?: string | null;
  month?: string | null;
  year?: string | null;
};

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
  paper_routes?: PaperRoutes;
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
  spotlight_kinds?: string[];
  spotlight_primary_kind?: string;
  spotlight_primary_label?: string;
  spotlight_comment?: string;
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
  section_score?: number;
  items: SourceSectionItem[];
};

export type CompactTopic = {
  topic_id: string;
  slug: string;
  headline: string;
  category: string;
  summary_short: string;
  why_it_matters: string;
  key_takeaways: string[];
  scores: {
    final: number;
    quality: number;
    heat: number;
    importance: number;
    occurrence: number;
    display_priority: number;
    confidence: number;
  };
  source_tier: string;
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
    paper_routes: PaperRoutes;
    counts: Record<string, number>;
  };
  totals: Record<string, number>;
  costs: Record<string, number>;
  usage?: {
    llm?: {
      provider?: string;
      billing_model?: string;
      screen_model?: string;
      summary_model?: string;
      requests: number;
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      prompt_cost: number;
      completion_cost: number;
      total_cost: number;
    };
    external?: Record<string, {
      provider: string;
      billing_model: string;
      requests: number;
      items: number;
      estimated_cost: number;
      cache_hit?: boolean;
    }>;
    summary?: {
      external_requests: number;
      x_requests: number;
      estimated_external_cost: number;
    };
  };
  source_stats: Record<string, number>;
  source_section_counts: Record<string, number>;
  source_sections: SourceSection[];
  paper_spotlight: SourceSection[];
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
  paper_routes: PaperRoutes;
  totals: {
    days: number;
    featured_topics: number;
    source_items: number;
    topic_summary: number;
  };
  source_section_totals: Record<string, number>;
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
  paper_routes: PaperRoutes;
  totals: {
    days: number;
    featured_topics: number;
    source_items: number;
    topic_summary: number;
  };
  source_section_totals: Record<string, number>;
  months: Array<{
    month: string;
    days: number;
    featured_topics: number;
    source_items: number;
    topic_summary: number;
    source_section_totals: Record<string, number>;
    route: string;
  }>;
};
