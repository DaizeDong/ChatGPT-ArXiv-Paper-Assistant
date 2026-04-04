import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type Lang = "en" | "zh";

type I18nContextType = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
  /** Return the _zh variant of a field value when in zh mode */
  tz: (original: string | undefined, zhVariant: string | undefined) => string;
};

const UI_LABELS: Record<string, Record<Lang, string>> = {
  "site.title": { en: "Daily AI Hotspots", zh: "每日 AI 热点" },
  "nav.prev": { en: "Previous Day", zh: "前一天" },
  "nav.next": { en: "Next Day", zh: "后一天" },
  "nav.monthly": { en: "Monthly Overview", zh: "月度概览" },
  "nav.paper": { en: "Personalized Daily Arxiv Paper", zh: "个性化每日论文" },
  "section.featured": { en: "Featured Stories", zh: "今日要闻" },
  "section.papers": { en: "Daily Hot Papers", zh: "今日热门论文" },
  "section.topics": { en: "All Topics", zh: "全部话题" },
  "section.feed": { en: "Source Feed", zh: "信息源动态" },
  "section.other": { en: "Other Updates", zh: "其他动态" },
  "section.usage": { en: "Daily Usage", zh: "每日用量" },
  "section.watchlist": { en: "Watchlist", zh: "关注列表" },
  "table.title": { en: "Title", zh: "标题" },
  "table.score": { en: "Score", zh: "评分" },
  "table.heat": { en: "Heat", zh: "热度" },
  "label.search": { en: "Search", zh: "搜索" },
  "label.sources": { en: "sources", zh: "来源" },
  "label.why": { en: "Why it matters:", zh: "为什么重要：" },
  "label.featured": { en: "featured", zh: "精选" },
  "label.categorized": { en: "categorized", zh: "分类" },
  "label.spotlight": { en: "spotlight", zh: "聚焦" },
  "label.sourceItems": { en: "source items", zh: "信息条目" },
  "label.otherItems": { en: "other items", zh: "其他条目" },
  "label.noMatch": { en: "No matching items", zh: "无匹配条目" },
  "label.noSignals": { en: "No matching signals", zh: "无匹配信号" },
  "usage.api": { en: "API", zh: "API" },
  "usage.mode": { en: "Mode", zh: "模式" },
  "usage.requests": { en: "Requests", zh: "请求数" },
  "usage.items": { en: "Items", zh: "条目数" },
  "usage.prompt": { en: "Prompt", zh: "提示词" },
  "usage.completion": { en: "Completion", zh: "补全" },
  "usage.cost": { en: "Cost", zh: "费用" },
  // Category names
  "cat.Product Release": { en: "Product Release", zh: "产品发布" },
  "cat.Market Signal": { en: "Market Signal", zh: "市场信号" },
  "cat.Industry Update": { en: "Industry Update", zh: "行业动态" },
  "cat.Tooling": { en: "Tooling", zh: "工具与平台" },
  "cat.Research": { en: "Research", zh: "研究论文" },
  "cat.Other": { en: "Other", zh: "其他" },
};

const I18nContext = createContext<I18nContextType | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    try {
      return (localStorage.getItem("hotspot-lang") as Lang) || "en";
    } catch {
      return "en";
    }
  });

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem("hotspot-lang", l);
    } catch { /* ignore */ }
  }, []);

  const t = useCallback(
    (key: string) => UI_LABELS[key]?.[lang] ?? UI_LABELS[key]?.en ?? key,
    [lang],
  );

  const tz = useCallback(
    (original: string | undefined, zhVariant: string | undefined) => {
      if (lang === "zh" && zhVariant) return zhVariant;
      return original ?? "";
    },
    [lang],
  );

  return <I18nContext.Provider value={{ lang, setLang, t, tz }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}

/** Translate a category name */
export function tCategory(category: string, lang: Lang): string {
  return UI_LABELS[`cat.${category}`]?.[lang] ?? category;
}
