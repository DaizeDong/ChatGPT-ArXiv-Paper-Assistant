import { useI18n } from "../lib/i18n";

export function LangSwitch() {
  const { lang, setLang } = useI18n();
  return (
    <div className="toggle-group lang-switch">
      <button type="button" className={lang === "en" ? "active" : ""} onClick={() => setLang("en")}>
        EN
      </button>
      <button type="button" className={lang === "zh" ? "active" : ""} onClick={() => setLang("zh")}>
        中文
      </button>
    </div>
  );
}
