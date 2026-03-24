import { DailyHotspotPage } from "./DailyHotspotPage";
import { MonthArchivePage } from "./MonthArchivePage";
import { NotFoundPage } from "./NotFoundPage";
import { YearArchivePage } from "./YearArchivePage";
import { useParams } from "react-router-dom";

const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const MONTH_PATTERN = /^\d{4}-\d{2}$/;
const YEAR_PATTERN = /^\d{4}$/;

export function HotspotRoutePage() {
  const { hotspotKey = "" } = useParams();

  if (DATE_PATTERN.test(hotspotKey)) {
    return <DailyHotspotPage date={hotspotKey} />;
  }
  if (MONTH_PATTERN.test(hotspotKey)) {
    return <MonthArchivePage month={hotspotKey} />;
  }
  if (YEAR_PATTERN.test(hotspotKey)) {
    return <YearArchivePage year={hotspotKey} />;
  }
  return <NotFoundPage />;
}
