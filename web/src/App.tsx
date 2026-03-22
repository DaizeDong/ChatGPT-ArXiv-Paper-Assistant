import { Route, Routes } from "react-router-dom";

import { PageShell } from "./components/PageShell";
import { HotspotHomePage } from "./pages/HotspotHomePage";
import { HotspotRoutePage } from "./pages/HotspotRoutePage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { SourceDetailPage } from "./pages/SourceDetailPage";
import { TopicDetailPage } from "./pages/TopicDetailPage";

export function App() {
  return (
    <PageShell>
      <Routes>
        <Route path="/" element={<HotspotHomePage />} />
        <Route path="/hot" element={<HotspotHomePage />} />
        <Route path="/hot/:hotspotKey" element={<HotspotRoutePage />} />
        <Route path="/hot/:date/source/:sourceSlug" element={<SourceDetailPage />} />
        <Route path="/hot/:date/topic/:topicSlug" element={<TopicDetailPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </PageShell>
  );
}
