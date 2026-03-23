import { Route, Routes } from "react-router-dom";

import { PageShell } from "./components/PageShell";
import { HotspotLatestRedirectPage } from "./pages/HotspotLatestRedirectPage";
import { HotspotRoutePage } from "./pages/HotspotRoutePage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function App() {
  return (
    <PageShell>
      <Routes>
        <Route path="/" element={<HotspotLatestRedirectPage />} />
        <Route path="/hot" element={<HotspotLatestRedirectPage />} />
        <Route path="/hot/:hotspotKey" element={<HotspotRoutePage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </PageShell>
  );
}
