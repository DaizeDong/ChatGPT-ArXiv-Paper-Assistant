import { useParams } from "react-router-dom";

export function SourceDetailPage() {
  const { date, sourceSlug } = useParams();
  return (
    <section className="panel">
      <h1>Source detail</h1>
      <p>
        Phase 2 scaffold route for <code>{sourceSlug}</code> on <code>{date}</code>.
      </p>
      <p>The dense source-specific page lands in the next phase.</p>
    </section>
  );
}
