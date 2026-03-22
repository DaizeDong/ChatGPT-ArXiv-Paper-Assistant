import { useParams } from "react-router-dom";

export function TopicDetailPage() {
  const { date, topicSlug } = useParams();
  return (
    <section className="panel">
      <h1>Topic detail</h1>
      <p>
        Phase 2 scaffold route for <code>{topicSlug}</code> on <code>{date}</code>.
      </p>
      <p>Detailed topic drill-down lands in a later phase.</p>
    </section>
  );
}
