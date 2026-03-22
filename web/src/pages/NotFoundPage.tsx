import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <section className="panel">
      <h1>Not found</h1>
      <p>The requested hotspot route does not exist.</p>
      <Link className="primary-link" to="/hot">
        Return to hotspot home
      </Link>
    </section>
  );
}
