import { Link } from "react-router-dom";

export function CrossSiteSwitch({ href, label }: { href: string; label: string }) {
  return (
    <div className="cross-site-switch">
      <Link className="cross-site-switch-link" to={href}>
        {label}
      </Link>
    </div>
  );
}
