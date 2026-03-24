export function CrossSiteSwitch({ href, label }: { href: string; label: string }) {
  return (
    <div className="cross-site-switch">
      <a className="cross-site-switch-link" href={href}>
        {label}
      </a>
    </div>
  );
}
