export default function EmptyState({ title, body, action }) {
  return (
    <div className="border border-dashed border-rule bg-paper px-5 py-8">
      <p className="font-display text-base font-semibold text-ink">{title}</p>
      {body && <p className="mt-2 max-w-lg text-sm text-muted">{body}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
