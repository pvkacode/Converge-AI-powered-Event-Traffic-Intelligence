export default function Loading() {
  return (
    <div className="page-loading" aria-live="polite" aria-busy="true">
      <div className="page-loading-bar" />
      <p className="page-loading-text">Loading pipeline view…</p>
    </div>
  );
}
