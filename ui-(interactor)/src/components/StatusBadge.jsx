export default function StatusBadge({ status }) {
  const normalized = String(status || 'UNKNOWN').toLowerCase();
  return <span className={`status status-${normalized}`}><i />{status || 'UNKNOWN'}</span>;
}
