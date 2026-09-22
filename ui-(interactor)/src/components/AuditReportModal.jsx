import { useEffect, useState } from 'react';
import { apiRequest } from '../lib/api';

export default function AuditReportModal({ datasetId, onClose, token }) {
  const [state, setState] = useState({ loading: true, report: null, error: '' });
  useEffect(() => {
    let active = true;
    apiRequest(`/api/datasets/${datasetId}/audit`, { token })
      .then((value) => {
        if (!active) return;
        let report = value;
        if (typeof value === 'string') { try { report = JSON.parse(value); } catch { report = { raw: value }; } }
        setState({ loading: false, report, error: '' });
      })
      .catch((error) => active && setState({ loading: false, report: null, error: error.message }));
    return () => { active = false; };
  }, [datasetId, token]);

  const { loading, report, error } = state;
  const metrics = report ? [
    ['Format', report.schema_type || 'Unknown'], ['Records', report.total_records ?? 0],
    ['Invalid rows', report.inconsistent_rows ?? 0], ['Missing values', `${report.missing_percentage ?? 0}%`],
    ['Duplicates', `${report.duplicate_percentage ?? 0}%`], ['Estimated tokens', report.estimated_total_tokens ?? 0],
    ['Mean length', `${report.char_length_mean ?? 0} chars`], ['P95 length', `${report.char_length_p95 ?? 0} chars`],
    ['Emails found', report.pii_emails_detected ?? 0], ['IP addresses found', report.pii_ips_detected ?? 0],
  ] : [];

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="audit-title" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <section className="modal-card">
        <header className="modal-header"><div><p className="eyebrow">DATASET #{datasetId}</p><h2 id="audit-title">Audit report</h2></div><button className="modal-close" onClick={onClose} aria-label="Close">×</button></header>
        <div className="modal-body">
          {loading && <p className="empty-state">Loading audit…</p>}
          {(error || report?.error) && <div className="notice notice-error report-notice"><b>Audit detail</b><span>{error || report.error}</span></div>}
          {report && !report.error && (
            <>
              <section className="health-panel">
                <div>
                  <span className="eyebrow">Detected schema</span>
                  <strong className="schema-badge">{report.schema_type || 'Unknown'}</strong>
                </div>
                <div>
                  <p className="eyebrow">OBJECTIVE READINESS</p>
                  <h3>Dataset preparation report</h3>
                  <p>Readiness is determined separately for each training objective. There is no universal quality score.</p>
                </div>
              </section>
              {!!report.findings?.length && (
                <section className="report-section">
                  <p className="eyebrow">FINDINGS</p>
                  <h3>Review items</h3>
                  <ul className="finding-list">
                    {report.findings.map((item, index) => (
                      <li key={index}><b>{item.severity}</b> · {item.message}</li>
                    ))}
                  </ul>
                </section>
              )}
              <section className="report-section">
                <p className="eyebrow">READINESS</p>
                <dl className="audit-grid">
                  {Object.entries(report.readiness || {}).map(([objective, value]) => (
                    <div key={objective}><dt>{objective}</dt><dd>{value.state}</dd></div>
                  ))}
                </dl>
              </section>
              <section className="report-section">
                <p className="eyebrow">METRICS</p>
                <h3>Dataset profile</h3>
                <dl className="audit-grid">
                  {metrics.map(([label, value]) => (
                    <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
                  ))}
                </dl>
              </section>
            </>
          )}
          {!loading && !report && !error && <p className="empty-state">No audit is available.</p>}
        </div>
        <footer className="modal-footer"><button className="button button-secondary" onClick={onClose}>Close report</button></footer>
      </section>
    </div>
  );
}
