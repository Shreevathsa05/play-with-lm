import { useEffect, useRef, useState } from 'react';
import { apiRequest } from '../lib/api';
import { formatJobLogEvent, getHfPushState, getJobEvents, getJobPhase, parseJobReport } from '../lib/hfExport';
import StatusBadge from './StatusBadge';

const ACTIVE_STATUSES = new Set(['CHECKING', 'WAITING', 'RUNNING', 'PENDING', 'QUEUED']);

export default function JobReportModal({ jobUuid, onClose, token }) {
  const [state, setState] = useState({ loading: true, job: null, report: null, error: '' });
  const logEndRef = useRef(null);

  useEffect(() => {
    let active = true;
    let timer;

    const refresh = async ({ silent = false } = {}) => {
      if (!silent) setState((current) => ({ ...current, loading: current.job == null, error: '' }));
      try {
        const job = await apiRequest(`/api/jobs/${jobUuid}`, { token });
        if (!active) return;
        const report = parseJobReport(job);
        setState({ loading: false, job, report, error: '' });
        const status = String(job.status || '').toUpperCase();
        if (ACTIVE_STATUSES.has(status)) {
          timer = window.setTimeout(() => refresh({ silent: true }), 3000);
        }
      } catch (error) {
        if (active) setState({ loading: false, job: null, report: null, error: error.message });
      }
    };

    refresh();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobUuid, token]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.report?.events?.length]);

  const { loading, job, report, error } = state;
  const evaluation = report?.eval || report?.runner?.eval || report;
  const nll = evaluation?.reference_nll;
  const accuracy = evaluation?.task_accuracy;
  const samples = evaluation?.samples || [];
  const exportInfo = report?.export || report?.runner?.export;
  const hfPush = getHfPushState(job, report);
  const stderrTail = report?.stderr_tail || report?.runner?.stderr_tail;
  const reason = report?.reason || report?.error || (!stderrTail && job?.errorMessage) || error;
  const phase = getJobPhase(job, report);
  const events = getJobEvents(report);
  const isActive = job && ACTIVE_STATUSES.has(String(job.status || '').toUpperCase());

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="job-report-title" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <section className="modal-card modal-card-wide">
        <header className="modal-header"><div><p className="eyebrow">JOB REPORT</p><h2 id="job-report-title">Fine-tune run</h2><p className="mono">{jobUuid}</p>{phase && <p className="job-phase-line">{phase}</p>}</div><button className="modal-close" onClick={onClose} aria-label="Close">×</button></header>
        <div className="modal-body">
          {loading && <p className="empty-state">Loading report…</p>}
          {!loading && job && <div className="report-summary"><div><span>Model</span><strong>{job.baseModelId}</strong></div><div><span>Recipe</span><strong>{job.recipe?.replaceAll('_', ' ')}</strong></div><div><span>Status</span><StatusBadge status={job.status} /></div><div><span>Wait time</span><strong>{report?.waited_sec ?? (isActive ? '…' : 0)}s</strong></div></div>}
          <section className="report-section job-log-section">
            <p className="eyebrow">LIVE LOG</p>
            <h3>{isActive ? 'Training progress' : 'Run log'}</h3>
            {isActive && <p className="muted job-log-hint">Refreshing every 3 seconds while the worker is active.</p>}
            {events.length > 0 ? (
              <div className="job-log-panel" aria-live="polite">
                {events.map((event, index) => (
                  <div key={`${event.ts || 'log'}-${event.event || 'event'}-${index}`} className={`job-log-line job-log-${event.event || 'log'}`}>
                    {formatJobLogEvent(event)}
                  </div>
                ))}
                <div ref={logEndRef} />
              </div>
            ) : (
              <p className="empty-state">{isActive ? 'Waiting for worker log events…' : 'No worker log events were captured for this run.'}</p>
            )}
          </section>
          {reason && <div className="notice notice-error report-notice"><b>Worker detail</b><span>{reason}</span></div>}
          {stderrTail && <section className="report-section"><p className="eyebrow">FAILURE LOG</p><h3>Worker traceback</h3><pre className="raw-report">{stderrTail}</pre></section>}
          {(exportInfo || hfPush) && <section className="report-section"><p className="eyebrow">EXPORT</p><h3>Model artifact</h3><dl><div><dt>MinIO</dt><dd className="mono">{exportInfo?.export_dest || '—'}</dd></div><div><dt>Hugging Face</dt><dd className="export-status"><StatusBadge status={hfPush?.status || 'Not requested'} />{hfPush?.status === 'published' && hfPush.url ? <a href={hfPush.url} target="_blank" rel="noreferrer">{hfPush.repo_id}</a> : hfPush?.repo_id ? <span className="mono">{hfPush.repo_id}</span> : null}</dd></div>{hfPush?.error && <div><dt>Push error</dt><dd>{hfPush.error}</dd></div>}</dl></section>}
          {nll && <section className="report-section"><p className="eyebrow">REFERENCE NLL</p><h3>Before and after</h3><div className="score-grid"><div><span>Base loss</span><strong>{nll.mean_loss_base ?? '—'}</strong></div><div><span>Fine-tuned loss</span><strong>{nll.mean_loss_ft ?? '—'}</strong></div><div><span>Improvement</span><strong>{nll.mean_improvement == null ? '—' : `${(nll.mean_improvement * 100).toFixed(1)}%`}</strong></div></div></section>}
          {accuracy && <section className="report-section"><p className="eyebrow">TASK ACCURACY</p><h3>Exact match</h3><div className="score-grid"><div><span>Base</span><strong>{accuracy.base ?? '—'}</strong></div><div><span>Fine-tuned</span><strong>{accuracy.ft ?? '—'}</strong></div><div><span>Delta</span><strong>{accuracy.delta ?? '—'}</strong></div></div></section>}
          {!!samples.length && <section className="report-section"><p className="eyebrow">SAMPLES</p><h3>Side-by-side generations</h3><div className="sample-list">{samples.map((sample, index) => <article key={`${sample.prompt}-${index}`}><span>Prompt</span><p>{sample.prompt}</p><div><section><span>Base</span><pre>{sample.base_generation}</pre></section><section><span>Fine-tuned</span><pre>{sample.ft_generation}</pre></section></div></article>)}</div></section>}
          {!loading && !report && !reason && !events.length && <p className="empty-state">No report payload is available yet.</p>}
          {report?.raw && <pre className="raw-report">{report.raw}</pre>}
        </div>
        <footer className="modal-footer"><button className="button button-secondary" onClick={onClose}>Close report</button></footer>
      </section>
    </div>
  );
}
