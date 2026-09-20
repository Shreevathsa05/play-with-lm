import { useState } from 'react';
import AuditReportModal from '../components/AuditReportModal';
import ConfirmDialog from '../components/ConfirmDialog';
import DashboardShell from '../components/DashboardShell';
import ResourceTable from '../components/ResourceTable';
import StatusBadge from '../components/StatusBadge';
import { useAuth } from '../context/auth-context';
import { usePollingList } from '../hooks/usePollingList';
import { apiRequest } from '../lib/api';

const date = (value) => value ? new Date(value).toLocaleString() : '—';

export default function DataScoringPage({ admin = false }) {
  const { token } = useAuth();
  const { items: datasets, error, loading, refresh } = usePollingList('/api/datasets', token);
  const [file, setFile] = useState(null);
  const [hfImport, setHfImport] = useState({ id: '', config: '' });
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [auditId, setAuditId] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const run = async (key, action, success) => {
    setBusy(key); setNotice('');
    try { await action(); setNotice(success); await refresh({ silent: true }); }
    catch (requestError) { setNotice(requestError.message || 'Request failed'); }
    finally { setBusy(''); }
  };

  const upload = (event) => {
    event.preventDefault();
    const body = new FormData(); body.append('file', file);
    run('upload', () => apiRequest('/api/datasets/upload', { token, method: 'POST', body }), 'Dataset queued for scoring.');
  };

  const importDataset = (event) => {
    event.preventDefault();
    run('import', () => apiRequest('/api/datasets/huggingface', {
      token, method: 'POST', body: JSON.stringify({ huggingFaceId: hfImport.id.trim(), huggingFaceConfig: hfImport.config.trim() || undefined }),
    }), 'Hugging Face dataset queued for scoring.');
  };

  const remove = () => run('delete', async () => {
    await apiRequest(`/api/datasets/${deleteTarget.id}`, { token, method: 'DELETE' });
    setDeleteTarget(null);
  }, 'Dataset deleted.');

  const columns = [
    { key: 'dataset', label: 'Dataset', render: (row) => <div className="table-primary"><strong>{row.filename}</strong><span>#{row.id} · {row.source || 'MINIO'}</span></div> },
    ...(admin ? [{ key: 'owner', label: 'Owner', render: (row) => row.user?.email || '—' }] : []),
    { key: 'status', label: 'Preparation status', render: (row) => <StatusBadge status={row.status} /> },
    { key: 'added', label: 'Added', render: (row) => date(row.uploadedAt) },
    { key: 'actions', label: '', render: (row) => <div className="row-actions">{['COMPLETED', 'FAILED'].includes(row.status) && <button className="text-button" onClick={() => setAuditId(row.id)}>View report</button>}<button className="text-button danger-link" onClick={() => setDeleteTarget(row)}>Delete</button></div> },
  ];

  const complete = datasets.filter((item) => item.status === 'COMPLETED').length;
  const pending = datasets.filter((item) => !['COMPLETED', 'FAILED'].includes(item.status)).length;

  return (
    <DashboardShell section="data" eyebrow="DATA PREPARATION" title="Know the data before it touches a model." description="Ingest, normalize, and assess each dataset for the training objectives it can actually support.">
      {(notice || error) && <div className={`notice ${(error || notice.toLowerCase().includes('failed')) ? 'notice-error' : ''}`}>{error || notice}</div>}
      <section className="metric-row"><div><span>Total datasets</span><strong>{datasets.length}</strong></div><div><span>Ready</span><strong>{complete}</strong></div><div><span>In review</span><strong>{pending}</strong></div><div><span>Refresh</span><strong>{loading ? 'Syncing' : '5 sec'}</strong></div></section>

      {!admin && <section className="workspace-section bone-panel">
        <div className="section-heading"><div><p className="eyebrow">INGEST</p><h2>Bring in source material</h2></div><p>Uploads and Hub datasets follow the same standardization and scoring pipeline.</p></div>
        <div className="ingest-grid">
          <form className="editorial-card" onSubmit={upload}><span className="card-index">01</span><h3>Upload a file</h3><p>JSON, JSONL, CSV, or plain text.</p><label className="file-control"><input type="file" accept=".json,.jsonl,.csv,.txt" onChange={(event) => setFile(event.target.files?.[0] || null)} required /><span>{file?.name || 'Choose a local file'}</span><small>{file ? `${Math.ceil(file.size / 1024)} KB` : 'Select from this computer'}</small></label><button className="button button-primary" disabled={!file || busy === 'upload'}>{busy === 'upload' ? 'Queueing…' : 'Upload and score'}</button></form>
          <form className="editorial-card" onSubmit={importDataset}><span className="card-index">02</span><h3>Import from the Hub</h3><p>Reference a public Hugging Face dataset and optional subset.</p><label>Dataset repository<input value={hfImport.id} onChange={(event) => setHfImport({ ...hfImport, id: event.target.value })} placeholder="HuggingFaceTB/smoltalk2" required /></label><label>Subset / configuration<input value={hfImport.config} onChange={(event) => setHfImport({ ...hfImport, config: event.target.value })} placeholder="SFT or SFT/OpenHermes_2.5_no_think" /></label><button className="button button-ghost" disabled={busy === 'import'}>{busy === 'import' ? 'Queueing…' : 'Import and score'}</button></form>
        </div>
      </section>}

      <section className="workspace-section"><div className="section-heading"><div><p className="eyebrow">DATASET ARCHIVE</p><h2>{admin ? 'All prepared datasets' : 'Your prepared datasets'}</h2></div><p>{loading ? 'Refreshing records…' : 'Open a completed record for objective readiness, findings, and measured properties.'}</p></div><ResourceTable columns={columns} rows={datasets} rowKey={(row) => row.id} emptyMessage="No datasets yet. Add one to begin." /></section>
      {auditId && <AuditReportModal datasetId={auditId} onClose={() => setAuditId(null)} token={token} />}
      {deleteTarget && <ConfirmDialog title={`Delete “${deleteTarget.filename}”?`} description={['COMPLETED', 'FAILED'].includes(deleteTarget.status) ? 'This removes the scoring record and its uploaded MinIO source. Any jobs using it must be deleted first.' : 'This cancels an in-progress score and removes the record. If a processor finishes later, its webhook is ignored.'} busy={busy === 'delete'} onCancel={() => setDeleteTarget(null)} onConfirm={remove} />}
    </DashboardShell>
  );
}
