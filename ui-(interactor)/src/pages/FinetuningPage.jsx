import { useState } from 'react';
import ConfirmDialog from '../components/ConfirmDialog';
import DashboardShell from '../components/DashboardShell';
import JobReportModal from '../components/JobReportModal';
import ModelPicker from '../components/ModelPicker';
import ResourceTable from '../components/ResourceTable';
import StatusBadge from '../components/StatusBadge';
import { useAuth } from '../context/auth-context';
import { usePollingList } from '../hooks/usePollingList';
import { apiRequest } from '../lib/api';
import { getHfPushState, parseJobReport, getJobPhase } from '../lib/hfExport';

const RECIPES = [
  { id: 'qlora_4bit', label: 'QLoRA 4-bit', note: 'Smallest VRAM footprint' },
  { id: 'qlora_8bit', label: 'QLoRA 8-bit', note: 'More base precision' },
  { id: 'lora', label: 'LoRA', note: '16-bit adapter training' },
  { id: 'fullparams', label: 'Full parameters', note: 'Train every weight' },
  { id: 'embedding', label: 'Embeddings', note: 'Pair or triplet contrastive' },
];
const OBJECTIVES = [
  ['sft', 'Supervised fine-tuning'], ['cpt', 'Continued pretraining'], ['pretrain', 'Train from scratch'],
  ['dpo', 'Direct preference optimization'], ['kto', 'Binary feedback alignment'], ['reward_model', 'Reward model'],
  ['ppo', 'PPO with reward model'], ['grpo', 'GRPO with verifier'], ['embedding', 'Embedding training'],
];

const initialJob = {
  datasetId: '', baseModelId: 'unsloth/Llama-3.2-1B-Instruct', recipe: 'qlora_4bit', objective: 'sft', rewardModelId: '',
  modelParamsB: '1b', exportHfRepo: '', evalPrompt: '', evalReference: '',
  dryRun: false, maxSeqLength: 512, batchSize: 1,
};

const date = (value) => value ? new Date(value).toLocaleString() : '—';
const ACTIVE_WORKER_STATUSES = ['CHECKING', 'WAITING', 'RUNNING', 'CANCEL_REQUESTED'];
const hasFreshHeartbeat = (row) => ACTIVE_WORKER_STATUSES.includes(String(row.status || '').toUpperCase())
  && Date.now() - new Date(row.updatedAt).getTime() < 60000;
const activity = (row) => {
  const status = String(row.status || '').toUpperCase();
  const phase = getJobPhase(row, parseJobReport(row));
  if (status === 'PENDING') return 'Saving request';
  if (status === 'QUEUED') return 'Waiting for a worker';
  if (ACTIVE_WORKER_STATUSES.includes(status) && !hasFreshHeartbeat(row)) return `Heartbeat stopped at ${date(row.updatedAt)} · worker may have stopped`;
  if (ACTIVE_WORKER_STATUSES.includes(status) && phase) return phase;
  if (status === 'CHECKING') return 'Worker connected · checking capacity';
  if (status === 'WAITING') return 'Worker alive · waiting for GPU/RAM';
  if (status === 'RUNNING') return `Training alive · heartbeat ${date(row.updatedAt)}`;
  if (status === 'COMPLETED') return phase || 'Training finished';
  if (status === 'FAILED') return phase || 'Training stopped';
  return status || 'Unknown';
};

export default function FinetuningPage({ admin = false }) {
  const { token } = useAuth();
  const jobsState = usePollingList('/api/jobs', token);
  const datasetsState = usePollingList('/api/datasets', token, 10000);
  const jobs = jobsState.items;
  const datasets = datasetsState.items.filter((item) => item.status === 'COMPLETED');
  const [job, setJob] = useState(initialJob);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [reportUuid, setReportUuid] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const run = async (key, action, success) => {
    setBusy(key); setNotice('');
    try { await action(); setNotice(success); await jobsState.refresh({ silent: true }); }
    catch (requestError) { setNotice(requestError.message || 'Request failed'); }
    finally { setBusy(''); }
  };

  const selectModel = (baseModelId, modelParamsB) => setJob((current) => ({
    ...current, baseModelId, ...(modelParamsB && { modelParamsB }),
  }));

  const selectRecipe = (recipe) => setJob((current) => ({
    ...current,
    recipe,
    baseModelId: recipe === 'embedding'
      ? 'unsloth/all-MiniLM-L6-v2'
      : recipe === 'fullparams'
        ? 'unsloth/gemma-3-270m-it'
        : ['unsloth/all-MiniLM-L6-v2', 'unsloth/gemma-3-270m-it'].includes(current.baseModelId)
          ? initialJob.baseModelId : current.baseModelId,
    modelParamsB: recipe === 'embedding' ? '22m' : recipe === 'fullparams' ? '270m' : ['22m', '270m'].includes(current.modelParamsB) ? initialJob.modelParamsB : current.modelParamsB,
  }));

  const create = (event) => {
    event.preventDefault();
    const evalPrompts = job.evalPrompt.trim() ? [{ prompt: job.evalPrompt.trim(), ...(job.evalReference.trim() && { reference: job.evalReference.trim() }) }] : [];
    run('create', () => apiRequest('/api/jobs', {
      token,
      method: 'POST',
      body: JSON.stringify({
        datasetId: Number(job.datasetId), baseModelId: job.baseModelId.trim(), recipe: job.recipe, objective: job.objective,
        rewardModelId: job.rewardModelId.trim() || undefined,
        modelParamsB: job.modelParamsB.trim() || undefined, exportHfRepo: job.exportHfRepo.trim() || undefined,
        exportFormat: job.recipe === 'fullparams' ? 'merged_16bit' : 'lora', evalPrompts,
        dryRun: job.dryRun, maxSeqLength: Number(job.maxSeqLength), batchSize: Number(job.batchSize),
      }),
    }), 'Fine-tune joined the FCFS queue.');
  };

  const remove = () => run('delete', async () => {
    await apiRequest(`/api/jobs/${deleteTarget.jobUuid}`, { token, method: 'DELETE' });
    setDeleteTarget(null);
  }, 'Fine-tune job deleted.');
  const cancel = (row) => run(`cancel-${row.jobUuid}`, () => apiRequest(`/api/jobs/${row.jobUuid}/cancel`, { token, method: 'POST' }), 'Cancellation requested.');
  const resume = (row) => run(`resume-${row.jobUuid}`, () => apiRequest(`/api/jobs/${row.jobUuid}/resume`, { token, method: 'POST' }), 'Checkpoint queued for resume.');

  const columns = [
    { key: 'job', label: 'Run', render: (row) => <div className="table-primary"><strong>{row.baseModelId}</strong><span>{row.jobUuid?.slice(0, 8)} · {row.recipe?.replaceAll('_', ' ')}</span></div> },
    ...(admin ? [{ key: 'owner', label: 'Owner', render: (row) => row.userEmail || '—' }] : []),
    { key: 'status', label: 'Run status', render: (row) => <StatusBadge status={row.status} /> },
    { key: 'activity', label: 'Activity', render: (row) => <span className="muted">{activity(row)}</span> },
    { key: 'hf', label: 'Hub export', render: (row) => { const push = getHfPushState(row); return push ? <StatusBadge status={push.status} /> : <span className="muted">Not requested</span>; } },
    { key: 'updated', label: 'Updated', render: (row) => date(row.updatedAt || row.createdAt) },
    { key: 'actions', label: '', render: (row) => <div className="row-actions"><button className="text-button" onClick={() => setReportUuid(row.jobUuid)}>Open run</button>{['QUEUED', ...ACTIVE_WORKER_STATUSES].includes(String(row.status).toUpperCase()) && <button className="text-button danger-link" disabled={busy === `cancel-${row.jobUuid}`} onClick={() => cancel(row)}>Cancel</button>}{['CANCELLED', 'INTERRUPTED'].includes(String(row.status).toUpperCase()) && row.resumeFromCheckpoint && <button className="text-button" disabled={busy === `resume-${row.jobUuid}`} onClick={() => resume(row)}>Resume</button>}<button className="text-button danger-link" disabled={hasFreshHeartbeat(row)} title={hasFreshHeartbeat(row) ? 'This worker is still reporting as alive' : 'Delete this run'} onClick={() => setDeleteTarget(row)}>Delete</button></div> },
  ];

  const active = jobs.filter(hasFreshHeartbeat).length;
  const published = jobs.filter((item) => getHfPushState(item)?.status === 'published').length;

  return (
    <DashboardShell section="finetune" eyebrow="FINE-TUNING" title="Turn approved data into a useful model." description="Discover Unsloth models, choose an explicit training recipe, and follow every export to completion.">
      {(notice || jobsState.error || datasetsState.error) && <div className={`notice ${(jobsState.error || datasetsState.error || notice.toLowerCase().includes('failed')) ? 'notice-error' : ''}`}>{jobsState.error || datasetsState.error || notice}</div>}
      <section className="metric-row"><div><span>Total runs</span><strong>{jobs.length}</strong></div><div><span>Active</span><strong>{active}</strong></div><div><span>Published</span><strong>{published}</strong></div><div><span>Ready datasets</span><strong>{datasets.length}</strong></div></section>

      {!admin && <form className="tuning-workbench" onSubmit={create}>
        {job.recipe !== 'embedding' && <ModelPicker token={token} value={job.baseModelId} onSelect={selectModel} />}

        <section className="workspace-section bone-panel">
          <div className="section-heading"><div><p className="eyebrow">TRAINING RECIPE</p><h2>Set the adaptation strategy</h2></div><p>Nothing is silently downgraded. Capacity checks can reject or suggest, but never swap your choice.</p></div>
          <div className="recipe-grid">{RECIPES.map((recipe) => <button type="button" key={recipe.id} className={`recipe-card ${job.recipe === recipe.id ? 'selected' : ''}`} onClick={() => selectRecipe(recipe.id)}><span>{recipe.label}</span><small>{recipe.note}</small></button>)}</div>
        </section>

        <section className="workspace-section configure-section">
          <div className="section-heading"><div><p className="eyebrow">CONFIGURATION</p><h2>Connect data and destination</h2></div><p>Use a scored dataset. Hugging Face publication requires a valid worker-side token.</p></div>
          <div className="configuration-grid">
            <div className="form-card"><span className="card-index">01</span><h3>Inputs</h3><label>Training objective<select value={job.objective} onChange={(event) => setJob({ ...job, objective: event.target.value })}>{OBJECTIVES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label><label>Scored dataset<select value={job.datasetId} onChange={(event) => setJob({ ...job, datasetId: event.target.value })} required><option value="">Select approved data</option>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>#{dataset.id} · {dataset.filename}</option>)}</select></label><label>Base model ID<input value={job.baseModelId} onChange={(event) => setJob({ ...job, baseModelId: event.target.value })} required /></label><label>Parameter hint<input value={job.modelParamsB} onChange={(event) => setJob({ ...job, modelParamsB: event.target.value })} placeholder="270m or 1b (maximum)" pattern="(?:[0-9]+(?:\\.[0-9]+)?[mMbB]?)" title="Use a numeric size up to 1B, for example 270m or 1b." /><small className="muted">GTX 1650 profile: models above 1B are rejected by the worker.</small></label>{job.objective === 'ppo' && <label>Reward model ID<input value={job.rewardModelId} onChange={(event) => setJob({ ...job, rewardModelId: event.target.value })} required placeholder="Reward model repository or path" /></label>}</div>
            <div className="form-card"><span className="card-index">02</span><h3>Output</h3><label>Hugging Face repository<input value={job.exportHfRepo} onChange={(event) => setJob({ ...job, exportHfRepo: event.target.value })} placeholder="Shreevathsa05/<job-uuid> · leave blank to auto" /></label><label>Sequence length<input type="number" min="64" max="32768" value={job.maxSeqLength} onChange={(event) => setJob({ ...job, maxSeqLength: event.target.value })} /></label><label>Micro batch<input type="number" min="1" max="64" value={job.batchSize} onChange={(event) => setJob({ ...job, batchSize: event.target.value })} /></label></div>
            {job.recipe !== 'embedding' && <div className="form-card"><span className="card-index">03</span><h3>Evaluation</h3><label>Evaluation prompt<textarea rows="3" value={job.evalPrompt} onChange={(event) => setJob({ ...job, evalPrompt: event.target.value })} placeholder="Optional baseline and fine-tuned comparison" /></label><label>Expected reference<input value={job.evalReference} onChange={(event) => setJob({ ...job, evalReference: event.target.value })} placeholder="Optional reference answer" /></label></div>}
          </div>
          <div className="launch-bar"><label className="toggle-control"><input type="checkbox" checked={job.dryRun} onChange={(event) => setJob({ ...job, dryRun: event.target.checked })} /><span><b>Simulation only</b><small>Leave this off to train, export, evaluate, and publish for real.</small></span></label><button className="button button-primary" disabled={busy === 'create' || !datasets.length}>{busy === 'create' ? 'Joining queue…' : job.dryRun ? 'Run simulation' : 'Train model now'}</button></div>
        </section>
      </form>}

      <section className="workspace-section"><div className="section-heading"><div><p className="eyebrow">RUN ARCHIVE</p><h2>{admin ? 'All fine-tune runs' : 'Your fine-tune runs'}</h2></div><p>{jobsState.loading ? 'Refreshing runs…' : 'Publication is linked only after the worker confirms the Hub upload.'}</p></div><ResourceTable columns={columns} rows={jobs} rowKey={(row) => row.jobUuid} emptyMessage="No fine-tune runs yet." /></section>
      {reportUuid && <JobReportModal jobUuid={reportUuid} onClose={() => setReportUuid(null)} token={token} />}
      {deleteTarget && <ConfirmDialog title={`Delete run ${deleteTarget.jobUuid.slice(0, 8)}?`} description="This removes the abandoned manager record. If its queued message still exists, the worker will discard it instead of starting training. MinIO and published Hugging Face artifacts are left intact." busy={busy === 'delete'} onCancel={() => setDeleteTarget(null)} onConfirm={remove} />}
    </DashboardShell>
  );
}
