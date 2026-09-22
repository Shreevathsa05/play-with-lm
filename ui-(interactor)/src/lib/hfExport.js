export function parseJobReport(job) {
  if (!job?.reportJson) return null;
  try { return JSON.parse(job.reportJson); }
  catch { return { raw: job.reportJson }; }
}

const EVENT_LABELS = {
  job_received: 'Job received from queue',
  heartbeat: 'Worker heartbeat',
  pipeline_start: 'Pipeline started',
  importing_training_stack: 'Importing training stack',
  dataset_loading: 'Loading dataset',
  model_loading: 'Loading base model',
  model_loaded: 'Base model loaded',
  dataset_loaded: 'Dataset loaded',
  model_loading: 'Loading base model',
  model_loaded: 'Base model loaded',
  baseline_eval_complete: 'Baseline evaluation complete',
  train_progress: 'Training step',
  train_complete: 'Training finished',
  eval_complete: 'Evaluation complete',
  hf_push_complete: 'Published to Hugging Face',
  pipeline_complete: 'Pipeline complete',
  pipeline_failed: 'Pipeline failed',
  dry_run_pipeline: 'Dry-run simulation',
  scratch_wiped: 'Workspace cleaned up',
};

export function getJobPhase(job, report = parseJobReport(job)) {
  return report?.phase || null;
}

export function getJobEvents(report) {
  return Array.isArray(report?.events) ? report.events : [];
}

export function formatJobLogEvent(event) {
  if (!event) return '';
  const name = event.event || 'log';
  const label = EVENT_LABELS[name] || name.replaceAll('_', ' ');
  const time = event.ts ? new Date(event.ts).toLocaleTimeString() : '';
  const parts = [time, label].filter(Boolean);
  if (name === 'train_progress') {
    if (event.step != null && event.total) parts.push(`step ${event.step}/${event.total}`);
    else if (event.step != null) parts.push(`step ${event.step}`);
    if (event.loss != null) parts.push(`loss ${Number(event.loss).toFixed(4)}`);
    if (event.learning_rate != null) parts.push(`lr ${Number(event.learning_rate).toExponential(2)}`);
  } else if (event.recipe) {
    parts.push(event.recipe);
  } else if (event.status) {
    parts.push(String(event.status));
  } else if (event.error) {
    parts.push(String(event.error));
  }
  return parts.join(' · ');
}

export function getHfPushState(job, report = parseJobReport(job)) {
  const explicit = report?.export?.hf_push || report?.runner?.export?.hf_push;
  if (explicit?.status) return explicit;
  if (!job?.exportHfRepo) return null;

  const jobStatus = String(job.status || '').toUpperCase();
  if (jobStatus === 'FAILED') {
    return { status: 'failed', repo_id: job.exportHfRepo, url: null };
  }
  if (jobStatus === 'RUNNING') {
    return { status: 'pushing', repo_id: job.exportHfRepo, url: null };
  }
  return { status: 'requested', repo_id: job.exportHfRepo, url: null };
}
