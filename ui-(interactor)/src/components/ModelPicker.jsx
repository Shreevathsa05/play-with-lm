import { useEffect, useState } from 'react';
import { apiRequest } from '../lib/api';

const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

function parameterHint(model) {
  const total = model.safetensors?.total;
  if (!total) return '';
  if (total >= 1e9) return `${(total / 1e9).toFixed(1).replace('.0', '')}b`;
  return `${Math.round(total / 1e6)}m`;
}

export default function ModelPicker({ token, value, onSelect }) {
  const [query, setQuery] = useState('llama');
  const [models, setModels] = useState([]);
  const [state, setState] = useState({ loading: true, error: '' });

  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(async () => {
      setState({ loading: true, error: '' });
      try {
        const result = await apiRequest(`/api/huggingface/models?q=${encodeURIComponent(query)}&limit=12`, { token });
        if (active) { setModels(result || []); setState({ loading: false, error: '' }); }
      } catch (error) {
        if (active) setState({ loading: false, error: error.message });
      }
    }, 350);
    return () => { active = false; window.clearTimeout(timer); };
  }, [query, token]);

  return (
    <section className="model-browser">
      <div className="section-heading compact-heading">
        <div><p className="eyebrow">UNSLOTH CATALOG</p><h2>Choose a starting model</h2></div>
        <a className="text-link" href={`https://huggingface.co/search/full-text?q=${encodeURIComponent(query)}&type=space`} target="_blank" rel="noreferrer">Search model cards & Spaces ↗</a>
      </div>
      <label className="search-field"><span>Search Unsloth models</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="llama, gemma, qwen…" /></label>
      {state.error && <div className="notice notice-error">{state.error} — you can still enter a model ID manually below.</div>}
      {state.loading ? <div className="model-grid skeleton-grid">{Array.from({ length: 4 }, (_, index) => <div className="model-card skeleton" key={index} />)}</div> :
        <div className="model-grid">{models.map((model) => {
          const selected = value === model.id;
          const hint = parameterHint(model);
          return <button type="button" className={`model-card ${selected ? 'selected' : ''}`} key={model.id} onClick={() => onSelect(model.id, hint)}>
            <span className="model-card-top"><span className="model-owner">UNSLOTH</span><span>{selected ? 'Selected' : 'Choose'}</span></span>
            <strong>{model.id?.replace(/^unsloth\//, '')}</strong>
            <span className="model-meta"><span>{model.pipeline_tag || 'model'}</span><span>{compact.format(model.downloads || 0)} downloads</span></span>
          </button>;
        })}</div>}
      {!state.loading && !models.length && !state.error && <p className="empty-state">No Unsloth models matched “{query}”.</p>}
    </section>
  );
}

