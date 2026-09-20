export default function ConfirmDialog({ title, description, confirmLabel = 'Delete', busy, onCancel, onConfirm }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="confirm-title" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="confirm-card">
        <p className="eyebrow">CONFIRM ACTION</p>
        <h2 id="confirm-title">{title}</h2>
        <p>{description}</p>
        <div className="dialog-actions">
          <button className="button button-ghost" onClick={onCancel} disabled={busy}>Keep it</button>
          <button className="button button-danger" onClick={onConfirm} disabled={busy}>{busy ? 'Deleting…' : confirmLabel}</button>
        </div>
      </section>
    </div>
  );
}

