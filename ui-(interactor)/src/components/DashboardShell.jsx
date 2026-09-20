import { useAuth } from '../context/auth-context';
import { NavLink } from 'react-router-dom';

export default function DashboardShell({ section, eyebrow, title, description, children }) {
  const { user, logout } = useAuth();
  const root = user?.role === 'ROLE_ADMIN' ? '/admin' : '/student';
  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink className="brand" to={`${root}/data`} aria-label="LM Customizer home"><span className="brand-mark" />LM Customizer</NavLink>
        <nav className="workflow-nav" aria-label="Workspace">
          <NavLink className={section === 'data' ? 'active' : ''} to={`${root}/data`}>Data scoring</NavLink>
          <NavLink className={section === 'finetune' ? 'active' : ''} to={`${root}/finetune`}>Fine-tuning</NavLink>
        </nav>
        <div className="topbar-user">
          <span><b>{user?.email}</b><small>{user?.role === 'ROLE_ADMIN' ? 'Administrator' : 'Student workspace'}</small></span>
          <button className="button button-ghost button-small" onClick={logout}>Sign out</button>
        </div>
      </header>
      <main className="page-wrap">
        <section className="page-heading">
          <p className="announcement">{eyebrow}</p>
          <h1>{title}</h1>
          <p>{description}</p>
        </section>
        {children}
      </main>
    </div>
  );
}
