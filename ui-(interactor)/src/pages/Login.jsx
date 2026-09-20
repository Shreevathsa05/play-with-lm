import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/auth-context';
import { apiRequest } from '../lib/api';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const data = await apiRequest('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
      login(data.token); navigate('/');
    } catch (requestError) {
      setError(requestError.message || 'Invalid email or password.');
    } finally { setBusy(false); }
  };

  return (
    <main className="login-page">
      <section className="login-intro"><a className="brand" href="/"><span className="brand-mark" />LM Customizer</a><div><p className="announcement">TWO CLEAR WORKFLOWS · ONE SHARED GPU</p><h1>Score the data. Then tune the model.</h1><p>Keep quality review separate from training while every run stays traceable from source to published artifact.</p></div><p className="eyebrow">BUILT FOR CAMPUS AI LABS</p></section>
      <section className="login-panel"><form className="login-card" onSubmit={handleSubmit}><p className="eyebrow">PRIVATE WORKSPACE</p><h2>Continue to the lab</h2><p>Sign in with the account issued by your administrator.</p>{error && <div className="notice notice-error">{error}</div>}<label>Email address<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" placeholder="you@college.edu" required autoFocus /></label><label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required /></label><button className="button button-primary" disabled={busy}>{busy ? 'Signing in…' : 'Enter workspace'}</button></form></section>
    </main>
  );
}
