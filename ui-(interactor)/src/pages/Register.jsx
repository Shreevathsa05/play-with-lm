import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/auth-context';
import { apiRequest } from '../lib/api';

export default function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setBusy(true);
    try {
      const data = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email, password, role: 'ROLE_STUDENT' }),
      });
      login(data.token);
      navigate('/');
    } catch (requestError) {
      const message = requestError.message || '';
      if (message.includes('409')) {
        setError('An account with this email already exists. Sign in instead.');
      } else if (message.includes('403')) {
        setError('Registration was blocked by the server. Restart the backend and try again.');
      } else {
        setError(message || 'Could not create your account.');
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="login-page">
      <section className="login-intro">
        <Link className="brand" to="/"><span className="brand-mark" />LM Customizer</Link>
        <div>
          <p className="announcement">TWO CLEAR WORKFLOWS · ONE SHARED GPU</p>
          <h1>Score the data. Then tune the model.</h1>
          <p>Keep quality review separate from training while every run stays traceable from source to published artifact.</p>
        </div>
        <p className="eyebrow">BUILT FOR CAMPUS AI LABS</p>
      </section>
      <section className="login-panel">
        <form className="login-card" onSubmit={handleSubmit}>
          <p className="eyebrow">NEW WORKSPACE</p>
          <h2>Create your account</h2>
          <p>Register as a student to upload datasets and launch fine-tuning jobs.</p>
          {error && <div className="notice notice-error">{error}</div>}
          <label>
            Email address
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              placeholder="you@college.edu"
              required
              autoFocus
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              required
            />
          </label>
          <label>
            Confirm password
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              autoComplete="new-password"
              required
            />
          </label>
          <button className="button button-primary" disabled={busy}>
            {busy ? 'Creating account…' : 'Create account'}
          </button>
          <p>
            Already have an account? <Link className="text-link" to="/login">Sign in</Link>
          </p>
        </form>
      </section>
    </main>
  );
}
