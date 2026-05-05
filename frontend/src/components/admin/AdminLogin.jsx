import { useState } from 'react';
import { Shield, ArrowRight, X } from 'lucide-react';

/**
 * Faux admin login page.
 * Accepts hardcoded credentials: username="admin", password="admin".
 * This is a placeholder — will be replaced with proper auth in production.
 */
export default function AdminLogin({ onLogin, onCancel }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isShaking, setIsShaking] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    if (username === 'admin' && password === 'admin') {
      onLogin();
    } else {
      setError('Invalid credentials. Use admin / admin');
      setIsShaking(true);
      setTimeout(() => setIsShaking(false), 500);
    }
  }

  return (
    <div className="adminLoginOverlay">
      <div className={`adminLoginCard ${isShaking ? 'shake' : ''}`}>
        <button
          className="adminLoginClose"
          type="button"
          onClick={onCancel}
          aria-label="Back to chat"
        >
          <X size={18} />
        </button>

        <div className="adminLoginHeader">
          <div className="adminLoginIcon">
            <Shield size={28} />
          </div>
          <h2>Admin Panel</h2>
          <p>Sign in to access the dashboard</p>
        </div>

        <form className="adminLoginForm" onSubmit={handleSubmit}>
          <div className="adminLoginField">
            <label htmlFor="admin-username">Username</label>
            <input
              id="admin-username"
              type="text"
              value={username}
              onChange={(e) => { setUsername(e.target.value); setError(''); }}
              placeholder="Enter username"
              autoComplete="username"
              autoFocus
            />
          </div>
          <div className="adminLoginField">
            <label htmlFor="admin-password">Password</label>
            <input
              id="admin-password"
              type="password"
              value={password}
              onChange={(e) => { setPassword(e.target.value); setError(''); }}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </div>

          {error && <div className="adminLoginError">{error}</div>}

          <button className="adminLoginSubmit" type="submit">
            Sign In
            <ArrowRight size={16} />
          </button>
        </form>

        <p className="adminLoginHint">
          Demo credentials: <strong>admin</strong> / <strong>admin</strong>
        </p>
      </div>
    </div>
  );
}
