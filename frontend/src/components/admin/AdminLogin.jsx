import React, { useState } from 'react';
import { Shield, ArrowRight, X } from 'lucide-react';

export default function AdminLogin({ onLogin, onCancel }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username === 'admin' && password === 'admin') {
      onLogin();
    } else {
      setError('Invalid username or password');
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[#f8f9fa] p-6">
      <div className="admin-card w-full max-w-md p-8 relative">
        {/* Close Button */}
        <button 
          onClick={onCancel}
          className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        {/* Header */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-xl bg-emerald-50 flex items-center justify-center mb-4">
            <Shield size={28} className="text-emerald-600" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900">Admin Access</h2>
          <p className="text-sm text-gray-500 mt-1">Sign in to view analytics dashboard</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="admin-input"
              placeholder="Enter username"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="admin-input"
              placeholder="Enter password"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 text-sm rounded-lg p-3 text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="admin-btn admin-btn-primary w-full py-2.5 mt-2"
          >
            Sign In <ArrowRight size={16} />
          </button>
        </form>

        <p className="text-xs text-gray-400 text-center mt-6">
          Use <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">admin</span> / <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">admin</span> to sign in
        </p>
      </div>
    </div>
  );
}
