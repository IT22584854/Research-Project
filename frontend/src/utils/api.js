/**
 * API client for the FastAPI backend.
 * In dev, Vite proxies /api/* → http://localhost:8000, so API_BASE can be empty.
 * In production, set VITE_API_BASE to the backend URL.
 */

const API_BASE = (
    globalThis.__APP_CONFIG__?.VITE_API_BASE ||
    import.meta.env.VITE_API_BASE ||
    ''
).replace(/\/$/, '');

/**
 * Send a chat message to the agent.
 * @param {string} message
 * @param {string} sessionId
 * @returns {Promise<{ response: string, sources: Array, session_id: string }>}
 */
export async function sendChatMessage(message, sessionId) {
    const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }

    return res.json();
}

/**
 * Reset a conversation session on the backend.
 * @param {string} sessionId
 */
export async function resetSession(sessionId) {
    await fetch(`${API_BASE}/api/session/${sessionId}`, { method: 'DELETE' });
}

/**
 * Health check.
 */
export async function healthCheck() {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
}
