export function getApiBase() {
  const runtimeBase = window.__APP_CONFIG__?.VITE_API_BASE;
  const envBase = import.meta.env.VITE_API_BASE;
  const fallbackBase = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';
  return (runtimeBase || envBase || fallbackBase).replace(/\/$/, '');
}

export function apiUrl(path) {
  const base = getApiBase();
  if (!base) {
    throw new Error('API base is not configured. Set VITE_API_BASE or runtime-config.js.');
  }
  return `${base}${path}`;
}

export async function apiRequest(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof data === 'object' ? data.detail || data.message : data;
    throw new Error(detail || `Request failed with HTTP ${response.status}`);
  }

  return data;
}
