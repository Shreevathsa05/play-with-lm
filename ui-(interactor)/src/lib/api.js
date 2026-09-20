export async function apiRequest(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try {
      const parsed = JSON.parse(body);
      message = parsed.detail || parsed.message || parsed.error || body;
    } catch { /* Keep non-JSON response text. */ }
    if (response.status === 401) {
      window.dispatchEvent(new Event('fearless-gpus:unauthorized'));
      throw new Error('Your session expired. Please sign in again.');
    }
    throw new Error(message || `Request failed (${response.status})`);
  }
  if (response.status === 204) return null;
  const contentType = response.headers.get('content-type') || '';
  return contentType.includes('json') ? response.json() : response.text();
}
