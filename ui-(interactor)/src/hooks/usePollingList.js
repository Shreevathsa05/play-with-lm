import { useCallback, useEffect, useState } from 'react';
import { apiRequest } from '../lib/api';

export function usePollingList(path, token, intervalMs = 5000) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async ({ silent = false } = {}) => {
    if (!token) return;
    if (!silent) setLoading(true);
    try {
      const value = await apiRequest(path, { token });
      setItems(value || []);
      setError('');
    } catch (requestError) {
      setError(requestError.message || 'Unable to refresh');
    } finally {
      if (!silent) setLoading(false);
    }
  }, [path, token]);

  useEffect(() => {
    const initial = window.setTimeout(() => refresh(), 0);
    const interval = window.setInterval(() => refresh({ silent: true }), intervalMs);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(interval);
    };
  }, [intervalMs, refresh]);

  return { items, error, loading, refresh };
}

