const API_BASE = 'http://127.0.0.1:8000/api';

/**
 * Thin fetch wrapper that turns non-2xx into a thrown Error carrying the
 * server's own `detail` message. Every caller can then handle failure in one
 * place instead of re-checking `res.ok` and guessing at the reason.
 */
async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      // A non-JSON error body (a proxy page, an empty 502) leaves the
      // status-code message above, which is still better than throwing here.
    }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  stats: () => request('/stats'),

  listCollections: () => request('/collections'),
  createCollection: (name, title, description) =>
    request('/collections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, title, description }),
    }),
  deleteCollection: (id) => request(`/collections/${id}`, { method: 'DELETE' }),

  listFiles: (collectionId) => request(`/collections/${collectionId}/files`),
  listNodes: (fileId) => request(`/files/${fileId}/nodes`),
  node: (nodeId) => request(`/nodes/${nodeId}`),

  search: (q, mode, collectionId) => {
    const params = new URLSearchParams({ q, mode });
    if (collectionId) params.set('collection_id', collectionId);
    return request(`/search?${params}`);
  },
  searchStatus: () => request('/search/status'),

  ask: (question, collectionId) =>
    request('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, collection_id: collectionId }),
    }),

  upload: (collectionId, file) => {
    const form = new FormData();
    form.append('file', file);
    return request(`/collections/${collectionId}/upload`, { method: 'POST', body: form });
  },
  job: (jobId) => request(`/jobs/${jobId}`),
  jobs: () => request('/jobs'),

  // Media is referenced by URL rather than fetched, so <img>/<video> can
  // stream it directly instead of round-tripping through JS.
  thumbnailUrl: (nodeId) => `${API_BASE}/nodes/${nodeId}/thumbnail`,
  nodeMediaUrl: (nodeId) => `${API_BASE}/nodes/${nodeId}/media`,
  fileMediaUrl: (fileId) => `${API_BASE}/files/${fileId}/media`,
};
