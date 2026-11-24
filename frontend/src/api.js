// Thin wrapper around the backend API. Every page imports from here,
// so the server address lives in one place.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
   } catch (err) {
    console.error("fetch failed:", err);
    throw new Error(`Cannot reach the server. Is the backend running on ${BASE}?`);
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* the error body was not JSON */
    }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

export const health = () => request("/health");

export function uploadSong(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/songs", { method: "POST", body: form });
}

export function listSongs({ genre, quadrant, tag, limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset });
  if (genre) params.set("genre", genre);
  if (quadrant) params.set("quadrant", quadrant);
  if (tag) params.set("tag", tag);
  return request(`/songs?${params}`);
}

export const getSong = (id) => request(`/songs/${id}`);
export const similarTo = (id, k = 10) => request(`/songs/${id}/similar?k=${k}`);
export const deleteSong = (id) => request(`/songs/${id}`, { method: "DELETE" });
export const addTag = (id, name) =>
  request(`/songs/${id}/tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });

  export const audioUrl = (id) => `${BASE}/songs/${id}/audio`;