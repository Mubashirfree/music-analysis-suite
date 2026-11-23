import { listSongs, audioUrl } from "../api";
import { useEffect, useState } from "react";

// Accept either a plain array or an object that wraps the array.
function toArray(data) {
  if (Array.isArray(data)) return data;
  return data?.items ?? data?.songs ?? [];
}

export default function CatalogPage() {
  const [filters, setFilters] = useState({ genre: "", quadrant: "", tag: "" });
  const [applied, setApplied] = useState({});
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    listSongs(applied)
      .then((data) => { if (!cancelled) setSongs(toArray(data)); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [applied]);

  function handleChange(event) {
    setFilters({ ...filters, [event.target.name]: event.target.value });
  }

  function handleApply(event) {
    event.preventDefault();
    const clean = {};
    for (const [key, value] of Object.entries(filters)) {
      if (value.trim()) clean[key] = value.trim().toLowerCase();
    }
    setApplied(clean);
  }

  function topGenre(song) {
    if (!song.genre) return null;
    const best = Object.entries(song.genre.probabilities).sort((a, b) => b[1] - a[1])[0];
    return best ? `${best[0]} ${Math.round(best[1] * 100)}%` : null;
  }

  return (
    <section>
      <h2>Catalogue</h2>
      <form onSubmit={handleApply} className="filters">
        <input name="genre" placeholder="genre (e.g. rock)" value={filters.genre} onChange={handleChange} />
        <input name="quadrant" placeholder="mood quadrant" value={filters.quadrant} onChange={handleChange} />
        <input name="tag" placeholder="tag (e.g. calm)" value={filters.tag} onChange={handleChange} />
        <button type="submit">Filter</button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {loading && <p className="muted">Loading...</p>}
      {!loading && !error && songs.length === 0 && (
        <p className="muted">No songs yet. Analyse one on the Upload tab.</p>
      )}

      <ul className="song-list">
        {songs.map((song) => (
                    <li key={song.song_id} className="song-row">
            <strong>{song.filename}</strong>
            <span className="muted">
              {[topGenre(song), song.mood?.quadrant].filter(Boolean).join(" · ") || "not analysed"}
            </span>
            <audio controls preload="none" src={audioUrl(song.song_id)} />
            <span className="tags">
              {(song.tags || []).map((t) => (
                <span key={t.name} className={`tag ${t.source}`}>{t.name}</span>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}