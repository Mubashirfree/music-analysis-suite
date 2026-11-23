// Shows one analysed song: genre (top 3), mood numbers and tags.
function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function Bar({ label, value }) {
  const percent = Math.round(value * 100);
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="bar-value">{percent}%</span>
    </div>
  );
}

export default function SongCard({ song }) {
  const topGenres = song.genre
    ? Object.entries(song.genre.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3)
    : [];
  return (
    <article className="card">
      <h2>{song.filename}</h2>
      {song.duration_s != null && <p className="muted">{formatDuration(song.duration_s)}</p>}

      <h3>Genre</h3>
      {topGenres.length === 0 && <p className="muted">Not analysed.</p>}
      {topGenres.map(([name, p]) => <Bar key={name} label={name} value={p} />)}

      <h3>Mood</h3>
      {song.mood ? (
        <>
          <p><strong>{song.mood.quadrant}</strong></p>
          <Bar label="valence" value={song.mood.valence} />
          <Bar label="arousal" value={song.mood.arousal} />
        </>
      ) : (
        <p className="muted">Not analysed.</p>
      )}

      <h3>Tags</h3>
      {song.tags.length === 0 ? (
        <p className="muted">No tags yet.</p>
      ) : (
        <div className="tags">
          {song.tags.map((t) => <span key={t.name} className={`tag ${t.source}`}>{t.name}</span>)}
        </div>
      )}
    </article>
  );
}