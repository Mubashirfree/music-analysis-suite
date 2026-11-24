import { useState } from "react";
import { uploadSong } from "../api";
import SongCard from "../components/SongCard";

const MAX_BYTES = 50 * 1024 * 1024; // the server enforces the same limit

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [song, setSong] = useState(null);

  function handleChoose(event) {
    const chosen = event.target.files[0] || null;
    setSong(null);
    if (chosen && chosen.size > MAX_BYTES) {
      setFile(null);
      setError("That file is larger than 50 MB.");
      return;
    }
    setError("");
    setFile(chosen);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setSong(null);
    try {
      setSong(await uploadSong(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>Analyse a song</h2>
      <p className="muted">
        Choose an audio file (WAV, MP3, FLAC or OGG, up to 50 MB). The first analysis after the
        server starts can take a minute while the models load.
      </p>
      <form onSubmit={handleSubmit} className="upload-form">
        <input type="file" accept=".wav,.mp3,.flac,.ogg,audio/*" onChange={handleChoose} />
        <button type="submit" disabled={!file || busy}>
          {busy ? "Analysing..." : "Analyse"}
        </button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {song && <SongCard song={song} />}
    </section>
  );
}