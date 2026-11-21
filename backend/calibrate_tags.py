"""Spread of predicted valence and arousal over the library, to set tag thresholds (P2.6).

Reads the cached song vectors and runs only the small mood network (no audio, no MERT),
so it takes seconds. It prints numbers and changes nothing.
"""
import numpy as np

from backend.autotag import HIGH, LOW
from backend.similarity import VEC
from offline.features.cache import load_feature, load_manifest
from offline.inference import _mood_from_vector

PCTS = [5, 10, 25, 33, 50, 67, 75, 90, 95]


def show(name, values):
    row = np.percentile(values, PCTS)
    print(f"   {name:30s}" + "  ".join(f"{v:5.2f}" for v in row))


def main():
    genre = load_manifest("genre")
    mood = load_manifest("mood")
    uids = list(genre["uid"]) + list(mood["uid"])
    moods = (_mood_from_vector(load_feature(VEC, u)) for u in uids)
    pred = np.array([[m["valence"], m["arousal"]] for m in moods])
    n = len(genre)  # the first n rows are GTZAN songs, the rest are DEAM and PMEmo

    print("Percentiles:                   " + "  ".join(f"p{p:<4d}" for p in PCTS))
    print("Predicted, all songs")
    show("valence", pred[:, 0])
    show("arousal", pred[:, 1])
    print("DEAM + PMEmo songs: true labels against predictions")
    show("true valence", mood["valence_mean"].to_numpy())
    show("predicted valence", pred[n:, 0])
    show("true arousal", mood["arousal_mean"].to_numpy())
    show("predicted arousal", pred[n:, 1])

    print(f"Share of all songs that would get each tag now (thresholds {LOW} and {HIGH}):")
    for label, mask in (
        ("energetic", pred[:, 1] >= HIGH),
        ("calm", pred[:, 1] <= LOW),
        ("positive", pred[:, 0] >= HIGH),
        ("negative", pred[:, 0] <= LOW),
    ):
        print(f"   {label:10s} {mask.mean():.1%}")

    print("Mean predicted values per GTZAN genre (does the order make sense?)")
    table = genre.assign(valence=pred[:n, 0], arousal=pred[:n, 1])
    print(table.groupby("genre")[["valence", "arousal"]].mean().round(2).sort_values("arousal").to_string())


if __name__ == "__main__":
    main()