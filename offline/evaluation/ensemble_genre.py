import json

import torch
from torch.utils.data import DataLoader

from offline.config import DOCS_DIR, MODELS_DIR
from offline.training.genre_data import GenreSegments
from offline.training.models import GenreCNN
from offline.training.train_genre import predict, song_metrics

TAGS = ["eca", "eca_s1", "eca_s2", "noeca", "noeca_s1", "noeca_s2"]
GROUPS = {
    "eca_x3": ["eca", "eca_s1", "eca_s2"],
    "noeca_x3": ["noeca", "noeca_s1", "noeca_s2"],
    "all_6": TAGS,
}


def load_model(path, device: str):
    ckpt = torch.load(path, map_location=device)
    model = GenreCNN(use_eca=ckpt["use_eca"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["mean"], ckpt["std"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs, labels = {}, {}
    for tag in TAGS:
        model, mean, std = load_model(MODELS_DIR / f"genre_cnn_{tag}.pt", device)
        for split in ["val", "test"]:
            ds = GenreSegments(split, mean, std)
            _, p = predict(model, DataLoader(ds, batch_size=128), device, len(ds.songs))
            probs[(tag, split)] = p
            labels[split] = ds.labels
        print("loaded", tag)

    results = {}
    for name, tags in GROUPS.items():
        results[name] = {}
        for split in ["val", "test"]:
            avg = torch.stack([probs[(t, split)] for t in tags]).mean(dim=0)
            metrics, _ = song_metrics(avg, labels[split])
            results[name][split] = metrics
        print(name, results[name])

    with open(DOCS_DIR / "results" / "cnn_genre_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()