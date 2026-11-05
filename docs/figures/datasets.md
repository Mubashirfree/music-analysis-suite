# Datasets

## Genre

### GTZAN
| Item | Detail |
|---|---|
| Used for | Genre classification (CNN with ECA, Random Forest baseline) |
| Source | Tzanetakis and Cook, "Musical genre classification of audio signals", 2002 |
| Obtained from | Kaggle mirror: andradaolteanu/gtzan-dataset-music-genre-classification |
| Format | 30 s mono WAV clips, 22050 Hz |
| Size | 1000 clips downloaded, 999 used (10 genres, 100 each, jazz 99) |
| Labels | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| License | Kaggle lists "other"; no formal license, widely used for research |
| Known issues | jazz.00054.wav excluded as corrupted. Repetitions, mislabels and artist overlap are documented (Sturm, 2013), so scores can be optimistic. |

## Mood

### DEAM
| Item | Detail |
|---|---|
| Used for | Valence and arousal regression plus categorical mood |
| Source | Aljanaki, Yang, Soleymani, "Developing a benchmark for emotional analysis of music", 2017 |
| Obtained from | Kaggle mirror: imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music |
| Format | MP3 clips plus CSV annotations (static, song-level averages) |
| Size | 1802 audio files; 1744 with static annotations |
| Labels | Valence and arousal, 1 to 9 |
| License | CC-BY-NC-SA-4.0 (non-commercial) |

### PMEmo
| Item | Detail |
|---|---|
| Used for | Valence and arousal |
| Source | Zhang et al., "The PMEmo dataset for music emotion recognition", ICMR 2018 |
| Obtained from | Kaggle mirror: adityaraghuvanshi999/pmemo-original |
| Format | MP3 chorus clips plus static_annotations.csv |
| Size | 767 usable clips (original has 794 songs) |
| Labels | Valence and arousal, 0 to 1 |
| License | Unknown on the mirror; check the original PMEmo page before citing |

### EmoMusic (1000 Songs)
| Item | Detail |
|---|---|
| Status | Not downloaded yet; will be added with the same loader pattern |
| Source | Soleymani et al., "1000 songs for emotional analysis of music", 2013 |
| Obtained from | TODO |
| Format | TODO |
| Size | TODO |
| Labels | TODO: check the scale |
| License | TODO |

## Notes
- Mood datasets use different scales (DEAM 1 to 9, PMEmo 0 to 1), so all are normalised to 0 to 1 before joint training.
- After normalising, PMEmo is skewed happier and higher-energy than DEAM (valence mean 0.60 vs 0.49), which is why cross-dataset evaluation matters.