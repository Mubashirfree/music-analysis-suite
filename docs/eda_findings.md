# EDA Findings and Their Effect on Preprocessing

## 1. Genre class balance (bar chart of clip counts)
- **Why this chart:** we are counting clips in categories, and a bar chart
  makes unequal counts obvious. A pie chart makes small differences hard to see.
- **What we saw:** 9 genres have 100 clips and jazz has 99 (one corrupted file removed).
- **Meaning:** the data is balanced, so no genre can win by being common.
- **Decisions:** (a) use accuracy plus a confusion matrix; (b) split train/val/test
  stratified by genre so each split keeps about 10% per genre.

## 2. Mel spectrogram, one clip per genre
- **Why this chart:** the mel spectrogram is the actual input image the CNN will
  see, so looking at it checks that genres differ in a way a CNN can learn.
  A plain waveform hides frequency content, and an average spectrum hides time.
- **What we saw:** metal is bright across all frequencies; classical has
  horizontal stripes (held notes) and dark high frequencies; disco and hip-hop show
  vertical beat lines; jazz has quiet gaps; blues, country, rock and reggae look similar.
- **Meaning:** genres have visible texture, so a CNN is a sensible model. The
  similar-looking group is where errors will happen.
- **Decisions:** (a) resample everything to 22050 Hz mono so images are comparable;
  (b) cut clips into 3 s segments, because the texture is visible within a few
  seconds and this gives about 19x more training samples; (c) trim silence only at
  clip edges, since jazz silence inside a clip is part of the genre; (d) peak-normalise
  loudness so the model learns texture, not recording volume. This is a design
  choice we will test later, because loudness may carry some mood information.

## 3. Valence-arousal scatter plot (DEAM vs PMEmo)
- **Why this chart:** each song has two continuous numbers. A scatter plot is the only
  simple chart that shows both together, including their relationship and
  empty regions. Two separate histograms cannot show that.
- **What we saw:** a diagonal band (happier songs are also more energetic;
  r = 0.59 DEAM, 0.71 PMEmo). The Angry/Tense and Calm/Relaxed corners are nearly empty.
- **Meaning:** the two outputs are related, which supports one shared network
  predicting both. The sparse corners mean few training examples for those moods.
- **Decisions:** (a) predict valence and arousal with a shared trunk;
  (b) treat the regression outputs as the main mood result, since quadrant labels
  are made by cutting at 0.5 and songs near the line can flip class.

## 4. Valence and arousal distributions (overlaid density histograms)
- **Why this chart:** DEAM has 1744 songs and PMEmo has 767, so raw counts would
  mislead. Density histograms compare shape fairly, and overlaying them shows the shift directly.
- **What we saw:** PMEmo is shifted toward higher valence (about 0.7 vs 0.45)
  and higher arousal (about 0.75 vs 0.55).
- **Meaning:** the datasets were rated by different people on different songs, so the
  same score does not mean exactly the same thing. This is a dataset bias.
- **Decisions:** (a) rescale both datasets to 0-1 before mixing (DEAM 1-9, PMEmo 0-1);
  (b) keep a `dataset` column on every row; (c) stratify the mood split by
  dataset AND mood class; (d) report results per dataset, and use cross-dataset
  tests in Phase 3.

## 5. Mood class balance (grouped bars by dataset)
- **Why this chart:** it compares counts per class and per source at once.
- **What we saw:** Happy about 1135, Sad about 800, Angry about 290, Calm about 280.
  PMEmo contributes mostly Happy songs.
- **Meaning:** the classes are imbalanced, so accuracy alone would be misleading.
- **Decisions:** (a) stratified split so rare classes appear in every split;
  (b) class weights in the loss and macro-F1 as the metric (during training, P1.7).

## Overall conclusion
Genre data is clean and balanced, so the genre pipeline is simple. Mood data
is the hard part: different scales, a shift between datasets, rare mood
classes and correlated outputs. So preprocessing standardises audio, splits by song
(never by segment) with stratification, normalises the scales, and keeps the dataset
label on every row, ready for the Phase 3 generalisation tests.

## Limitations to state honestly
- GTZAN has no artist information, so we split by clip; scores may be optimistic.
- PMEmo uses chorus clips, DEAM uses 45 s excerpts, so the audio differs between datasets.
