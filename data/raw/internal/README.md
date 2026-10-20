# Internal "scam-style" test set

A small, hand-collected test set that looks like real scam media (phone-quality
video calls, voice notes) rather than the clean academic datasets. It's **only ever
used for testing**, never for training, and it shows how well DeepSecure generalises.

## Folder layout (the folder name is the label)

```
data/raw/internal/
├── real/          genuine recordings
├── fake/          deepfakes / voice clones
├── metadata.csv   one row per file (copy metadata_template.csv)
└── README.md      this file
```

Accepted types: video `.mp4 .mov .avi .mkv .webm`, audio `.wav .mp3 .flac .m4a .ogg`.
Clips should be 1 s to 10 min long and under 200 MB (the app's limits).

## How to fill it

**Target: 10-20 real and 10-20 fake clips.** Small is fine; label quality matters more than quantity.

### Real clips
- Record yourself on your phone or laptop webcam: talking to the camera, 5–30 s, a few
  different rooms and lighting conditions.
- Screen-record a video call **with friends who agree**.
- Audio-only: voice notes you recorded yourself.

### Fake clips
- Publicly released deepfake examples that are *labelled as fake by their publisher*
  (e.g. samples from research datasets you have access to, or official demos). Note the
  source URL in `metadata.csv`.
- Fakes you make yourself with an open-source face-swap or voice-cloning tool, **only using
  your own face/voice or that of people who gave written consent**.

### Rules (ethics; the panel may ask about these)
1. **Consent:** never use anyone's face or voice without their explicit permission.
   Record who consented in `metadata.csv`.
2. **No public figures' likenesses for new fakes.** Making a new deepfake of a real
   person who didn't consent is harmful, and in many places illegal.
3. **Never redistribute** these clips. This folder is git-ignored on purpose.
4. **Label honestly.** If you're not sure a downloaded "fake" is really fake, leave it out.
5. **Same person, many clips?** Give them the same `person` value in `metadata.csv`.
   That keeps them in one identity group (see the leakage section in the docs).

## Then

```bash
uv run python scripts/make_splits.py --dataset internal
```

This writes `data/processed/splits/internal.csv` (all rows `split = test`).
