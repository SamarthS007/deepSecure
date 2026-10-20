# Bundled assets

## `nasa_portrait_face.jpg`

- **What:** a 400×400 head crop of NASA's official portrait of astronaut Eileen Collins
  ("Commander Eileen Collins", NASA image GPN-2000-001177, photographer Robert Markowitz).
- **Source:** https://commons.wikimedia.org/wiki/File:Commander_Eileen_Collins_-_GPN-2000-001177.jpg
  (Great Images in NASA archive), downloaded 2026-10-16 as the 1280-px rendition and
  cropped/resized with OpenCV.
- **Licence:** public domain (work of the U.S. federal government; Wikimedia Commons metadata:
  `LicenseShortName = Public domain`, `Copyrighted = False`).
- **Why it's here:** tests and demo clips need a real human face for MediaPipe to detect,
  and none of the synthetic clips have one. A public-domain photo avoids any consent or
  licensing issue. `deepsecure/utils/synthetic_media.py` animates its mouth to make
  "talking face" clips with *known* lip-sync. These are **not** deepfakes of the person; they
  are visibly synthetic test signals and must never be presented as real footage.
- **Measured with MediaPipe (2026-10-16):** face box (92, 101, 197, 197), score 0.936;
  inner-lip centre (landmarks 13/14) at pixel (187, 243); face height (landmarks 10→152) ≈ 210 px.
