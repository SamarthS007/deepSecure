"""Phase 8: the Streamlit app, driven headlessly with Streamlit's AppTest.

Every test points the app at a temporary database, a temporary media folder and
an empty models folder (tiny CNN on the CPU), so it never touches data/deepsecure.db.
Files are analysed the way a user does it: dashboard card -> pop-up -> upload -> Analyze.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from deepsecure.config.settings import Settings, get_settings
from deepsecure.storage.repository import Repository
from deepsecure.utils.synthetic_media import (
    write_corrupted_file,
    write_face_photo,
    write_group_photo,
    write_no_face_image,
)
from tests.conftest import ROOT, isolated_models_dir, needs_mediapipe_assets

APP = str(ROOT / "deepsecure" / "app" / "streamlit_app.py")
pytestmark = [pytest.mark.slow, needs_mediapipe_assets]


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    samples = tmp_path / "data" / "samples"
    write_face_photo(samples / "face.jpg")
    write_corrupted_file(samples / "corrupted.mp4")
    write_no_face_image(samples / "no_face.png")
    write_group_photo(samples / "group.jpg")
    env = {
        "DEEPSECURE_DATA_DIR": str(tmp_path / "data"),
        "DEEPSECURE_DB_URL": f"sqlite:///{tmp_path / 'app.db'}",
        "DEEPSECURE_MODELS_DIR": str(isolated_models_dir(tmp_path)),
        "DEEPSECURE_VISUAL_ARCH": "test_resnet",
        "DEEPSECURE_DEVICE": "cpu",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    st.cache_resource.clear()  # the cached pipeline must pick up these settings
    yield tmp_path
    get_settings.cache_clear()
    st.cache_resource.clear()


def started(app_env: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def analysed(app_env: Path, name: str) -> AppTest:
    """Open the Analyse pop-up, upload ``data/samples/<name>`` and press Analyze."""
    at = started(app_env)
    at.button(key="card_analyse").click().run()
    path = app_env / "data" / "samples" / name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    at.file_uploader(key="upload").set_value((name, path.read_bytes(), mime)).run()
    at.button(key="analyze_upload").click().run()
    assert not at.exception, at.exception
    return at


def to_results(at: AppTest) -> AppTest:
    assert any("Check the **Results** tab" in s.value for s in at.success)
    at.button(key="to_results").click().run()
    assert not at.exception, at.exception
    assert at.session_state["page"] == "results"
    return at


def test_app_starts_on_the_dashboard(app_env: Path) -> None:
    at = started(app_env)
    assert any("Demo mode" in w.value for w in at.warning)
    keys = {b.key for b in at.button}
    assert {"card_analyse", "card_results", "card_history", "card_whats_new", "nav_about"} <= keys
    assert not at.radio  # no "use a sample" choice any more


def test_analyse_pop_up_lists_the_accepted_formats(app_env: Path) -> None:
    at = started(app_env)
    at.button(key="card_analyse").click().run()
    assert any("Accepted formats" in m.value and ".mp4" not in m.value for m in at.markdown)
    assert any("mp4" in m.value and "wav" in m.value and "jpg" in m.value for m in at.markdown)
    assert at.button(key="analyze_upload").disabled  # nothing uploaded yet


def test_analyse_a_file_end_to_end(app_env: Path) -> None:
    at = to_results(analysed(app_env, "face.jpg"))
    result = at.session_state["result"]
    level = result.report.security.risk_level
    banners = [*at.success, *at.warning, *at.error]
    assert any(level in b.value and "face.jpg" in b.value for b in banners)
    assert {m.label for m in at.metric} >= {"Risk score", "Fused fake probability", "Confidence",
                                            "Evidence coverage"}  # fmt: skip
    s = Settings(_env_file=None, db_url=f"sqlite:///{app_env / 'app.db'}")
    assert Repository(s).counts()["detection_results"] == 1  # the History page's data


def test_no_face_photo_explains_that_nothing_was_checked(app_env: Path) -> None:
    at = to_results(analysed(app_env, "no_face.png"))
    assert at.session_state["result"].report.security.risk_level == "SUSPICIOUS"
    assert any("Nothing in this file could be checked" in i.value for i in at.info)
    assert any("no face found" in i.value for i in at.info)


def test_group_photo_lists_every_face(app_env: Path) -> None:
    at = to_results(analysed(app_env, "group.jpg"))
    assert at.session_state["result"].frames.faces_found == 4
    assert any("Group photo: 4 faces checked" in h.value for h in at.subheader)
    # demo mode has no small-face calibration: the faces are shown, but not scored
    assert any("small-face calibration" in i.value for i in at.info)


def test_invalid_file_is_rejected_cleanly(app_env: Path) -> None:
    at = analysed(app_env, "corrupted.mp4")
    assert any("can't be analysed" in e.value for e in at.error)
    assert "result" not in at.session_state
    assert not at.success


def test_every_page_opens(app_env: Path) -> None:
    at = started(app_env)
    at.button(key="card_results").click().run()
    assert any("No analysis yet" in i.value for i in at.info)
    at.button(key="nav_history").click().run()
    assert any("Analyses (0)" in h.value for h in at.subheader)
    at.button(key="nav_whats_new").click().run()
    assert not at.exception, at.exception
    assert any("Where we stand today" in h.value for h in at.subheader)
    assert any(m.label == "Automated tests" for m in at.metric)
    at.button(key="nav_about").click().run()
    assert any("About DeepSecure" in h.value for h in at.header)
    at.button(key="nav_home").click().run()
    assert at.session_state["page"] == "home" and at.button(key="card_analyse")


def test_missing_mediapipe_files_stop_the_app(app_env: Path, monkeypatch) -> None:
    empty = app_env / "empty_models"
    empty.mkdir()
    monkeypatch.setenv("DEEPSECURE_MODELS_DIR", str(empty))
    get_settings.cache_clear()
    at = started(app_env)
    assert any("Missing MediaPipe model files" in e.value for e in at.error)


def _install_trained_checkpoints(models: Path) -> None:
    """Small checkpoints for all three detectors, labelled as trained on real data."""
    import numpy as np

    from deepsecure.models.audio_model import save_classifier, select_model
    from deepsecure.models.lipsync_model import (
        LipSyncDetector,
        calibration_rows,
        fit_calibration,
        save_calibration,
    )
    from deepsecure.models.visual_model import build_model, save_checkpoint
    from deepsecure.preprocessing.audio_extractor import AudioExtractor
    from deepsecure.utils.synthetic_media import synthetic_voice
    from tests.test_lipsync_model import make_audio, make_frames

    s = Settings(_env_file=None, models_dir=models)
    model, arch = build_model("test_resnet", pretrained=False)
    save_checkpoint(model, arch, models / "visual" / "visual_best.pt", data_source="unit-test")

    ax = AudioExtractor(s)
    x = np.stack(
        [
            ax.features_from_waveform(synthetic_voice(bool(i % 2), i, 2.0), 16_000).summary_vector
            for i in range(24)
        ]
    ).astype(np.float64)
    y = np.array([i % 2 for i in range(24)])
    _, clf, _ = select_model(x[:16], y[:16], x[16:], y[16:], s)
    save_classifier(clf, models / "audio" / "audio_classifier.joblib", s, data_source="unit-test")

    clips = [(make_frames(seed=i), make_audio(seed=i)) for i in range(6)]
    rows, labels, _ = calibration_rows(clips, LipSyncDetector(s), np.random.default_rng(0))
    save_calibration(
        fit_calibration(rows, labels),
        models / "lipsync" / "lipsync_baseline_calibration.joblib",
        data_source="unit-test",
    )


def test_demo_banner_disappears_when_all_detectors_are_trained(app_env: Path) -> None:
    _install_trained_checkpoints(app_env / "models")
    at = started(app_env)
    assert not any("Demo mode" in w.value for w in at.warning)
    at = analysed(app_env, "face.jpg")
    assert at.session_state["result"].report.demo_mode is False
