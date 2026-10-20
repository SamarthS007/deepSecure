"""The "What's new" page: where DeepSecure started and where it stands now, at a glance.

A static page: every number is a measured result copied from ``docs/FINAL_REPORT.md``
(the section is in each row's ``source``), never computed here and never estimated.
"Before" is an earlier iteration of this project measured on the same test data
(the published model, the first model, the first whole-system run), so each pair is a
like-for-like comparison.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# One hue, two shades (dataviz "before -> after" form), validated as an ordinal pair
# on the light and dark chart surfaces; the darker shade marks "now".
BEFORE, NOW = "#86b6ef", "#2a78d6"
SAME_SOURCE, NEW_SOURCE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK = "#8a8f98"
# decorative accents for tiles and steps (categorical slots), "r,g,b" for CSS tints
TINTS = ["42,120,214", "27,175,122", "98,80,214", "235,104,52", "232,123,164", "237,161,0"]
TILE_CSS = (
    "<style>"
    + "".join(
        f""".st-key-tile_{i} [data-testid="stMetric"] {{
  background: linear-gradient(135deg, rgba({c},.26), rgba({c},.05));
  border-color: rgba({c},.55); border-top: 4px solid rgb({c});
}}
.st-key-step_{i} {{
  background: linear-gradient(160deg, rgba({c},.22), rgba({c},.03));
  border-color: rgba({c},.55) !important; border-top: 4px solid rgb({c}) !important;
}}
"""
        for i, c in enumerate(TINTS)
    )
    + "</style>"
)  # mid grey: legible on both the light and the dark page

QUALITY_UP = [  # ROC-AUC, higher is better
    {
        "metric": "Voice · real-world speech (In-the-Wild, unseen speakers)",
        "before": 0.505,
        "before_label": "published AASIST",
        "now": 0.853,
        "now_label": "our fine-tuned AASIST",
        "source": "FINAL_REPORT §7.1",
    },
    {
        "metric": "Voice · unseen attacks (ASVspoof 2019 LA eval)",
        "before": 0.764,
        "before_label": "our first voice model (MFCC + MLP)",
        "now": 0.961,
        "now_label": "fine-tuned AASIST (used by the app)",
        "source": "FINAL_REPORT §7.1",
    },
    {
        "metric": "Face · validation set during training",
        "before": 0.736,
        "before_label": "after the first (head-only) epoch",
        "now": 0.990,
        "now_label": "epoch 6 (kept)",
        "source": "FINAL_REPORT §7.1 (training curve)",
    },
]

FALSE_ALARMS_DOWN = [  # % of genuine files flagged, lower is better
    {
        "metric": "Genuine group photos, 20 faces (simulated)",
        "before": 85.0,
        "before_label": "raw scores",
        "now": 0.0,
        "now_label": "small-face correction",
        "source": "FINAL_REPORT §7.1 (group photos)",
    },
    {
        "metric": "Genuine group photos, 5 faces (simulated)",
        "before": 38.8,
        "before_label": "raw scores",
        "now": 2.6,
        "now_label": "small-face correction",
        "source": "DOCUMENTATION §8.18",
    },
    {
        "metric": "Real YouTube speech (voice detector)",
        "before": 95.1,
        "before_label": "published AASIST",
        "now": 27.1,
        "now_label": "our fine-tuned AASIST",
        "source": "FINAL_REPORT §7.1",
    },
    {
        "metric": "Genuine voices (whole system, ASVspoof eval)",
        "before": 49.0,
        "before_label": "first whole-system run",
        "now": 9.0,
        "now_label": "after the 3 design fixes",
        "source": "FINAL_REPORT §7.1",
    },
    {
        "metric": "Genuine talking-face videos (whole system, RAVDESS)",
        "before": 47.5,
        "before_label": "first whole-system run",
        "now": 6.7,
        "now_label": "after the 3 design fixes",
        "source": "FINAL_REPORT §7.1",
    },
]

STANDING = [  # ROC-AUC today, by test set
    ("Face · DeepFakeFace (unseen people)", 0.989, "same source as training"),
    ("Voice · ASVspoof 2019 LA eval", 0.961, "same source as training"),
    ("Voice · In-the-Wild (unseen speakers)", 0.853, "same source as training"),
    ("Lip-sync · RAVDESS (unseen actors)", 0.714, "same source as training"),
    ("Face · DF40 (40 newer fake methods)", 0.717, "new source, never seen"),
    ("Voice · modern commercial TTS", 0.473, "new source, never seen"),
]

MILESTONES = [
    ("1 · The plan", "Video + audio, one detector each, Flask app, published models."),
    ("2 · Built", "3 detectors, fusion, security layer, photos too, audit database."),
    ("3 · Trained", "Face AUC **0.989** on unseen people; voice EER **0.7 %** on ASVspoof."),
    ("4 · Tested as a system", "Genuine videos wrongly flagged: **47.5 % → 6.7 %**."),
    ("5 · Real-world voices", "Fine-tuned voice model: AUC **0.505 → 0.853**."),
    ("6 · Group photos", "Up to 20 faces; false alarms on groups **85 % → 0 %**."),
]

CAPABILITIES = [
    ("Faces, voices and lip-sync together", "one of them", "✅ all three"),
    ("Videos, audio and photos (incl. groups)", "one media type", "✅ all"),
    ("Decision with written reasons", "a score only", "✅"),
    ("Says so when nothing could be checked", "❌", "✅"),
    ("Flags unusual input it can't judge", "❌", "✅"),
    ("Recognises repeat attacks", "❌", "✅"),
    ("Audit trail of every analysis", "❌", "✅"),
]


def dumbbell(rows: list[dict], title: str, domain: tuple[float, float], unit: str):
    """Before -> now per row: a light dot, a dark dot and a line between them."""
    df = pd.DataFrame(rows)
    fmt = (lambda v: f"{v:.3f}") if unit == "AUC" else (lambda v: f"{v:g} %")
    df["now_text"] = df["now"].map(fmt)
    long = pd.concat(
        [
            df.assign(stage="Before", value=df["before"], detail=df["before_label"]),
            df.assign(stage="Now", value=df["now"], detail=df["now_label"]),
        ]
    )
    long["value_text"] = long["value"].map(fmt)
    y = alt.Y("metric:N", sort=None, title=None, axis=alt.Axis(labelLimit=360, labelPadding=8))
    x_scale = alt.Scale(domain=list(domain))
    line = (
        alt.Chart(df)
        .mark_rule(strokeWidth=2, color=BEFORE)
        .encode(y=y, x=alt.X("before:Q", scale=x_scale, title=title), x2="now:Q")
    )
    dots = (
        alt.Chart(long)
        .mark_circle(size=170, opacity=1, stroke="white", strokeWidth=2)
        .encode(
            y=y,
            x=alt.X("value:Q", scale=x_scale),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=["Before", "Now"], range=[BEFORE, NOW]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("metric:N", title="Test"),
                alt.Tooltip("stage:N", title="When"),
                alt.Tooltip("detail:N", title="Model / version"),
                alt.Tooltip("value_text:N", title=unit),
            ],
        )
    )
    value = (  # the "now" value above its dot (text in neutral ink, readable in both themes)
        alt.Chart(df)
        .mark_text(dy=-15, fontSize=13, fontWeight="bold", color=INK)
        .encode(y=y, x=alt.X("now:Q", scale=x_scale), text="now_text:N")
    )
    return (line + dots + value).properties(height=alt.Step(56), background="transparent")


def standing_chart():
    df = pd.DataFrame(STANDING, columns=["test", "auc", "data"])
    y = alt.Y("test:N", sort=None, title=None, axis=alt.Axis(labelLimit=360, labelPadding=8))
    x = alt.X("auc:Q", scale=alt.Scale(domain=[0, 1]), title="ROC-AUC (1.0 = perfect)")
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            y=y,
            x=x,
            color=alt.Color(
                "data:N",
                scale=alt.Scale(
                    domain=["same source as training", "new source, never seen"],
                    range=[SAME_SOURCE, NEW_SOURCE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("test:N", title="Test"),
                alt.Tooltip("auc:Q", title="ROC-AUC", format=".3f"),
                alt.Tooltip("data:N", title="Test data"),
            ],
        )
    )
    values = bars.mark_text(align="left", dx=6, fontSize=13).encode(
        text=alt.Text("auc:Q", format=".3f"), color=alt.value(INK)
    )
    chance = (
        alt.Chart(pd.DataFrame({"x": [0.5]}))
        .mark_rule(strokeDash=[4, 4], color=INK)
        .encode(x="x:Q")
    )
    return (bars + values + chance).properties(height=alt.Step(34), background="transparent")


def render() -> None:
    st.header("What's new: from the first version to now")
    st.caption(
        "Every number is measured on held-out test data and taken from the final report; "
        "“before” is an earlier version of DeepSecure tested on the same data."
    )

    st.html(TILE_CSS)
    c1, c2, c3, c4 = (col.container(key=f"tile_{i}") for i, col in enumerate(st.columns(4)))
    c1.metric("Face detector, unseen people", "0.989 AUC", "94.4 % accuracy", delta_color="off",
              border=True)  # fmt: skip
    c2.metric("Voice detector, real-world speech", "0.853 AUC", "+0.348 vs 0.505", border=True)
    c3.metric("Genuine videos wrongly flagged", "6.7 %", "-40.8 pts vs 47.5 %",
              delta_color="inverse", border=True)  # fmt: skip
    c4.metric("Automated tests", "293", "96 % code coverage", delta_color="off", border=True)

    st.subheader("How we got here")
    cols = st.columns(len(MILESTONES))
    for i, (col, (step, text)) in enumerate(zip(cols, MILESTONES, strict=True)):
        with col.container(border=True, height="stretch", key=f"step_{i}"):
            st.markdown(f"**{step}**")
            st.caption(text)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Detection got better ↑")
        st.altair_chart(
            dumbbell(QUALITY_UP, "ROC-AUC (0.5 = guessing, 1.0 = perfect)", (0.5, 1.0), "AUC"),
            width="stretch",
        )
    with right:
        st.subheader("False alarms went down ↓")
        st.altair_chart(
            dumbbell(FALSE_ALARMS_DOWN, "% of genuine files flagged", (0, 100), "%"),
            width="stretch",
        )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Where we stand today")
        st.altair_chart(standing_chart(), width="stretch")
        st.caption(
            "Blue: test data from the same sources the models learned from (but unseen "
            "people, speakers or attacks). Orange: sources never seen. These show the honest "
            "limit: newer fake methods and modern commercial voice clones are still hard. "
            "Dashed line: guessing."
        )
    with right:
        st.subheader("One detector vs DeepSecure")
        st.dataframe(
            pd.DataFrame(CAPABILITIES, columns=["", "A single detector", "DeepSecure"]),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Show all numbers and their sources"):
        rows = [
            {"test": r["metric"], "before": r["before"], "before = ": r["before_label"],
             "now": r["now"], "now = ": r["now_label"], "source": r["source"]}
            for r in QUALITY_UP + FALSE_ALARMS_DOWN
        ]  # fmt: skip
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "AUC rows: ROC-AUC. Other rows: % of genuine files flagged. Simulated group photos "
            "use DeepFakeFace test faces shrunk to 64–128 px. The fused system on jointly faked "
            "face + voice media: TBD (no open dataset has them)."
        )
