"""FinGuard Streamlit app: scan text for PII, inspect detections, anonymize.

Run from the repository root:  streamlit run app/streamlit_app.py

This file only renders. Detection is the frozen ``HybridDetector``,
redaction is ``finguard.anonymization.anonymize``, and the workflow, session
state, highlighting and exports live in ``finguard.interface``.

Privacy: nothing is printed, logged or written to disk. The submitted text
and results live in this browser session's state only and are cleared when
the input changes. The cached detector holds the model, never user data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import finguard
from finguard.interface import exports, workflow
from finguard.interface.highlight import LABEL_STYLES, highlight_html, legend_html, stylesheet
from finguard.interface.samples import SAMPLE_TEXT

INPUT_KEY = "fg_input"

st.set_page_config(page_title="FinGuard", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:2.2rem;max-width:1200px}
.fg-hero{display:flex;align-items:center;gap:.9rem;margin-bottom:.2rem}
.fg-hero .fg-logo{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:1.4rem;background:linear-gradient(135deg,#0F766E,#155E75);color:#fff}
.fg-hero h1{font-size:1.9rem;margin:0;padding:0;letter-spacing:-.01em}
.fg-sub{opacity:.75;margin:.1rem 0 1.2rem;font-size:1rem}
.fg-step{font-size:.75rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#0F766E}
</style>
""",
    unsafe_allow_html=True,
)


# --- shared resources (model only; never user data) -------------------------

@st.cache_resource(show_spinner="Loading the FinGuard detector…")
def get_detector():
    return workflow.build_detector()


@st.cache_resource(show_spinner=False)
def get_detector_info():
    return workflow.detector_info()


def load_detector():
    try:
        return get_detector(), None
    except workflow.UserFacingError as exc:
        return None, str(exc)


# --- callbacks (run before widgets are created) -----------------------------

def _load_sample():
    st.session_state[INPUT_KEY] = SAMPLE_TEXT


def _clear():
    st.session_state[INPUT_KEY] = ""
    workflow.clear_results(st.session_state)


# --- layout -------------------------------------------------------------------

st.markdown(
    '<div class="fg-hero"><div class="fg-logo">🛡️</div><h1>FinGuard</h1></div>'
    '<p class="fg-sub">Find and redact personal data in financial text (names, contact details, '
    "SSNs, card blocks, companies and URLs) before you share it.</p>",
    unsafe_allow_html=True,
)

detector, detector_error = load_detector()

with st.sidebar:
    st.subheader("Detector")
    if detector_error:
        st.error("Not available", icon="⚠️")
    else:
        st.success("Hybrid detector ready", icon="✅")
        info = get_detector_info()
        if info.get("freeze_verified"):
            st.caption(f"Matches the evaluated freeze `{info['freeze_sha256'][:12]}`.")
        else:
            st.warning("The detector differs from the evaluated freeze; published scores may not apply.")
    st.caption("Rules (email, phone, SSN, URL, card blocks) + spaCy NER (names, addresses, companies), "
               "combined by a fixed merge policy.")
    st.subheader("Privacy")
    st.caption(
        "Text is processed locally by this app. It is kept only in this browser session's memory, "
        "is not logged or saved, and is cleared when you change the input. Reports contain labels and "
        "offsets, not the detected values."
    )
    st.subheader("Limitations")
    st.caption(
        f"Plain text only, up to {workflow.MAX_INPUT_CHARS:,} characters; no PDF or batch processing. "
        "The detector can miss PII, and anything it misses stays in the output. Review sanitized text "
        "before sharing. FinGuard is a research prototype, not a certified privacy product."
    )
    st.caption(f"FinGuard v{finguard.__version__}")

if detector_error:
    st.error(detector_error, icon="⚠️")

# 1. Input -------------------------------------------------------------------
st.markdown('<div class="fg-step">Step 1 · Input</div>', unsafe_allow_html=True)
c1, c2, _ = st.columns([1.3, 0.7, 4])
c1.button("Load synthetic sample", on_click=_load_sample, key="fg_sample", width="stretch")
c2.button("Clear", on_click=_clear, key="fg_clear", width="stretch")

text = st.text_area(
    "Financial text",
    key=INPUT_KEY,
    height=260,
    max_chars=workflow.MAX_INPUT_CHARS,
    placeholder="Paste statement, invoice or form text here…",
)
text = text or ""

scan_clicked = st.button("Scan for PII", type="primary", key="fg_scan", disabled=detector is None)

if scan_clicked:
    try:
        workflow.validate_input(text)
        with st.spinner("Scanning…"):
            record = workflow.scan(detector, text)
        workflow.store_scan(st.session_state, record)
    except workflow.UserFacingError as exc:
        workflow.clear_results(st.session_state)
        st.error(str(exc), icon="⚠️")

scan, anon, stale = workflow.current_results(st.session_state, text)
if stale:
    st.info("The text changed after the last scan, so its results were cleared. Scan again.", icon="ℹ️")

# 2. Detections -----------------------------------------------------------------
if scan is not None:
    st.divider()
    st.markdown('<div class="fg-step">Step 2 · Review detections</div>', unsafe_allow_html=True)
    counts = workflow.counts_by_label(scan.entities)
    total = sum(counts.values())
    st.markdown(f"**{total} entit{'y' if total == 1 else 'ies'} detected** "
                f"· {len(scan.text):,} characters · {scan.seconds:.2f} s")
    labels = list(LABEL_STYLES)
    for row in (labels[:4], labels[4:]):
        cols = st.columns(4)
        for col, label in zip(cols, row):
            col.metric(LABEL_STYLES[label][0], counts[label], border=True)
    if scan.suppressed:
        st.caption(f"{scan.suppressed} overlapping candidate(s) were resolved by the hybrid merge policy.")
    if total == 0:
        st.info("No entities were detected. This does not mean the text contains no PII.", icon="ℹ️")

    tab_text, tab_table = st.tabs(["Highlighted text", "Entity table"])
    with tab_text:
        try:
            st.html(stylesheet() + legend_html() + highlight_html(scan.text, scan.entities))
            st.caption("Highlights use the detector's character offsets in your original text.")
        except ValueError as exc:  # message holds offsets and labels only
            st.error(f"Highlighting is unavailable because the detector output is inconsistent: {exc}.",
                     icon="⚠️")
    with tab_table:
        rows = workflow.entity_rows(scan.entities)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Offsets are 0-based Python character indices; end is exclusive. "
                       "Detected values are not repeated in the table.")
        else:
            st.caption("No rows.")

    # 3. Anonymize --------------------------------------------------------------
    st.divider()
    st.markdown('<div class="fg-step">Step 3 · Anonymize</div>', unsafe_allow_html=True)
    if st.button("Anonymize detected entities", type="primary", key="fg_anonymize"):
        anon = workflow.anonymize_scan(scan)
        workflow.store_anonymization(st.session_state, anon)

    if anon is not None and not anon.verified:
        st.error(anon.error or "Anonymization failed.", icon="⛔")
        st.caption("No sanitized output is shown or offered for download.")
    elif anon is not None:
        result = anon.result
        st.success(
            f"Detected-span replacement verified: all {result.redaction_count} detected span(s) were replaced "
            "with their placeholders, and all other text is unchanged.",
            icon="✅",
        )
        st.warning(
            "This does not guarantee that all PII was removed. Anything the detector missed is still in "
            "the output. Review it before sharing.",
            icon="⚠️",
        )
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Sanitized text**")
            st.code(result.sanitized_text, language=None, wrap_lines=True, height=360)
        with right:
            st.metric("Redactions", result.redaction_count)
            prow = workflow.placeholder_rows(result)
            if prow:
                st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")
            for w in result.warnings:
                st.caption(f"Note: {w}")

        st.markdown("**Downloads**")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Sanitized text (.txt)", exports.sanitized_txt(result), exports.TXT_FILENAME,
                           "text/plain", key="fg_dl_txt", on_click="ignore", width="stretch")
        d2.download_button("Entity report (.csv)", exports.entity_report_csv(result), exports.CSV_FILENAME,
                           "text/csv", key="fg_dl_csv", on_click="ignore", width="stretch")
        d3.download_button("Redaction summary (.json)",
                           exports.redaction_summary_json(result, get_detector_info()),
                           exports.JSON_FILENAME, "application/json", key="fg_dl_json", on_click="ignore",
                           width="stretch")
        st.caption("The CSV and JSON reports list labels, offsets, placeholders and detector sources. "
                   "They do not contain the detected values. Files are generated in memory.")
