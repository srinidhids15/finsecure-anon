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
from finguard.interface import exports, pdf_workflow, workflow
from finguard.pdf import report as pdf_report
from finguard.pdf.validate import DEFAULT_LIMITS as PDF_LIMITS
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
        "Text and PDFs are processed locally by this app. They are kept only in this browser session's "
        "memory, are not logged or saved, and are cleared when you change the input or upload. Reports "
        "contain labels and offsets, not the detected values."
    )
    st.subheader("Limitations")
    st.caption(
        f"Text up to {workflow.MAX_INPUT_CHARS:,} characters, or one text-based PDF (up to "
        f"{PDF_LIMITS.max_bytes // 1_048_576} MB, {PDF_LIMITS.max_pages} pages). No OCR for scanned PDFs, "
        "no batch processing. "
        "The detector can miss PII, and anything it misses stays in the output. Review sanitized text "
        "before sharing. FinGuard is a research prototype, not a certified privacy product."
    )
    st.caption(f"FinGuard v{finguard.__version__}")

if detector_error:
    st.error(detector_error, icon="⚠️")

tab_text, tab_pdf = st.tabs(["Text", "PDF document"])

with tab_text:
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

    if scan_clicked and detector is not None:
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

        tab_highlight, tab_table = st.tabs(["Highlighted text", "Entity table"])
        with tab_highlight:
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

        result = anon.result if anon is not None and anon.verified else None
        if anon is not None and result is None:
            st.error(anon.error or "Anonymization failed.", icon="⛔")
            st.caption("No sanitized output is shown or offered for download.")
        elif result is not None:
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

# --- PDF workspace -------------------------------------------------------------------

PDF_STATUS_TEXT = {
    "mapping_failed": "Some detections could not be mapped to exact page positions, so the PDF was not redacted.",
    "redaction_failed": "Applying or saving the redactions failed, so no PDF was produced.",
    "verification_failed": "The redacted PDF failed independent verification and was discarded.",
}

with tab_pdf:
    st.markdown('<div class="fg-step">Step 1 · Upload</div>', unsafe_allow_html=True)
    st.caption(
        f"Text-based PDFs only (selectable text), up to {PDF_LIMITS.max_bytes // 1_048_576} MB and "
        f"{PDF_LIMITS.max_pages} pages. Rejected: scanned/image-only pages (no OCR); rotated, invisible, clipped "
        "or image-covered text; fonts without a Unicode mapping; ActualText/alternate text; layered PDFs, "
        "XFA forms and encrypted files."
    )
    uploaded = st.file_uploader("PDF document", type=["pdf"], key="fg_pdf_upload", accept_multiple_files=False)
    pdf_bytes = uploaded.getvalue() if uploaded is not None else None
    current_id = pdf_workflow.upload_id(pdf_bytes) if pdf_bytes else None
    pdf_record, pdf_stale = pdf_workflow.current_pdf(st.session_state, current_id)
    if pdf_stale:
        st.info("The uploaded file changed, so the previous PDF results were cleared.", icon="ℹ️")

    pdf_clicked = st.button("Detect and redact PDF", type="primary", key="fg_pdf_run",
                            disabled=detector is None or not pdf_bytes)
    if pdf_clicked and detector is not None and pdf_bytes:
        with st.spinner("Processing PDF…"):
            pdf_record = pdf_workflow.run_pdf(detector, pdf_bytes)
        pdf_workflow.store_pdf(st.session_state, pdf_record)

    pdf_result = pdf_record.result if pdf_record is not None and not pdf_record.error else None
    if pdf_record is not None and pdf_result is None:
        st.error(pdf_record.error or "PDF processing failed. No output was produced.", icon="⛔")
        st.caption("No redacted PDF was produced.")
    elif pdf_result is not None:
        result = pdf_result
        # Set only when the whole pipeline, including independent verification, succeeded.
        verified_pdf = result.output if result.verified else None
        st.divider()
        st.markdown('<div class="fg-step">Step 2 · Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pages", len(result.pages), border=True)
        m2.metric("Entities detected", len(result.entities), border=True)
        m3.metric("Regions redacted", result.regions_applied if result.verified else 0, border=True)
        m4.metric("Hidden items removed", sum(
            v for v in (result.sanitization.to_dict() if result.sanitization else {}).values()
            if isinstance(v, int) and not isinstance(v, bool)), border=True)

        v = result.verification
        if verified_pdf is not None and v is not None:
            st.success(f"Redaction applied: {v.regions_checked} region(s) were physically removed from the "
                       "page content and filled black.", icon="✅")
            st.success(f"Targeted text removal verified: the saved PDF was reopened; no text remains in any "
                       f"redacted region, and all {v.glyphs_preserved} unrelated characters are unchanged.",
                       icon="✅")
            st.warning("Complete PII removal is not guaranteed. PII the detector missed, and any text inside "
                       "images, is still in the PDF. Review it before sharing.", icon="⚠️")
        else:
            st.error(PDF_STATUS_TEXT.get(result.status, "The PDF was not redacted."), icon="⛔")
            for problem in result.problems[:20]:
                st.caption(f"• {problem}")
            st.caption("No redacted PDF is offered for download.")

        counts = pdf_report.counts_by_label(result)
        labels = list(LABEL_STYLES)
        for row in (labels[:4], labels[4:]):
            cols = st.columns(4)
            for col, label in zip(cols, row):
                col.metric(LABEL_STYLES[label][0], counts[label])

        t_ent, t_pages, t_hidden, t_preview = st.tabs(["Entities", "Pages", "Hidden content", "Preview"])
        with t_ent:
            rows = pdf_report.entity_rows(result)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Offsets are 0-based positions in that page's extracted text (end exclusive). "
                           "Detected values are not shown.")
            else:
                st.caption("No entities were detected. This does not mean the PDF contains no PII.")
        with t_pages:
            st.dataframe(pd.DataFrame([
                {k: (", ".join(v) if isinstance(v, list) else v) for k, v in p.diagnostics().items()}
                for p in result.pages]), hide_index=True, width="stretch")
        with t_hidden:
            found = result.hidden.to_dict()
            st.dataframe(pd.DataFrame([{"item": k.replace("_", " "),
                                        "found": ("yes" if v else "no") if isinstance(v, bool) else str(v)}
                                       for k, v in found.items()]),
                         hide_index=True, width="stretch")
            st.caption("Annotations, links, form fields, metadata, XMP, embedded files, JavaScript, automatic "
                       "actions, outlines and the structure tree are removed from the output; their contents "
                       "were not scanned.")
        with t_preview:
            if verified_pdf is not None:
                page_no = st.selectbox("Page", list(range(1, len(result.pages) + 1)), key="fg_pdf_preview_page")
                st.image(pdf_workflow.preview_png(verified_pdf, page_no), width="stretch")
            else:
                st.caption("No preview: the PDF was not verified.")

        st.markdown("**Downloads**")
        d1, d2, d3 = st.columns(3)
        if verified_pdf is not None:
            d1.download_button("Redacted PDF (.pdf)", verified_pdf, pdf_report.PDF_FILENAME, "application/pdf",
                               key="fg_pdf_dl", on_click="ignore", width="stretch")
        d2.download_button("Entity report (.csv)", pdf_report.entity_report_csv(result), pdf_report.PDF_CSV_FILENAME,
                           "text/csv", key="fg_pdf_dl_csv", on_click="ignore", width="stretch")
        d3.download_button("Processing summary (.json)", pdf_report.summary_json(result), pdf_report.PDF_JSON_FILENAME,
                           "application/json", key="fg_pdf_dl_json", on_click="ignore", width="stretch")
        st.caption("Reports list pages, labels, offsets, regions, sources and statuses, without the detected values. "
                   "The original upload is not modified or stored.")
