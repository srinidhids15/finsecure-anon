"""FinGuard Streamlit frontend (placeholder).

Run with:  streamlit run app/streamlit_app.py

The UI only calls functions from the ``finguard`` package; it contains no
detection logic of its own. The full interface is built in the frontend phase.
"""

import streamlit as st

import finguard

st.set_page_config(page_title="FinGuard", page_icon="🛡️")
st.title("FinGuard")
st.caption(f"Financial-document PII detection and anonymization · v{finguard.__version__}")
st.info("The interface is under construction. Detection and anonymization are added in later phases.")
