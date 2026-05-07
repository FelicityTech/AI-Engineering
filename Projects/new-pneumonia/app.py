"""
app.py
------
Entry point. Wires the UI layer (ui.py) and backend logic (logic.py) together.
Contains no rendering code and no model code — only orchestration.

Run:
    pip install streamlit tensorflow opencv-python-headless pillow numpy
    streamlit run app.py
"""

import streamlit as st

import ui
from logic import load_model, run_inference


def main() -> None:
    # ── Page config & global styles (must be first Streamlit call) ────────────
    ui.setup_page()

    # ── Sidebar — returns the chosen threshold ────────────────────────────────
    threshold = ui.render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    ui.render_header()

    # ── Model loading (cached across reruns) ──────────────────────────────────
    with st.spinner("Loading model…"):
        model, is_demo = load_model()

    if is_demo:
        ui.render_demo_banner()

    st.divider()

    # ── Main content — two equal columns ─────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = ui.render_upload_panel()

    with right:
        if uploaded:
            ui._section_label("Analysis Result")
            with st.spinner("Analysing X-ray…"):
                try:
                    result = run_inference(model, is_demo, uploaded.getvalue(), threshold)
                except Exception as err:
                    st.error(f"Inference error: {err}")
                    st.stop()

            ui.render_result_card(result)
            st.divider()
            ui.render_metrics_row(result)
        else:
            ui.render_awaiting_panel()

    # ── Clinical note (full width, below columns) ─────────────────────────────
    if uploaded:
        ui.render_clinical_note(result)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    ui.render_how_it_works()
    ui.render_footer()


if __name__ == "__main__":
    main()
