from pathlib import Path

import pytest


def _render_single_case_page():
    import streamlit as st

    from app.views.single_case import render

    st.session_state["display_mode"] = "临床展示"
    render()


def _render_batch_audit_page():
    import streamlit as st

    from app.views.batch_audit import render

    st.session_state["display_mode"] = "临床展示"
    render()


def _render_transfer_sandbox_page():
    import streamlit as st

    from app.views.transfer_sandbox import render

    st.session_state["display_mode"] = "临床展示"
    render()


def test_default_home_page_renders_without_uncaught_exception():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as exc:
        pytest.fail(f"当前 Streamlit 不支持 streamlit.testing.v1.AppTest: {exc}")

    app_path = Path(__file__).resolve().parents[1] / "app" / "demo.py"
    app = AppTest.from_file(str(app_path), default_timeout=20)
    app.run()

    assert not app.exception
    rendered_text = "\n".join(
        element.value
        for element in [*app.title, *app.header, *app.markdown, *app.caption]
        if isinstance(getattr(element, "value", None), str)
    )
    assert "OphAgent Audit Demo" in rendered_text
    assert "科研证据快照 v0.7.2" in rendered_text


@pytest.mark.parametrize(
    "page_renderer",
    [
        _render_single_case_page,
        _render_batch_audit_page,
        _render_transfer_sandbox_page,
    ],
)
def test_key_incremental_pages_render_without_uncaught_exception(page_renderer):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(page_renderer, default_timeout=20)
    app.run()

    assert not app.exception
