"""Drive frontend/app.py with Streamlit's AppTest and a scripted service.

The runtime builder is patched, so no Ollama or Qdrant is needed. This checks
that the UI wires the conversation service correctly (history, clarification
round trip, details panel); it says nothing about model quality.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import streamlit as st  # noqa: E402

from ai_healthcare.services import runtime as runtime_module  # noqa: E402
from tests.integration.test_conversation_service import Rig  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")


@pytest.fixture
def app():
    st.cache_resource.clear()
    rig = Rig()
    fake = SimpleNamespace(
        service=rig.service,
        config=runtime_module.RuntimeConfig(),
        point_count=40,
    )
    with mock.patch.object(runtime_module, "build_runtime", return_value=fake):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        yield at, rig


def _texts(at):
    out = []
    for msg in at.chat_message:
        out.extend(m.value for m in msg.markdown)
        out.extend(c.value for c in msg.caption)
    return "\n".join(out)


def test_app_starts_with_an_empty_chat(app):
    at, _ = app
    assert not at.exception
    assert len(at.chat_message) == 0
    assert len(at.chat_input) == 1


def test_clarification_round_trip_keeps_history(app):
    at, rig = app

    at.chat_input[0].set_value("I have symptom-X.").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "Để hiểu rõ hơn" in _texts(at)
    assert rig.retriever.queries == []  # gate held the answer back

    at.chat_input[0].set_value("Since yesterday and it is severe.").run()
    assert not at.exception
    assert len(at.chat_message) == 4  # history preserved across reruns
    assert len(rig.retriever.queries) == 1
    assert "Nguồn:" in _texts(at)


def test_new_conversation_button_clears_history(app):
    at, _ = app
    at.chat_input[0].set_value("[info] general question here please").run()
    assert len(at.chat_message) == 2
    at.sidebar.button[0].click().run()
    assert len(at.chat_message) == 0
