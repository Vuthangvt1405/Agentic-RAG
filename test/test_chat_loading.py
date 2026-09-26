"""Chat loop + loading stage tests (no API calls)."""

import io
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from tool import loading
from tool.loading import stage


def fake_result(answer="mock answer"):
    return {
        "question": "hi",
        "route": "vectorstore",
        "docs": [],
        "grades": [],
        "web_results": [],
        "context": "ctx",
        "answer": answer,
        "hallucination": "grounded",
        "hallucination_reason": "",
        "hallucination_retries": 0,
        "answers_question": "yes",
        "answers_question_reason": "",
        "answers_question_rounds": 0,
    }


# Case 1: one question then quit -> pipeline runs once, loop exits.
with (
    patch("builtins.input", side_effect=["hello", "quit"]),
    patch.object(app, "answer_question", return_value=fake_result()) as mock_pipe,
):
    buf = io.StringIO()
    with redirect_stdout(buf):
        app.run_chat(client="fake-client", knowledge_base=[])
assert mock_pipe.call_count == 1
assert "mock answer" in buf.getvalue() and "Bye." in buf.getvalue()
print("case 1 ok: one question, then quit exits.")

# Case 2: error on first question -> loop continues, second works, quit.
with (
    patch("builtins.input", side_effect=["bad q", "good q", "quit"]),
    patch.object(
        app, "answer_question", side_effect=[RuntimeError("boom"), fake_result("recovered")]
    ) as mock_pipe,
):
    buf = io.StringIO()
    with redirect_stdout(buf):
        app.run_chat(client="fake-client", knowledge_base=[])
assert mock_pipe.call_count == 2 and "recovered" in buf.getvalue()
print("case 2 ok: per-question error does not kill the loop.")

# Case 3: loading stage prints [...] done line with timing (forced TTY).
buf = io.StringIO()
with redirect_stdout(buf):
    with patch.object(buf, "isatty", return_value=True):
        with stage("Routing"):
            pass
out = buf.getvalue()
assert "[...] Routing... done (" in out and out.rstrip().endswith("s)")
print("case 3 ok: stage prints '[...] Routing... done (Xs)'.")

# Case 4: stage silent when not a TTY.
with patch.object(sys.stdout, "isatty", return_value=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        with stage("Routing"):
            pass
assert buf.getvalue() == ""
print("case 4 ok: stage silent when piped.")

assert loading.VERBOSE is True
print("ALL CHAT + LOADING CHECKS PASSED")
