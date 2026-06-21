from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_app import FEATURES, MAX_INPUT_TOKENS

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """load_model is @st.cache_resource (process-global); clear between tests so
    each test's own mocked model/tokenizer is the one that gets used."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def fake_tokenizer() -> MagicMock:
    """Mock tokenizer that satisfies truncate_to_tokens and run_feature."""
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "decoded text"
    tok.apply_chat_template.return_value = "PROMPT"
    return tok


@pytest.fixture
def patched_model(fake_tokenizer: MagicMock) -> Iterator[MagicMock]:
    """Patch the mlx_lm boundary so the Run path never loads the real model.

    streamlit_app re-execs `from mlx_lm import generate, load` on every AppTest
    run, so patching mlx_lm.load / mlx_lm.generate makes those imports bind to
    the mocks. Yields the tokenizer so tests can tweak its encode/decode.
    """
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), fake_tokenizer)),
        patch(
            "mlx_lm.generate",
            return_value='{"sentiment": "positive", "confidence": 0.9}',
        ),
    ):
        yield fake_tokenizer


class TestInitialRender:
    """The first paint, before any Run — needs no model."""

    def test_run_disabled_without_input(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.button(key="run").disabled is True
        assert not at.exception

    def test_features_default_on(self) -> None:
        at = AppTest.from_file(APP).run()
        assert len(at.toggle) == 4
        assert all(toggle.value for toggle in at.toggle)
        assert at.toggle(key="feature_summary").label == "Summarization"

    def test_pre_run_prompt_shown(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.session_state["results"] is None
        assert any("click Run" in info.value for info in at.info)

    def test_language_selectbox_defaults_to_match_input(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.selectbox(key="language").value == "Match input"


class TestUIPolish:
    """Material Symbol icons on the tabs and Run button — no model needed."""

    def test_input_tabs_carry_icons(self) -> None:
        at = AppTest.from_file(APP).run()
        labels = {tab.label for tab in at.tabs}
        for label in (
            ":material/edit: Text",
            ":material/upload_file: Upload",
            ":material/dataset: Sample",
        ):
            assert label in labels

    def test_result_tabs_derive_from_features_with_icons(self) -> None:
        at = AppTest.from_file(APP).run()
        labels = {tab.label for tab in at.tabs}
        assert ":material/data_object: JSON" in labels
        # Each feature's result tab label is composed as "<icon> <tab_label>", so
        # this reads both from FEATURES and breaks if the composition regresses.
        for feature in FEATURES:
            assert f"{feature['icon']} {feature['tab_label']}" in labels

    def test_run_button_has_play_icon(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.button(key="run").icon == ":material/play_arrow:"


class TestRunInteraction:
    """The Run path and results panel — model mocked at the mlx_lm boundary."""

    def test_run_enables_once_text_entered(self) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text to analyze.")
        at.run()
        assert at.button(key="run").disabled is False

    def test_run_populates_results(self, patched_model: MagicMock) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Amazing product, I love it!")
        for key in ("feature_summary", "feature_topics", "feature_intents"):
            at.toggle(key=key).set_value(False)  # leave only Sentiment on
        at.run()
        at.button(key="run").click().run()

        assert not at.exception
        assert at.session_state["results"]["order"] == ["sentiment"]
        assert at.metric[0].label == "Sentiment"
        assert at.metric[0].value == ":green[positive]"  # colored by sentiment enum
        assert any("90%" in caption.value for caption in at.caption)

    def test_disabled_feature_shows_not_enabled_note(
        self, patched_model: MagicMock
    ) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        at.toggle(key="feature_summary").set_value(False)
        at.run()
        at.button(key="run").click().run()

        assert any(
            "Summarization was not enabled" in caption.value for caption in at.caption
        )

    def test_inputs_changed_note_after_edit(self, patched_model: MagicMock) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Original text.")
        at.run()
        at.button(key="run").click().run()
        at.text_area(key="paste").set_value("Different text now.")
        at.run()  # edited input, but Run not clicked again

        assert any("Inputs changed" in info.value for info in at.info)

    def test_long_input_truncation_warning(
        self, patched_model: MagicMock, fake_tokenizer: MagicMock
    ) -> None:
        fake_tokenizer.encode.return_value = list(range(MAX_INPUT_TOKENS + 5))
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Pretend this is very long.")
        at.run()
        at.button(key="run").click().run()

        assert not at.exception
        assert any(str(MAX_INPUT_TOKENS) in warning.value for warning in at.warning)

    def test_language_change_flags_inputs_changed(
        self, patched_model: MagicMock
    ) -> None:
        # Output language is part of the run signature, so changing it after a
        # run should flag the results as stale.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        at.button(key="run").click().run()
        at.selectbox(key="language").set_value("German")
        at.run()  # language changed, but Run not clicked again

        assert any("Inputs changed" in info.value for info in at.info)
