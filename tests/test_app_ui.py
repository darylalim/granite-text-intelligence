from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.proto.Block_pb2 import Block
from streamlit.testing.v1 import AppTest

from streamlit_app import FEATURES, MAX_INPUT_TOKENS, SAMPLE_TEXTS

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """load_model is @st.cache_resource (process-global); clear between tests so
    each test's own mocked model/tokenizer is the one that gets used."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


_TOPICS_SYSTEM = next(f["system"] for f in FEATURES if f["key"] == "topics")
# The second row has no confidence: a promoted / truncated topic must render
# as an empty bar, not raise, and that column of one float and one NaN is the
# shape the pyarrow path has to accept (see test_topics_tab_renders_a_dataframe).
TOPICS_PAYLOAD = (
    '{"topics": [{"label": "earbuds", "confidence": 0.9}, {"label": "battery"}]}'
)
SENTIMENT_PAYLOAD = '{"sentiment": "positive", "confidence": 0.9}'


@pytest.fixture
def fake_tokenizer() -> MagicMock:
    """Mock tokenizer that satisfies truncate_to_tokens and run_feature.

    apply_chat_template hands back the system message as the "rendered" prompt,
    so the generate mock below can tell which feature is asking.
    """
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "decoded text"
    tok.apply_chat_template.side_effect = lambda messages, **_: messages[0]["content"]
    return tok


def _fake_generate(_model: object, _tokenizer: object, prompt: str, **_: object) -> str:
    """Topics gets a topics payload; every other feature gets the sentiment one.

    A single fixed payload left the Topics tab rendering "No topics found." in
    every test, so st.dataframe — the app's one trip through pyarrow — was
    never executed by the gate.
    """
    return TOPICS_PAYLOAD if prompt == _TOPICS_SYSTEM else SENTIMENT_PAYLOAD


@pytest.fixture
def patched_model(fake_tokenizer: MagicMock) -> Iterator[MagicMock]:
    """Patch the mlx_lm boundary so the Run path never loads the real model.

    streamlit_app re-execs `from mlx_lm import generate, load` on every AppTest
    run, so patching mlx_lm.load / mlx_lm.generate makes those imports bind to
    the mocks. Yields the tokenizer so tests can tweak its encode/decode.
    """
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), fake_tokenizer)),
        patch("mlx_lm.generate", side_effect=_fake_generate),
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
        assert any(
            "in the sidebar" in info.value and "click Run" in info.value
            for info in at.info
        )

    def test_language_selectbox_defaults_to_match_input(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.selectbox(key="language").value == "Match input"

    def test_sample_picker_is_segmented_control(self) -> None:
        # The sample picker is a segmented_control (all options visible, native
        # None empty-state), not a selectbox with a "—" sentinel.
        at = AppTest.from_file(APP).run()
        sample = at.segmented_control(key="sample_select")
        assert sample.value is None  # nothing selected by default
        assert list(sample.options) == list(SAMPLE_TEXTS)
        assert len(at.selectbox) == 1  # only the output-language dropdown remains


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


JSON_TAB = ":material/data_object: JSON"
TOP_LEVEL_KINDS = ["title", "tab_container", "flex_container", "flex_container"]


def _results_panel(at: AppTest):
    """The wrapper container holding the result tabs, pre- or post-run.

    Found by content — the top-level block of `at.main` whose `tab_container`
    child carries the JSON tab — rather than by index or by the notices slot's
    kind: the input tabs are a top-level `tab_container` too (so a wrapper
    someone later puts around *them* must not match), and the notices slot is
    an `empty` only until the first run fills it with a container.
    """
    for block in at.main.children.values():
        for child in getattr(block, "children", {}).values():
            if getattr(child, "type", "") != "tab_container":
                continue
            labels = {getattr(t, "label", None) for t in child.children.values()}
            if JSON_TAB in labels:
                return block
    raise AssertionError("results panel not found")


def _run_row(at: AppTest):
    """The horizontal container holding Run — located by the button, not by
    being the first `flex_container` in main, which a plain `st.container()`
    around the input tabs would also be."""
    for block in at.main.children.values():
        if getattr(block, "type", "") == "flex_container" and any(
            button.key == "run" for button in block.button
        ):
            return block
    raise AssertionError("Run row not found")


class TestResultsPanelStructure:
    """The results panel's emission order, which nothing else pins.

    Both properties degrade silently if someone moves the run block back above
    the tabs: the tab block returns to a shifting delta path (remounting and
    resetting the selected tab whenever a notice appears or disappears), and
    the notices stop clearing at the top of a rerun, so a stale "Inputs changed
    since this run" note stays readable for the length of the next run.
    """

    def test_slots_are_reserved_before_the_tabs(self) -> None:
        at = AppTest.from_file(APP).run()
        kinds = [child.type for child in _results_panel(at).children.values()]
        # status slot, notice slot, then the tabs — the tabs must come last of
        # the three so their index cannot shift, and the notice slot must be an
        # `empty` (a container would preserve the previous run's note). The
        # wrapper must hold nothing else: an unconditional extra sibling would
        # be a fourth kind here (a conditional one is what the post-run test
        # below is for).
        assert kinds == ["flex_container", "empty", "tab_container"]

    def test_panel_is_the_fourth_top_level_block(self) -> None:
        # The panel's own delta path is [main, 3]: fixed only as long as the
        # three blocks above it (title, input tabs, Run row) are unconditional
        # and nothing is emitted between them. A stray top-level element moves
        # the whole panel and remounts it — the finder above would still find
        # the panel, so this is the assertion that sees the move.
        at = AppTest.from_file(APP).run()
        assert [b.type for b in at.main.children.values()] == TOP_LEVEL_KINDS
        assert _results_panel(at) is at.main.children[3]

    def test_notices_fill_the_reserved_slot_after_a_run(
        self, patched_model: MagicMock
    ) -> None:
        # After a run the notices go *into* slot 1 (the empty, now holding a
        # container) and the tabs stay last at index 2. Emitting the notices
        # into a fresh sibling — or any element gated on `results is not None`
        # between the slot and the tabs — passes the pre-run test above and
        # fails here, because both only exist once results do.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Original text.")
        at.run()
        at.button(key="run").click().run()
        panel = _results_panel(at)
        assert [c.type for c in panel.children.values()] == [
            "flex_container",
            "flex_container",
            "tab_container",
        ]
        at.text_area(key="paste").set_value("Different text now.")
        at.run()  # edited input, but Run not clicked again
        panel = _results_panel(at)
        assert [c.type for c in panel.children.values()] == [
            "flex_container",
            "flex_container",
            "tab_container",
        ]
        assert any("Inputs changed" in i.value for i in panel.children[1].info)


class TestSidebarLayout:
    """Settings in the sidebar, the action in the main area — no model needed.

    The toggles and the output language are run *settings*, which is what a
    sidebar is for; Run is the primary action and must never sit behind the
    chevron of a collapsed sidebar (auto-collapsed at narrow widths, and a
    manual collapse persists in localStorage). AppTest's root tree is
    `[main, sidebar, event]`, so the scoped accessors are what make placement
    assertable at all — every `at.<widget>` lookup spans the whole tree.
    """

    def test_settings_live_in_the_sidebar(self) -> None:
        at = AppTest.from_file(APP).run()
        assert len(at.sidebar.toggle) == len(FEATURES)
        assert at.sidebar.selectbox(key="language").value == "Match input"
        assert len(at.main.toggle) == 0
        assert len(at.main.selectbox) == 0

    def test_run_stays_in_the_main_area(self) -> None:
        at = AppTest.from_file(APP).run()
        assert len(at.sidebar.button) == 0
        assert at.main.button(key="run").icon == ":material/play_arrow:"

    def test_run_is_a_primary_button_at_its_natural_width(self) -> None:
        # AppTest's tree cannot see width: the Button node wraps the inner
        # proto and `width_config` lives on the outer Element. So the call is
        # spied instead — `st.button` resolves the module attribute at call
        # time, and the script is re-executed under the patch.
        real_button = st.button
        with patch("streamlit.button", wraps=real_button) as spy:
            at = AppTest.from_file(APP).run()
        assert at.main.button(key="run").proto.type == "primary"
        run_call = next(c for c in spy.call_args_list if c.kwargs.get("key") == "run")
        assert run_call.kwargs.get("width", "content") == "content"


class TestRunRow:
    """The caption beside Run: why it is disabled, or what a click will run.

    A greyed-out button says nothing about why; with the toggles and language
    in a sidebar that may be collapsed, the caption is the one place the main
    area states what a click would do. One caption on every path keeps the
    row's own shape constant — button, then one line beside it — so the
    explanation is always next to the control it explains. (The row's
    contents cannot move the results panel below it; that index is pinned by
    `TestResultsPanelStructure`.)
    """

    def test_no_input_names_the_input_sources(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.button(key="run").disabled is True
        caption = next(c.value for c in _run_row(at).caption)
        for phrase in ("Paste text", "upload a file", "pick a sample"):
            assert phrase in caption

    def test_all_features_off_names_the_sidebar(self) -> None:
        # The one disabled state that has input: nothing covered it before.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        for feature in FEATURES:
            at.toggle(key=f"feature_{feature['key']}").set_value(False)
        at.run()
        assert at.button(key="run").disabled is True
        caption = next(c.value for c in _run_row(at).caption)
        assert "at least one feature" in caption
        assert "in the sidebar" in caption

    def test_enabled_caption_mirrors_the_settings(self) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        at.toggle(key="feature_summary").set_value(False)
        at.selectbox(key="language").set_value("German")
        at.run()
        assert at.button(key="run").disabled is False
        n = len(FEATURES)
        expected = f"{n - 1} of {n} features · Output language: German"
        assert [c.value for c in _run_row(at).caption] == [expected]
        # A second data point, so a hardcoded count cannot pass.
        at.toggle(key="feature_topics").set_value(False)
        at.run()
        expected = f"{n - 2} of {n} features · Output language: German"
        assert [c.value for c in _run_row(at).caption] == [expected]

    def test_row_is_button_then_caption_in_every_state(self) -> None:
        # [button, caption] in all three states, in a *horizontal* container —
        # a vertical one has the same node kinds and would drop the caption
        # under the button on every viewport.
        at = AppTest.from_file(APP)
        at.run()
        states = [at]
        at.text_area(key="paste").set_value("Some text.")
        for feature in FEATURES:
            at.toggle(key=f"feature_{feature['key']}").set_value(False)
        at.run()
        states.append(at)
        at.toggle(key="feature_sentiment").set_value(True)
        at.run()
        states.append(at)
        for state in states:
            row = _run_row(state)
            assert [c.type for c in row.children.values()] == ["button", "caption"]
            assert (
                row.proto.flex_container.direction
                == Block.FlexContainer.Direction.HORIZONTAL
            )


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

    def test_topics_tab_renders_a_dataframe(self, patched_model: MagicMock) -> None:
        # The Topics tab holds the app's only st.dataframe, and so its only
        # trip through pyarrow — the one transitive dependency whose 25.0.0
        # release segfaulted in exactly Streamlit's script-thread pattern
        # (apache/arrow#50471); only streamlit's own `!=25.0.0` marker keeps
        # it out of the lock. Reading `.value` back decodes the Arrow bytes the
        # element carries, so both directions of that path execute here.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Great earbuds, battery lasts all day.")
        at.run()
        at.button(key="run").click().run()

        assert not at.exception
        assert len(at.dataframe) == 1
        frame = at.dataframe[0].value
        assert list(frame["label"]) == ["earbuds", "battery"]
        assert frame["confidence"].tolist()[0] == pytest.approx(0.9)
        # The promoted row's missing confidence is a null, not a raise or a
        # dropped row — that is what renders as the empty progress bar.
        assert frame["confidence"].isna().tolist() == [False, True]

    def test_sample_selection_feeds_the_run(self, patched_model: MagicMock) -> None:
        # Selecting a built-in sample resolves as the input (precedence falls
        # through to it), enabling Run and producing results.
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.run()
        assert at.button(key="run").disabled is False
        at.button(key="run").click().run()
        assert not at.exception
        # The picked sample is what actually fed the run — signature[0] is the
        # resolved input_text — not merely "a run happened".
        assert (
            at.session_state["results"]["signature"][0]
            == SAMPLE_TEXTS["Product review"]
        )

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
