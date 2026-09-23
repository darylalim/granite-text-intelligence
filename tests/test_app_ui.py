import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.elements.spinner import SpinnerMixin
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    FEATURES,
    JSON_TAB_LABEL,
    MAX_INPUT_TOKENS,
    MODEL_NAME,
    SAMPLE_TEXTS,
)

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """load_model is @st.cache_resource (process-global); clear between tests so
    each test's own mocked model/tokenizer is the one that gets used."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _refuse_to_wait(real_spinner: Any) -> Any:
    """`st.spinner`, except that the wait for the inference lock raises.

    A Run only waits when the lock is held, and inside AppTest — one script
    at a time — nothing else can hold it, so a wait means a run leaked it.
    Waiting would hang the test, so the spinner that opens the wait raises
    instead, inside the run guard: a leak surfaces as an exception on the
    page and a run that stored nothing.
    """

    def spinner(text: str, *args: Any, **kwargs: Any) -> Any:
        if text.startswith("Waiting for another run"):
            raise AssertionError("the inference lock was still held")
        return real_spinner(text, *args, **kwargs)

    return spinner


@pytest.fixture(autouse=True)
def _no_lock_waits() -> Iterator[None]:
    """Fail on a leaked inference lock in *every* test, instead of hanging.

    Not only in TestInferenceLock: any test that clicks Run twice runs into
    a leak first, and there the second run blocks in lock.acquire() while
    AppTest's timeout joins a script thread that never returns — the suite
    hangs until CI's job timeout without naming a test.
    """
    with patch("streamlit.spinner", side_effect=_refuse_to_wait(st.spinner)):
        yield


_TOPICS_SYSTEM = next(f["system"] for f in FEATURES if f["key"] == "topics")
# The second row has no confidence: a promoted / truncated topic must render
# as an empty bar, not raise, and that column of one float and one NaN is the
# shape the pyarrow path has to accept (see test_topics_tab_renders_a_dataframe).
TOPICS_PAYLOAD = (
    '{"topics": [{"label": "earbuds", "confidence": 0.9}, {"label": "battery"}]}'
)
SENTIMENT_PAYLOAD = '{"sentiment": "positive", "confidence": 0.9}'

# 0xe9 is latin-1 "é" and not valid UTF-8 on its own: the upload decode must
# replace it with U+FFFD rather than raise, since a strict decode puts a
# UnicodeDecodeError traceback where the Run row and results panel should be.
UPLOAD = ("notes.txt", b"caf\xe9 uploaded body", "text/plain")
UPLOAD_TEXT = "caf\ufffd uploaded body"


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

    def test_run_enabled_without_input_and_a_click_only_warns(self) -> None:
        # Run is not disabled for missing input: st.text_area commits only on
        # blur, so a Run disabled on the committed value swallowed the first
        # click after a paste (a browser-only effect — AppTest's set_value
        # commits at once). A click with nothing to analyze warns instead, and
        # loads nothing, generates nothing and stores nothing.
        with (
            patch("mlx_lm.load") as load,
            patch("mlx_lm.generate") as generate,
        ):
            at = AppTest.from_file(APP).run()
            assert at.button(key="run").disabled is False
            at.button(key="run").click().run()
        assert not at.exception
        # Among the notices, whose st.empty() clears as the next rerun starts,
        # and not in the status slot: a container keeps a child until the end
        # of the next run, so the paste-and-click that follows showed the
        # warning, faded, for the whole of that run.
        assert [n for n in _notices(at) if "Nothing to analyze" in n]
        assert not _results_panel(at).children[0].children
        assert at.session_state["results"] is None
        load.assert_not_called()
        generate.assert_not_called()

    def test_features_default_on(self) -> None:
        at = AppTest.from_file(APP).run()
        assert len(at.toggle) == 4
        assert all(toggle.value for toggle in at.toggle)
        assert at.toggle(key="feature_summary").label == "Summarization"

    def test_every_result_tab_carries_the_full_pre_run_prompt(self) -> None:
        # Per tab, not any() over the page: the tab the reader lands on must
        # carry the whole instruction, and an any() stayed green while only
        # the JSON tab did and every feature tab said just "Run to see
        # results here."
        at = AppTest.from_file(APP).run()
        assert at.session_state["results"] is None
        for tab in _result_tabs(at):
            [prompt] = [info.value for info in tab.info]
            assert "in the sidebar" in prompt
            assert "click Run" in prompt

    def test_results_open_on_the_first_feature_tab(self) -> None:
        # st.tabs selects the first tab, so the order decides what a Run
        # lands on: a rendered view, with the raw JSON last.
        at = AppTest.from_file(APP).run()
        labels = [tab.label for tab in _result_tabs(at)]
        assert labels == [
            *[f"{f['icon']} {f['tab_label']}" for f in FEATURES],
            JSON_TAB_LABEL,
        ]

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
        assert JSON_TAB_LABEL in labels
        assert JSON_TAB_LABEL.startswith(":material/")
        # Each feature's result tab label is composed as "<icon> <tab_label>", so
        # this reads both from FEATURES and breaks if the composition regresses.
        for feature in FEATURES:
            assert f"{feature['icon']} {feature['tab_label']}" in labels

    def test_run_button_has_play_icon(self) -> None:
        at = AppTest.from_file(APP).run()
        assert at.button(key="run").icon == ":material/play_arrow:"


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
            if JSON_TAB_LABEL in labels:
                return block
    raise AssertionError("results panel not found")


def _result_tabs(at: AppTest) -> list:
    """The results panel's tabs, in order — its last child is the tab block."""
    return list(_results_panel(at).children[2].children.values())


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
    """The order the results panel emits its slots and tabs in.

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


class TestUrlSettings:
    """The run settings and the sample come back from the URL — no model needed.

    A reload opens a new session — and with the file watcher off, a reload is
    how a code edit takes effect — so without `bind="query-params"` every
    reload put the toggles back on and the language back to "Match input".
    AppTest does not write a changed value to the URL (that is the frontend's
    job, checked live), so these seed `at.query_params` the way a reloaded or
    bookmarked address does and read the widgets back. It does reflect the
    drop of an unrecognized value, which the fallback test asserts.
    """

    def test_settings_and_sample_restore_from_the_url(self) -> None:
        at = AppTest.from_file(APP)
        at.query_params["language"] = "German"
        at.query_params["feature_summary"] = "false"
        at.query_params["sample_select"] = "Product review"
        at.run()
        assert not at.exception
        assert at.selectbox(key="language").value == "German"
        assert at.toggle(key="feature_summary").value is False
        assert all(
            at.toggle(key=f"feature_{f['key']}").value
            for f in FEATURES
            if f["key"] != "summary"
        )
        assert at.segmented_control(key="sample_select").value == "Product review"
        # Restored, not merely displayed: the run a click would start uses them.
        n = len(FEATURES)
        expected = f"Sample · {n - 1} of {n} features · Output: German"
        assert [c.value for c in _run_row(at).caption] == [expected]

    @pytest.mark.parametrize(
        "key, value",
        [
            pytest.param("language", "Klingon", id="unknown-language"),
            pytest.param("feature_summary", "maybe", id="non-boolean-toggle"),
            pytest.param("sample_select", "Nope", id="unknown-sample"),
        ],
    )
    def test_an_unrecognized_value_falls_back_to_the_default(
        self, key: str, value: str
    ) -> None:
        # A hand-edited or stale link (a renamed sample, a mistyped language)
        # must not raise: the value is dropped from the URL and the widget
        # keeps its default.
        at = AppTest.from_file(APP)
        at.query_params[key] = value
        at.run()
        assert not at.exception
        assert key not in at.query_params
        assert at.selectbox(key="language").value == "Match input"
        assert all(toggle.value for toggle in at.toggle)
        assert at.segmented_control(key="sample_select").value is None

    def test_the_pasted_text_is_not_read_from_the_url(self) -> None:
        # The paste is deliberately unbound: bound, the user's text would sit
        # in the address bar and in browser history.
        at = AppTest.from_file(APP)
        at.query_params["paste"] = "Text from a link."
        at.run()
        assert not at.exception
        assert at.text_area(key="paste").value == ""


class TestRunRow:
    """The caption beside Run: why it is disabled, what to do before clicking,
    or what a click will run.

    A greyed-out button says nothing about why; with the toggles and language
    in a sidebar that may be collapsed, the caption is the one place the main
    area states what a click would do. One caption on every path keeps the
    row's own shape constant — button, then one line beside it — so the
    explanation is always next to the control it explains. (The row's
    contents cannot move the results panel below it; that index is pinned by
    `TestResultsPanelStructure`.)
    """

    def test_no_input_names_the_input_sources(self) -> None:
        # An instruction, not a reason: Run stays enabled without input (see
        # TestInitialRender), so the caption says what to do, then to click.
        at = AppTest.from_file(APP).run()
        assert at.button(key="run").disabled is False
        caption = next(c.value for c in _run_row(at).caption)
        for phrase in ("Paste text", "upload a file", "pick a sample", "click Run"):
            assert phrase in caption

    def test_all_features_off_names_the_sidebar(self) -> None:
        # The one disabled state, with or without input — and its reason wins
        # over the no-input instruction, since a disabled button's caption
        # must say why it is disabled.
        at = AppTest.from_file(APP)
        at.run()
        for feature in FEATURES:
            at.toggle(key=f"feature_{feature['key']}").set_value(False)
        at.run()
        self._assert_names_the_sidebar(at)  # no input
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        self._assert_names_the_sidebar(at)  # input

    @staticmethod
    def _assert_names_the_sidebar(at: AppTest) -> None:
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
        expected = f"Pasted text · {n - 1} of {n} features · Output: German"
        assert [c.value for c in _run_row(at).caption] == [expected]
        # A second data point, so a hardcoded count cannot pass.
        at.toggle(key="feature_topics").set_value(False)
        at.run()
        expected = f"Pasted text · {n - 2} of {n} features · Output: German"
        assert [c.value for c in _run_row(at).caption] == [expected]

    @pytest.mark.parametrize(
        "sources, named",
        [
            pytest.param(("sample",), "Sample", id="sample"),
            pytest.param(("sample", "upload"), "Uploaded file", id="upload"),
            pytest.param(("sample", "upload", "paste"), "Pasted text", id="paste"),
        ],
    )
    def test_enabled_caption_names_the_source_run_will_analyze(
        self, sources: tuple[str, ...], named: str
    ) -> None:
        # The precedence is otherwise invisible — a sample on screen while
        # pasted text wins looks exactly like the input. Each case adds the
        # next-higher source, so the caption must follow the winner, not the
        # first source that was set.
        at = AppTest.from_file(APP)
        at.run()
        if "sample" in sources:
            at.segmented_control(key="sample_select").set_value("Product review")
        if "upload" in sources:
            at.file_uploader(key="upload").set_value(UPLOAD)
        if "paste" in sources:
            at.text_area(key="paste").set_value("Some text.")
        at.run()
        n = len(FEATURES)
        expected = f"{named} · {n} of {n} features · Output: Match input"
        assert [c.value for c in _run_row(at).caption] == [expected]

    @staticmethod
    def _assert_row_shape(at: AppTest) -> None:
        # Checked immediately after each run: `at` is one mutable object, so
        # collecting it per state and asserting afterwards would test the last
        # state three times (which is how a first version of this test passed
        # with both disabled-state captions removed). The proto import is local
        # on purpose: it is a private Streamlit path used by this one check, so
        # an upstream rename fails this test rather than the whole module.
        from streamlit.proto.Block_pb2 import Block

        row = _run_row(at)
        assert [c.type for c in row.children.values()] == ["button", "caption"]
        assert (
            row.proto.flex_container.direction
            == Block.FlexContainer.Direction.HORIZONTAL
        )

    def test_row_is_button_then_caption_in_every_state(self) -> None:
        # [button, caption] in all three states, in a *horizontal* container —
        # a vertical one has the same node kinds and would drop the caption
        # under the button on every viewport.
        at = AppTest.from_file(APP)
        at.run()
        self._assert_row_shape(at)  # no input
        at.text_area(key="paste").set_value("Some text.")
        for feature in FEATURES:
            at.toggle(key=f"feature_{feature['key']}").set_value(False)
        at.run()
        self._assert_row_shape(at)  # input, every feature off
        at.toggle(key="feature_sentiment").set_value(True)
        at.run()
        self._assert_row_shape(at)  # enabled


UPLOAD_TAB = ":material/upload_file: Upload"
SAMPLE_TAB = ":material/dataset: Sample"


def _tab_captions(at: AppTest, label: str) -> list[str]:
    [tab] = [t for t in at.tabs if t.label == label]
    return [c.value for c in tab.caption]


def _tab_kinds(at: AppTest, label: str) -> list[str]:
    [tab] = [t for t in at.tabs if t.label == label]
    return [child.type for child in tab.children.values()]


class TestOutrankedSources:
    """A tab whose content Run will *not* analyze says so — no model needed.

    Text > Upload > Sample is resolved silently, so a previewed sample or file
    that a higher source outranks looks exactly like the input. The note names
    the winner and how to clear it; the Run caption (`TestRunRow`) names the
    winner too, and both read it from `resolve_input` rather than re-deriving
    the precedence.
    """

    def test_paste_outranks_upload_and_sample(self) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.file_uploader(key="upload").set_value(UPLOAD)
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        assert _tab_captions(at, UPLOAD_TAB) == [
            (
                "Run analyzes the pasted text, not this file — "
                "clear the Text tab to use it."
            )
        ]
        assert _tab_captions(at, SAMPLE_TAB) == [
            (
                "Run analyzes the pasted text, not this sample — "
                "clear the Text tab to use it."
            )
        ]
        # The outranked previews still render, and the note sits between the
        # picker and the preview it qualifies — the reason the previews are
        # emitted in a second pass, after resolve_input.
        assert _tab_kinds(at, UPLOAD_TAB) == [
            "file_uploader",
            "caption",
            "flex_container",
        ]
        assert _tab_kinds(at, SAMPLE_TAB) == [
            "button_group",
            "caption",
            "flex_container",
        ]
        assert any(text.value == UPLOAD_TEXT for text in at.text)

    def test_upload_outranks_sample(self) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.file_uploader(key="upload").set_value(UPLOAD)
        at.run()
        assert _tab_captions(at, UPLOAD_TAB) == []  # the winner carries no note
        assert _tab_captions(at, SAMPLE_TAB) == [
            (
                "Run analyzes the uploaded file, not this sample — "
                "remove the file in the Upload tab to use it."
            )
        ]

    def test_no_note_once_the_higher_source_is_cleared(self) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        assert len(_tab_captions(at, SAMPLE_TAB)) == 1
        at.text_area(key="paste").set_value("   ")  # blank falls through
        at.run()
        assert _tab_captions(at, SAMPLE_TAB) == []
        assert not at.exception

    def test_blank_upload_is_not_outranked(self) -> None:
        # A whitespace-only file loses to the sample *below* it, since
        # resolve_input skips blank candidates. That is not being outranked,
        # so no note — and the note's clear-this lookup, which has no entry
        # for the sample, is never reached.
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.file_uploader(key="upload").set_value(("blank.txt", b" \n ", "text/plain"))
        at.run()
        assert not at.exception
        assert _tab_captions(at, UPLOAD_TAB) == []
        assert next(c.value for c in _run_row(at).caption).startswith("Sample · ")


class TestRunInteraction:
    """The Run path and results panel — model mocked at the mlx_lm boundary."""

    def test_paste_and_click_in_one_rerun_runs_the_paste(
        self, patched_model: MagicMock
    ) -> None:
        # In a browser the click on Run is what blurs a just-pasted text area,
        # and the blur's commit and the click arrive in the same rerun. Setting
        # the value and clicking with no run in between is that rerun: the
        # pasted text must be what one click analyzes.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text to analyze.")
        at.button(key="run").click().run()
        assert not at.exception
        assert at.session_state["results"]["signature"][0] == "Some text to analyze."

    def test_empty_click_after_a_run_keeps_the_results(
        self, patched_model: MagicMock
    ) -> None:
        # The no-input warning touches nothing else: the previous run's
        # results stand (flagged stale, since the input changed) and the
        # panel keeps its shape.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        at.button(key="run").click().run()
        before = at.session_state["results"]
        at.text_area(key="paste").set_value("")
        at.button(key="run").click().run()
        assert not at.exception
        assert at.session_state["results"] == before
        panel = _results_panel(at)
        assert [c.type for c in panel.children.values()] == [
            "flex_container",
            "flex_container",
            "tab_container",
        ]
        notices = _notices(at)
        assert [n for n in notices if "Nothing to analyze" in n]
        assert [n for n in notices if "Inputs changed" in n]

    def test_run_loads_the_configured_model(self, fake_tokenizer: MagicMock) -> None:
        # The call site passes MODEL_NAME — the id the sidebar caption names
        # and the model cache is keyed on (TestLoadModel pins the keying).
        with (
            patch("mlx_lm.load", return_value=(MagicMock(), fake_tokenizer)) as load,
            patch("mlx_lm.generate", side_effect=_fake_generate),
        ):
            at = AppTest.from_file(APP)
            at.run()
            at.text_area(key="paste").set_value("Some text.")
            at.button(key="run").click().run()
        assert not at.exception
        load.assert_called_once_with(MODEL_NAME)

    def test_cold_load_shows_one_timed_spinner(self, patched_model: MagicMock) -> None:
        # The autouse fixture clears the cache, so this Run is a cold load.
        # Two spies, because the two spinners arrive by different routes: the
        # app's `st.spinner` is a method bound at import, which a class patch
        # cannot reach, while the cache opens its own through the class
        # (`main_dg.spinner(..., _cache=True)`), which the module patch
        # cannot see. Keying on `_cache` keeps the second assertion true
        # whichever route a future Streamlit gives the app's call.
        real_spinner = st.spinner
        with (
            patch("streamlit.spinner", wraps=real_spinner) as app_spinner,
            patch.object(
                SpinnerMixin, "spinner", autospec=True, side_effect=SpinnerMixin.spinner
            ) as any_spinner,
        ):
            at = AppTest.from_file(APP)
            at.run()
            at.text_area(key="paste").set_value("Some text.")
            at.button(key="run").click().run()
        assert not at.exception
        [loading] = [
            c for c in app_spinner.call_args_list if c.args[0] == "Loading model…"
        ]
        # Timed, since a cold load or first download can take minutes.
        assert loading.kwargs.get("show_time") is True
        # And alone: no cached function opens a spinner of its own. The
        # model's default, "Running `load_model(...)`.", would stack under it
        # for the whole load.
        assert not [c for c in any_spinner.call_args_list if c.kwargs.get("_cache")]

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
        # Only the two configured columns are shown, whatever else a row holds.
        assert list(at.dataframe[0].proto.column_order) == ["label", "confidence"]

    def test_each_result_lands_in_its_labelled_tab(
        self, patched_model: MagicMock
    ) -> None:
        # The labels and the tab handles are paired by position, so a label
        # order and an index that disagree (JSON labelled last but filled
        # through tabs[0]) put every result under the wrong heading — which
        # the pre-run tests cannot see, since every tab then says the same.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Great earbuds, battery lasts all day.")
        at.button(key="run").click().run()
        assert not at.exception
        tabs = {tab.label: tab for tab in _result_tabs(at)}
        by_key = {f["key"]: tabs[f"{f['icon']} {f['tab_label']}"] for f in FEATURES}
        assert len(tabs[JSON_TAB_LABEL].json) == 1
        assert len(by_key["topics"].dataframe) == 1
        assert [m.label for m in by_key["sentiment"].metric] == ["Sentiment"]
        assert [s.value for s in by_key["intents"].caption][:1] == ["Intent"]
        assert len(by_key["summary"].text) == 1

    def test_sample_selection_feeds_the_run(self, patched_model: MagicMock) -> None:
        # Selecting a built-in sample resolves as the input (precedence falls
        # through to it), the Run caption says so, and a click produces results.
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.run()
        assert next(c.value for c in _run_row(at).caption).startswith("Sample · ")
        at.button(key="run").click().run()
        assert not at.exception
        # The picked sample is what actually fed the run — signature[0] is the
        # resolved input_text — not merely "a run happened".
        assert (
            at.session_state["results"]["signature"][0]
            == SAMPLE_TEXTS["Product review"]
        )

    def test_upload_feeds_the_run(self, patched_model: MagicMock) -> None:
        # The Upload source end to end: the decode, the preview, the Run
        # caption naming it, and the uploaded text reaching the run.
        at = AppTest.from_file(APP)
        at.run()
        at.file_uploader(key="upload").set_value(UPLOAD)
        at.run()
        assert not at.exception
        caption = next(c.value for c in _run_row(at).caption)
        assert caption.startswith("Uploaded file · ")
        assert any(text.value == UPLOAD_TEXT for text in at.text)  # the preview
        at.button(key="run").click().run()
        assert not at.exception
        assert at.session_state["results"]["signature"][0] == UPLOAD_TEXT

    def test_input_precedence_at_the_call_site(self, patched_model: MagicMock) -> None:
        # TestResolveInput pins the function's own precedence; this pins the
        # argument order the script passes it — Text > Upload > Sample. Each
        # step adds the next-higher source, so swapping any two arguments fails
        # one of the two assertions.
        at = AppTest.from_file(APP)
        at.run()
        at.segmented_control(key="sample_select").set_value("Product review")
        at.file_uploader(key="upload").set_value(UPLOAD)
        at.run()
        at.button(key="run").click().run()
        assert at.session_state["results"]["signature"][0] == UPLOAD_TEXT
        at.text_area(key="paste").set_value("Pasted text wins.")
        at.run()
        at.button(key="run").click().run()
        assert at.session_state["results"]["signature"][0] == "Pasted text wins."

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
        # Thousands-separated, the same rendering as the sidebar's cap caption.
        assert any(f"{MAX_INPUT_TOKENS:,}" in warning.value for warning in at.warning)

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


def _stop_mid_run(*_: object, **__: object) -> str:
    """Stop the run from inside generate, as a superseding rerun does.

    st.stop() requests a stop and enqueues, so StopException — a
    BaseException the run guard's `except Exception` never sees — is raised
    at a yield point in the middle of the run, which is how fastReruns ends
    a run superseded by a click or a settings change.
    """
    st.stop()


class TestInferenceLock:
    """The process-wide inference lock is released however a run ends.

    AppTest runs one script at a time, so the contention the lock exists for
    cannot happen here (it was exercised live, see Performance in CLAUDE.md);
    what can happen here is a run that leaks it, which would hang every later
    run in the process. The autouse _no_lock_waits turns that wait into a
    failure in every test of this file; this class is the named pin.
    """

    @pytest.mark.parametrize(
        "first_generate",
        [
            pytest.param(_fake_generate, id="finished"),
            pytest.param(RuntimeError("metal OOM"), id="failed"),
            pytest.param(_stop_mid_run, id="stopped"),
        ],
    )
    def test_the_next_run_does_not_wait(
        self, patched_model: MagicMock, first_generate: object
    ) -> None:
        # A wait raises rather than hangs: see _no_lock_waits.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        with patch("mlx_lm.generate", side_effect=first_generate):
            at.button(key="run").click().run()
        at.button(key="run").click().run()
        assert not at.exception
        assert at.session_state["results"] is not None


STOPPED_NOTE = "The last run was stopped"
PANEL_AFTER_A_NOTICE = ["flex_container", "flex_container", "tab_container"]


def _notices(at: AppTest) -> list[str]:
    """Every notice in the results panel's reserved notices slot.

    The slot is an `empty` until something fills it, and an empty has none.
    """
    slot = _results_panel(at).children[1]
    if slot.type == "empty":
        return []
    return [n.value for n in [*slot.info, *slot.warning]]


class TestInterruptedRun:
    """A run stopped mid-way by a widget change says so on the next rerun.

    Under runner.fastReruns a change during a run stops it with StopException
    at its next st.* call and reruns at once; the run's finished features go
    with it, and before this note a first run simply fell back to the pre-run
    prompts. `run_pending` is set when a run starts and cleared only when it
    stores its results or fails, so a stopped run leaves it set.
    """

    def test_seeded_flag_shows_the_note_in_the_notices_slot(self) -> None:
        # No run and no results: the note alone must still land in the
        # reserved slot, turning the `empty` into a container in place.
        at = AppTest.from_file(APP)
        at.session_state["run_pending"] = True
        at.run()
        assert not at.exception
        panel = _results_panel(at)
        assert [c.type for c in panel.children.values()] == PANEL_AFTER_A_NOTICE
        assert [n for n in _notices(at) if STOPPED_NOTE in n]

    def test_a_stopped_run_is_noted_beside_the_stale_results(
        self, patched_model: MagicMock
    ) -> None:
        # Both notes at once: st.empty() holds one element, so the stopped
        # note and the "Inputs changed" note must share one container — a
        # second notice_slot.container() call would replace the first.
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Original text.")
        at.button(key="run").click().run()
        before = at.session_state["results"]
        at.text_area(key="paste").set_value("Different text now.")
        with patch("mlx_lm.generate", side_effect=_stop_mid_run):
            at.button(key="run").click().run()
        assert at.session_state["run_pending"] is True
        assert at.session_state["results"] == before  # nothing new was stored
        at.run()  # the rerun the stopping change asked for
        panel = _results_panel(at)
        assert [c.type for c in panel.children.values()] == PANEL_AFTER_A_NOTICE
        notices = _notices(at)
        assert [n for n in notices if STOPPED_NOTE in n]
        assert [n for n in notices if "Inputs changed" in n]

    @pytest.mark.parametrize(
        "generate",
        [
            pytest.param(_fake_generate, id="finished"),
            pytest.param(RuntimeError("metal OOM"), id="failed"),
        ],
    )
    def test_a_run_that_ends_clears_the_flag(
        self, patched_model: MagicMock, generate: object
    ) -> None:
        # A failure is reported by the traceback, not as an interruption.
        at = AppTest.from_file(APP)
        at.session_state["run_pending"] = True  # an earlier run was stopped
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        with patch("mlx_lm.generate", side_effect=generate):
            at.button(key="run").click().run()
        assert at.session_state["run_pending"] is False
        at.run()
        assert not [n for n in _notices(at) if STOPPED_NOTE in n]

    def test_an_empty_click_keeps_the_note(self) -> None:
        # A click with no input starts nothing, so it says nothing about the
        # last run: the stopped note stays beside the no-input warning.
        with patch("mlx_lm.generate") as generate:
            at = AppTest.from_file(APP)
            at.session_state["run_pending"] = True
            at.run()
            at.button(key="run").click().run()
        generate.assert_not_called()
        assert at.session_state["run_pending"] is True
        notices = _notices(at)
        assert [n for n in notices if STOPPED_NOTE in n]
        assert [n for n in notices if "Nothing to analyze" in n]


class TestRunFailures:
    """What the page shows when a run, or the model's output, goes wrong.

    Both behaviours are argued in the app (the run guard's comment, and the
    `st.code` rationale in `render_result`) and neither was executed by any
    test: nothing made the mocked model raise or return unparseable output.
    """

    def test_failed_run_clears_results_and_shows_the_error(
        self, patched_model: MagicMock
    ) -> None:
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Some text.")
        at.run()
        at.button(key="run").click().run()
        assert at.session_state["results"] is not None
        assert len(at.metric) > 0  # the good run's answers are on screen
        # load_model stays cached from the good run, so only generate fails.
        with patch("mlx_lm.generate", side_effect=RuntimeError("metal OOM")):
            at.button(key="run").click().run()
        # The previous answers are dropped rather than left under a traceback
        # with nothing marking them stale, and the error itself is shown.
        assert at.session_state["results"] is None
        assert len(at.metric) == 0
        assert any("metal OOM" in exc.value for exc in at.exception)

    def test_unparseable_output_shows_the_raw_response(
        self, patched_model: MagicMock
    ) -> None:
        raw = "I think it is positive."
        at = AppTest.from_file(APP)
        at.run()
        at.text_area(key="paste").set_value("Amazing product, I love it!")
        for key in ("feature_summary", "feature_topics", "feature_intents"):
            at.toggle(key=key).set_value(False)  # leave only Sentiment on
        at.run()
        with patch("mlx_lm.generate", return_value=raw):
            at.button(key="run").click().run()

        assert not at.exception
        assert at.session_state["results"]["data"]["sentiment"]["parsed"] is None
        assert any("Could not parse" in warning.value for warning in at.warning)
        [code] = at.code
        assert code.value == raw
        # language=None serializes as "plaintext"; st.code's default would be
        # "python", syntax-highlighting model prose as code.
        assert code.language == "plaintext"
        assert code.proto.wrap_lines
        # The JSON tab falls back to the raw string rather than a null.
        assert json.loads(at.json[0].value) == {"sentiment": raw}
