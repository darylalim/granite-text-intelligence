import json
import os
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    _DEFAULT_MAX_INPUT_TOKENS,
    FEATURES,
    LABELS,
    LANGUAGE_AUTO,
    LANGUAGE_ENGLISH,
    MAX_INPUT_TOKENS,
    MODEL_MAX_TOKENS,
    _effective_max_tokens,
    _resolve_max_input_tokens,
    language_directive,
    parse_json_output,
    render_result,
    resolve_input,
    run_feature,
    truncate_to_tokens,
)


class TestFeatures:
    def test_four_features_in_order(self) -> None:
        assert [feature["key"] for feature in FEATURES] == [
            "summary",
            "topics",
            "intents",
            "sentiment",
        ]

    def test_each_feature_has_required_fields(self) -> None:
        for feature in FEATURES:
            for field in (
                "key",
                "label",
                "tab_label",
                "icon",
                "help",
                "output",
                "max_tokens",
                "system",
                "user_template",
            ):
                assert field in feature
            assert "{text}" in feature["user_template"]
            assert feature["output"] in ("prose", "json")
            # Material Symbol shortcode driving the result tab (e.g. ":material/mood:").
            assert feature["icon"].startswith(":material/")
            assert feature["icon"].endswith(":")

    def test_only_summary_is_prose(self) -> None:
        assert FEATURES[0]["output"] == "prose"
        assert all(feature["output"] == "json" for feature in FEATURES[1:])

    def test_labels_match_features(self) -> None:
        assert LABELS == {feature["key"]: feature["label"] for feature in FEATURES}

    def test_json_features_embed_valid_schema(self) -> None:
        for feature in FEATURES:
            if feature["output"] != "json":
                continue
            schema_text = (
                feature["system"].split("<schema>\n", 1)[1].split("\n</schema>", 1)[0]
            )
            schema = json.loads(schema_text)
            assert isinstance(schema, dict)
            assert "properties" in schema
            assert "JSON" in feature["user_template"]

    def test_json_system_follows_ibm_documented_pattern(self) -> None:
        # The JSON features reproduce IBM Granite's documented "answer in JSON …
        # <schema>" system prompt verbatim, including the trailing newline.
        for feature in FEATURES:
            if feature["output"] != "json":
                continue
            system = feature["system"]
            assert system.startswith(
                "You are a helpful assistant that answers in JSON. "
                "Here's the json schema you must adhere to:\n<schema>\n"
            )
            assert system.endswith("\n</schema>\n")


class TestParseJsonOutput:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            pytest.param('{"a": 1}', {"a": 1}, id="plain"),
            pytest.param(
                'Here you go: {"a": 1} done', {"a": 1}, id="embedded-in-prose"
            ),
            pytest.param(
                '```json\n{"sentiment": "positive"}\n```',
                {"sentiment": "positive"},
                id="code-fence",
            ),
            pytest.param(
                'prefix {"a": 1} middle {"b": 2} suffix',
                {"a": 1},
                id="first-of-multiple",
            ),
            pytest.param(
                'I think {maybe} the answer is {"sentiment": "positive"}',
                {"sentiment": "positive"},
                id="recovers-after-stray-braces",
            ),
            pytest.param("not json at all", None, id="not-json"),
            pytest.param("{not: valid}", None, id="malformed-braces"),
            pytest.param("[1, 2, 3]", None, id="top-level-array"),
            pytest.param("true", None, id="scalar-bool"),
            pytest.param("42", None, id="scalar-int"),
            pytest.param('"hello"', None, id="scalar-string"),
        ],
    )
    def test_parse_json_output(self, raw: str, expected: dict | None) -> None:
        assert parse_json_output(raw) == expected


class TestResolveInput:
    @pytest.mark.parametrize(
        "pasted, uploaded, sample, expected",
        [
            pytest.param("typed", "uploaded", "sample", "typed", id="pasted-wins"),
            pytest.param(
                "", "uploaded", "sample", "uploaded", id="upload-when-no-pasted"
            ),
            pytest.param("", "", "sample", "sample", id="sample-when-neither"),
            pytest.param("", "", "", "", id="all-empty"),
            pytest.param("  spaced  ", "", "", "spaced", id="strips-whitespace"),
            pytest.param(
                "   ",
                "uploaded",
                "sample",
                "uploaded",
                id="whitespace-only-falls-through",
            ),
        ],
    )
    def test_resolve_input(
        self, pasted: str, uploaded: str, sample: str, expected: str
    ) -> None:
        assert resolve_input(pasted, uploaded, sample) == expected


class TestTruncateToTokens:
    @pytest.mark.parametrize(
        "num_tokens, expected_truncated",
        [
            pytest.param(10, False, id="short"),
            pytest.param(MAX_INPUT_TOKENS, False, id="boundary-equal"),
            pytest.param(MAX_INPUT_TOKENS + 50, True, id="over-budget"),
        ],
    )
    def test_truncation_decision(
        self, num_tokens: int, expected_truncated: bool
    ) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(num_tokens))
        tokenizer.decode.return_value = "truncated text"

        text, truncated = truncate_to_tokens("original", tokenizer)

        assert truncated is expected_truncated
        if expected_truncated:
            assert text == "truncated text"
        else:
            assert text == "original"
            tokenizer.decode.assert_not_called()

    def test_truncates_to_exact_budget(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_INPUT_TOKENS + 50))
        tokenizer.decode.return_value = "truncated text"

        truncate_to_tokens("long", tokenizer)

        tokenizer.decode.assert_called_once_with(
            list(range(MAX_INPUT_TOKENS)), skip_special_tokens=True
        )

    def test_encode_excludes_special_tokens(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        truncate_to_tokens("hello", tokenizer)

        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)


@pytest.fixture
def tokenizer() -> MagicMock:
    """A mock tokenizer whose chat template renders to a fixed prompt string."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "PROMPT"
    return tok


class TestRunFeature:
    @pytest.mark.parametrize(
        "feature, generated, expected_raw, expected_parsed",
        [
            pytest.param(
                FEATURES[0],
                "  A concise summary.  ",
                "A concise summary.",
                None,
                id="prose-stripped-and-unparsed",
            ),
            pytest.param(
                FEATURES[3],
                '{"sentiment": "positive", "confidence": 0.9}',
                '{"sentiment": "positive", "confidence": 0.9}',
                {"sentiment": "positive", "confidence": 0.9},
                id="json-parsed",
            ),
            pytest.param(
                FEATURES[1],
                "totally not json",
                "totally not json",
                None,
                id="json-unparseable",
            ),
        ],
    )
    @patch("streamlit_app.generate")
    def test_raw_and_parsed(
        self,
        mock_generate: MagicMock,
        feature: dict,
        generated: str,
        expected_raw: str,
        expected_parsed: dict | None,
        tokenizer: MagicMock,
    ) -> None:
        mock_generate.return_value = generated

        result = run_feature(feature, "text", MagicMock(), tokenizer)

        assert result["raw"] == expected_raw
        assert result["parsed"] == expected_parsed

    @patch("streamlit_app.generate")
    def test_applies_chat_template_and_max_tokens(
        self, mock_generate: MagicMock, tokenizer: MagicMock
    ) -> None:
        mock_generate.return_value = "{}"

        # language="English" → no directive and the base (un-enlarged) budget.
        run_feature(
            FEATURES[1], "hello world", MagicMock(), tokenizer, language="English"
        )

        messages = tokenizer.apply_chat_template.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello world" in messages[1]["content"]
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["prompt"] == "PROMPT"
        assert call_kwargs["max_tokens"] == FEATURES[1]["max_tokens"]

    @patch("streamlit_app.generate")
    def test_language_directive_appended_to_user_turn(
        self, mock_generate: MagicMock, tokenizer: MagicMock
    ) -> None:
        mock_generate.return_value = "{}"

        run_feature(FEATURES[3], "hello", MagicMock(), tokenizer, language="German")

        messages = tokenizer.apply_chat_template.call_args[0][0]
        # Directive lands on the user turn; the system prompt stays verbatim.
        assert "hello" in messages[1]["content"]
        assert "German" in messages[1]["content"]
        assert messages[0]["content"] == FEATURES[3]["system"]

    @patch("streamlit_app.generate")
    def test_english_leaves_user_turn_unchanged(
        self, mock_generate: MagicMock, tokenizer: MagicMock
    ) -> None:
        mock_generate.return_value = "summary"

        run_feature(FEATURES[0], "hello", MagicMock(), tokenizer, language="English")

        messages = tokenizer.apply_chat_template.call_args[0][0]
        assert messages[1]["content"] == FEATURES[0]["user_template"].format(
            text="hello"
        )

    @pytest.mark.parametrize(
        "feature, expects_penalty",
        [
            pytest.param(FEATURES[0], True, id="prose-applies-penalty"),
            pytest.param(FEATURES[3], False, id="json-skips-penalty"),
        ],
    )
    @patch("streamlit_app.make_logits_processors")
    @patch("streamlit_app.make_sampler")
    @patch("streamlit_app.generate")
    def test_decoding_params(
        self,
        mock_generate: MagicMock,
        mock_make_sampler: MagicMock,
        mock_make_logits: MagicMock,
        feature: dict,
        expects_penalty: bool,
        tokenizer: MagicMock,
    ) -> None:
        mock_generate.return_value = "{}"
        mock_make_sampler.return_value = "SAMPLER"
        mock_make_logits.return_value = "PROCS"

        run_feature(feature, "text", MagicMock(), tokenizer)

        mock_make_sampler.assert_called_once_with(temp=0.0)
        assert mock_generate.call_args[1]["sampler"] == "SAMPLER"
        if expects_penalty:
            mock_make_logits.assert_called_once_with(repetition_penalty=1.2)
            assert mock_generate.call_args[1]["logits_processors"] == "PROCS"
        else:
            mock_make_logits.assert_not_called()
            assert mock_generate.call_args[1]["logits_processors"] is None


class TestRenderResult:
    @pytest.mark.parametrize(
        "key, parsed",
        [
            pytest.param("intents", {"intent": ["a", "b"]}, id="intent-list"),
            pytest.param("sentiment", {"sentiment": {"x": 1}}, id="sentiment-dict"),
        ],
    )
    @patch("streamlit_app.st")
    def test_metric_value_coerced_to_string(
        self, mock_st: MagicMock, key: str, parsed: dict
    ) -> None:
        render_result(key, {"raw": "x", "parsed": parsed})
        _, value = mock_st.metric.call_args[0]
        assert isinstance(value, str)

    @patch("streamlit_app.st")
    def test_non_list_topics_not_sent_to_dataframe(self, mock_st: MagicMock) -> None:
        render_result("topics", {"raw": "x", "parsed": {"topics": "politics"}})
        mock_st.dataframe.assert_not_called()

    @patch("streamlit_app.st")
    def test_numeric_confidence_rendered_as_percent(self, mock_st: MagicMock) -> None:
        render_result(
            "sentiment",
            {"raw": "x", "parsed": {"sentiment": "positive", "confidence": 0.9}},
        )
        captions = [call.args[0] for call in mock_st.caption.call_args_list]
        assert any("90%" in str(text) for text in captions)

    @pytest.mark.parametrize(
        "sentiment, color",
        [
            pytest.param("positive", "green", id="positive-green"),
            pytest.param("negative", "red", id="negative-red"),
            pytest.param("neutral", "gray", id="neutral-gray"),
            pytest.param("mixed", "orange", id="mixed-orange"),
        ],
    )
    @patch("streamlit_app.st")
    def test_sentiment_value_colored_by_enum(
        self, mock_st: MagicMock, sentiment: str, color: str
    ) -> None:
        render_result("sentiment", {"raw": "x", "parsed": {"sentiment": sentiment}})
        _, value = mock_st.metric.call_args[0]
        assert value == f":{color}[{sentiment}]"

    @patch("streamlit_app.st")
    def test_unknown_sentiment_renders_uncolored(self, mock_st: MagicMock) -> None:
        # An out-of-enum label must not be wrapped in a bogus `:None[...]` color.
        render_result("sentiment", {"raw": "x", "parsed": {"sentiment": "ecstatic"}})
        _, value = mock_st.metric.call_args[0]
        assert value == "ecstatic"


class TestLanguageDirective:
    def test_english_is_empty_for_all_features(self) -> None:
        # Prompts are already English, so no directive is added.
        for feature in FEATURES:
            assert language_directive(feature, LANGUAGE_ENGLISH) == ""

    def test_prose_targets_the_language(self) -> None:
        directive = language_directive(FEATURES[0], "German")  # summary = prose
        assert "German" in directive
        assert "entire response" in directive

    def test_json_localizes_values_but_keeps_keys_english(self) -> None:
        directive = language_directive(FEATURES[3], "German")  # sentiment = json
        assert "German" in directive  # free-text values localized
        assert "English" in directive  # keys/enums stay English
        assert "key" in directive.lower()

    def test_match_input_uses_relative_phrase(self) -> None:
        directive = language_directive(FEATURES[0], LANGUAGE_AUTO)
        assert "same language as the text" in directive
        assert LANGUAGE_AUTO not in directive  # not the literal "Match input" label


class TestResolveMaxInputTokens:
    def test_default_when_unset(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("MAX_INPUT_TOKENS", None)
            assert _resolve_max_input_tokens() == _DEFAULT_MAX_INPUT_TOKENS

    def test_reads_env_value(self) -> None:
        with patch.dict(os.environ, {"MAX_INPUT_TOKENS": "4096"}):
            assert _resolve_max_input_tokens() == 4096

    def test_clamps_to_model_max(self) -> None:
        with patch.dict(os.environ, {"MAX_INPUT_TOKENS": "999999"}):
            assert _resolve_max_input_tokens() == MODEL_MAX_TOKENS

    def test_non_integer_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {"MAX_INPUT_TOKENS": "lots"}):
            assert _resolve_max_input_tokens() == _DEFAULT_MAX_INPUT_TOKENS

    def test_non_positive_falls_back_to_default(self) -> None:
        # A sign typo or 0 must NOT silently clamp to a 1-token cap.
        for bad in ("0", "-1", "-16384"):
            with patch.dict(os.environ, {"MAX_INPUT_TOKENS": bad}):
                assert _resolve_max_input_tokens() == _DEFAULT_MAX_INPUT_TOKENS

    def test_default_and_ceiling_pinned(self) -> None:
        # Deliberate choices: 16K is the memory-safe bf16 default on a 32 GB Mac;
        # 131072 is Granite 4.1's 128K ceiling. Pinned so neither drifts silently.
        assert _DEFAULT_MAX_INPUT_TOKENS == 16384
        assert MODEL_MAX_TOKENS == 131072
        # The default must itself sit inside the clamp range.
        assert 1 <= _DEFAULT_MAX_INPUT_TOKENS <= MODEL_MAX_TOKENS


class TestEffectiveMaxTokens:
    def test_english_uses_base_budget(self) -> None:
        feature = FEATURES[3]  # sentiment
        assert _effective_max_tokens(feature, "English") == feature["max_tokens"]

    def test_latin_language_uses_base_budget(self) -> None:
        feature = FEATURES[3]
        assert _effective_max_tokens(feature, "German") == feature["max_tokens"]

    def test_token_heavy_language_enlarges_budget(self) -> None:
        feature = FEATURES[3]
        assert _effective_max_tokens(feature, "Japanese") == feature["max_tokens"] * 2

    def test_match_input_enlarges_budget(self) -> None:
        feature = FEATURES[0]  # summary
        assert (
            _effective_max_tokens(feature, LANGUAGE_AUTO) == feature["max_tokens"] * 2
        )


def _flatten_theme_items(
    table: dict, prefix: str = "theme"
) -> Iterator[tuple[str, object]]:
    """Yield (dotted option key, leaf value) pairs for a parsed [theme] table,
    recursing into the light/dark sub-tables (e.g. ("theme.light.primaryColor",
    "#0f62fe"))."""
    for name, value in table.items():
        key = f"{prefix}.{name}"
        if isinstance(value, dict):
            yield from _flatten_theme_items(value, key)
        else:
            yield key, value


def _flatten_theme_keys(table: dict, prefix: str = "theme") -> Iterator[str]:
    """The dotted option keys of a parsed [theme] table (see _flatten_theme_items)."""
    return (key for key, _ in _flatten_theme_items(table, prefix))


class TestThemeConfig:
    """The IBM Carbon-inspired theme ships in .streamlit/config.toml.

    Streamlit only *warns* on a malformed theme — it never raises — so a typo or
    a dropped section would silently disable styling without any test noticing.
    These assertions make that failure mode visible.
    """

    CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"

    def _theme(self) -> dict:
        with self.CONFIG.open("rb") as handle:
            return tomllib.load(handle)["theme"]

    def test_config_exists_and_parses(self) -> None:
        assert self.CONFIG.is_file()
        with self.CONFIG.open("rb") as handle:
            tomllib.load(handle)  # raises TOMLDecodeError on a syntax error

    def test_defines_light_and_dark_modes(self) -> None:
        # Both sections must exist for the in-app light/dark toggle to appear.
        theme = self._theme()
        assert "light" in theme
        assert "dark" in theme

    def test_uses_ibm_blue_primary(self) -> None:
        # IBM Blue 60 — the on-brand accent the whole theme is built around.
        theme = self._theme()
        assert theme["light"]["primaryColor"] == "#0f62fe"
        assert theme["dark"]["primaryColor"] == "#0f62fe"

    def test_loads_ibm_plex_fonts(self) -> None:
        theme = self._theme()
        assert "IBM Plex Sans" in theme["font"]
        assert "IBM Plex Mono" in theme["codeFont"]

    def test_only_recognized_theme_keys(self) -> None:
        # Streamlit silently ignores unrecognized theme keys (it warns, never
        # raises), so a mis-cased key like `backgroundcolor` would disable that
        # style with no error and no failing test. Cross-check every key — incl.
        # the light/dark sub-tables — against Streamlit's own option registry so a
        # typo or future drift fails loudly here instead of going unnoticed.
        import streamlit.config

        recognized = set(streamlit.config.get_config_options())
        unknown = [
            key for key in _flatten_theme_keys(self._theme()) if key not in recognized
        ]
        assert not unknown, f"unrecognized theme keys: {unknown}"

    def test_color_values_are_six_digit_hex(self) -> None:
        # Streamlit doesn't validate color *values* either — a dropped "#" or
        # digit passes the key check yet silently disables that color, the same
        # failure mode as a mis-cased key. Enforce this project's 6-digit-hex
        # house style on every single-string *Color value (list-valued chart
        # color keys, if ever added, are skipped by the str guard).
        malformed = [
            f"{key}={value!r}"
            for key, value in _flatten_theme_items(self._theme())
            if key.endswith("Color")
            and isinstance(value, str)
            and not re.fullmatch(r"#[0-9a-fA-F]{6}", value)
        ]
        assert not malformed, f"malformed hex color values: {malformed}"
