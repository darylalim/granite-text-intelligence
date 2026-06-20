import json
import os
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
                "help",
                "output",
                "max_tokens",
                "system",
                "user_template",
            ):
                assert field in feature
            assert "{text}" in feature["user_template"]
            assert feature["output"] in ("prose", "json")

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
