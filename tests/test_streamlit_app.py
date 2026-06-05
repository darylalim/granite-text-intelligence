import json
from unittest.mock import MagicMock, patch

from streamlit_app import (
    FEATURES,
    LABELS,
    MAX_INPUT_TOKENS,
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
    def test_plain_json(self) -> None:
        assert parse_json_output('{"a": 1}') == {"a": 1}

    def test_json_embedded_in_text(self) -> None:
        assert parse_json_output('Here you go: {"a": 1} done') == {"a": 1}

    def test_json_in_code_fence(self) -> None:
        assert parse_json_output('```json\n{"sentiment": "positive"}\n```') == {
            "sentiment": "positive"
        }

    def test_invalid_returns_none(self) -> None:
        assert parse_json_output("not json at all") is None

    def test_malformed_braces_return_none(self) -> None:
        assert parse_json_output("{not: valid}") is None

    def test_first_object_when_multiple(self) -> None:
        assert parse_json_output('prefix {"a": 1} middle {"b": 2} suffix') == {"a": 1}

    def test_recovers_object_after_stray_braces(self) -> None:
        assert parse_json_output(
            'I think {maybe} the answer is {"sentiment": "positive"}'
        ) == {"sentiment": "positive"}

    def test_top_level_array_returns_none(self) -> None:
        assert parse_json_output("[1, 2, 3]") is None

    def test_scalar_json_returns_none(self) -> None:
        assert parse_json_output("true") is None
        assert parse_json_output("42") is None
        assert parse_json_output('"hello"') is None


class TestResolveInput:
    def test_pasted_wins(self) -> None:
        assert resolve_input("typed", "uploaded", "sample") == "typed"

    def test_upload_when_no_pasted(self) -> None:
        assert resolve_input("", "uploaded", "sample") == "uploaded"

    def test_sample_when_neither(self) -> None:
        assert resolve_input("", "", "sample") == "sample"

    def test_all_empty(self) -> None:
        assert resolve_input("", "", "") == ""

    def test_strips_whitespace(self) -> None:
        assert resolve_input("  spaced  ", "", "") == "spaced"

    def test_whitespace_only_pasted_falls_through(self) -> None:
        assert resolve_input("   ", "uploaded", "sample") == "uploaded"


class TestTruncateToTokens:
    def test_short_text_unchanged(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        text, truncated = truncate_to_tokens("hello", tokenizer)

        assert text == "hello"
        assert truncated is False
        tokenizer.decode.assert_not_called()

    def test_long_text_truncated(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_INPUT_TOKENS + 50))
        tokenizer.decode.return_value = "truncated text"

        text, truncated = truncate_to_tokens("long", tokenizer)

        assert truncated is True
        assert text == "truncated text"
        tokenizer.decode.assert_called_once_with(
            list(range(MAX_INPUT_TOKENS)), skip_special_tokens=True
        )

    def test_boundary_not_truncated(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_INPUT_TOKENS))

        _, truncated = truncate_to_tokens("boundary", tokenizer)

        assert truncated is False
        tokenizer.decode.assert_not_called()

    def test_encode_excludes_special_tokens(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))

        truncate_to_tokens("hello", tokenizer)

        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    return tokenizer


class TestRunFeature:
    @patch("streamlit_app.generate")
    def test_prose_feature_returns_raw_unparsed(self, mock_generate: MagicMock) -> None:
        mock_generate.return_value = "  A concise summary.  "

        result = run_feature(FEATURES[0], "text", MagicMock(), _tokenizer())

        assert result["raw"] == "A concise summary."
        assert result["parsed"] is None

    @patch("streamlit_app.generate")
    def test_json_feature_parses_output(self, mock_generate: MagicMock) -> None:
        mock_generate.return_value = '{"sentiment": "positive", "confidence": 0.9}'

        result = run_feature(FEATURES[3], "text", MagicMock(), _tokenizer())

        assert result["parsed"] == {"sentiment": "positive", "confidence": 0.9}

    @patch("streamlit_app.generate")
    def test_json_feature_unparseable_returns_none(
        self, mock_generate: MagicMock
    ) -> None:
        mock_generate.return_value = "totally not json"

        result = run_feature(FEATURES[1], "text", MagicMock(), _tokenizer())

        assert result["parsed"] is None
        assert result["raw"] == "totally not json"

    @patch("streamlit_app.generate")
    def test_applies_chat_template_and_max_tokens(
        self, mock_generate: MagicMock
    ) -> None:
        mock_generate.return_value = "{}"
        tokenizer = _tokenizer()

        run_feature(FEATURES[1], "hello world", MagicMock(), tokenizer)

        messages = tokenizer.apply_chat_template.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello world" in messages[1]["content"]
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["prompt"] == "PROMPT"
        assert call_kwargs["max_tokens"] == FEATURES[1]["max_tokens"]

    @patch("streamlit_app.make_logits_processors")
    @patch("streamlit_app.make_sampler")
    @patch("streamlit_app.generate")
    def test_prose_uses_greedy_sampler_and_repetition_penalty(
        self,
        mock_generate: MagicMock,
        mock_make_sampler: MagicMock,
        mock_make_logits: MagicMock,
    ) -> None:
        mock_generate.return_value = "summary"
        mock_make_sampler.return_value = "SAMPLER"
        mock_make_logits.return_value = "PROCS"

        run_feature(FEATURES[0], "text", MagicMock(), _tokenizer())

        mock_make_sampler.assert_called_once_with(temp=0.0, top_p=1.0)
        mock_make_logits.assert_called_once_with(repetition_penalty=1.2)
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["sampler"] == "SAMPLER"
        assert call_kwargs["logits_processors"] == "PROCS"

    @patch("streamlit_app.make_logits_processors")
    @patch("streamlit_app.make_sampler")
    @patch("streamlit_app.generate")
    def test_json_feature_skips_repetition_penalty(
        self,
        mock_generate: MagicMock,
        mock_make_sampler: MagicMock,
        mock_make_logits: MagicMock,
    ) -> None:
        mock_generate.return_value = "{}"

        run_feature(FEATURES[3], "text", MagicMock(), _tokenizer())

        mock_make_sampler.assert_called_once_with(temp=0.0, top_p=1.0)
        mock_make_logits.assert_not_called()
        assert mock_generate.call_args[1]["logits_processors"] is None


class TestRenderResult:
    @patch("streamlit_app.st")
    def test_intent_value_coerced_to_string(self, mock_st: MagicMock) -> None:
        render_result("intents", {"raw": "x", "parsed": {"intent": ["a", "b"]}})
        _, value = mock_st.metric.call_args[0]
        assert isinstance(value, str)

    @patch("streamlit_app.st")
    def test_sentiment_value_coerced_to_string(self, mock_st: MagicMock) -> None:
        render_result("sentiment", {"raw": "x", "parsed": {"sentiment": {"x": 1}}})
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
