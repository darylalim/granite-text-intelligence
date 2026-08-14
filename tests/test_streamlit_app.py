import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    _DEFAULT_MAX_INPUT_TOKENS,
    _SENTIMENT_COLOR,
    FEATURES,
    LABELS,
    LANGUAGE_AUTO,
    LANGUAGE_ENGLISH,
    LANGUAGES,
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

_JSON_FEATURES = [
    pytest.param(i, id=feature["key"])
    for i, feature in enumerate(FEATURES)
    if feature["output"] == "json"
]
_ALL_FEATURES = [pytest.param(i, id=f["key"]) for i, f in enumerate(FEATURES)]


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

    @pytest.mark.parametrize("index", _JSON_FEATURES)
    def test_json_features_declare_their_localized_field(self, index: int) -> None:
        # language_directive indexes this directly, so a new JSON feature that
        # omits it raises rather than inheriting another feature's field name
        # and telling the model to translate a field its schema lacks.
        feature = FEATURES[index]
        assert feature["localized_field"]
        # Some word in the phrase must name a field the feature's own schema
        # declares ("every topic label" -> label, "the rationale text" ->
        # rationale), so the directive can't point at a field that isn't there.
        words = feature["localized_field"].split()
        assert any(f'"{word}"' in feature["system"] for word in words), (
            f"{feature['key']}: {feature['localized_field']!r} names no schema field"
        )

    def test_prose_features_declare_no_localized_field(self) -> None:
        # Prose localizes the entire response, so there is no field to name.
        assert "localized_field" not in FEATURES[0]

    def test_only_summary_is_prose(self) -> None:
        assert FEATURES[0]["output"] == "prose"
        assert all(feature["output"] == "json" for feature in FEATURES[1:])

    def test_labels_match_features(self) -> None:
        assert LABELS == {feature["key"]: feature["label"] for feature in FEATURES}

    def test_labels_use_sentence_case(self) -> None:
        # Streamlit's design guidance ("Use sentence casing for titles and
        # labels. Title Case Feels Shouty."). These toggle labels carry no
        # acronyms or proper nouns, so each must equal its sentence-cased form
        # (first character upper, the rest lower).
        for feature in FEATURES:
            label = feature["label"]
            assert label == label.capitalize(), (
                f"toggle label {label!r} is not sentence case"
            )

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

    @pytest.mark.parametrize(
        ("index", "field"),
        [
            pytest.param(1, "label", id="topics-names-label"),
            pytest.param(2, "rationale", id="intents-names-rationale"),
            pytest.param(3, "rationale", id="sentiment-names-rationale"),
        ],
    )
    def test_json_names_the_feature_own_localizable_field(
        self, index: int, field: str
    ) -> None:
        # Naming the concrete field beats "all free-text field values": the
        # abstract phrasing was honored for topic labels but left rationale
        # in English.
        directive = language_directive(FEATURES[index], "Japanese")
        assert field in directive
        # ...but the name must stay UNQUOTED. Quoting it ('each topic "label"')
        # collapsed Japanese labels into meaningless katakana, apparently read
        # as a literal string to emit rather than a field to translate.
        assert f'"{field}"' not in directive

    @pytest.mark.parametrize("index", _JSON_FEATURES)
    @pytest.mark.parametrize("language", [LANGUAGE_AUTO, "Japanese", "German"])
    def test_json_puts_the_localize_requirement_last(
        self, index: int, language: str
    ) -> None:
        # Instructions nearest the generation point carry the most weight, so
        # the keep-English exception must not be the trailing clause. Located
        # with find(), not rindex(): a missing clause must fail this assertion
        # rather than raise ValueError and obscure which invariant broke.
        directive = language_directive(FEATURES[index], language)
        keep_at = directive.find("Keep every JSON key")
        write_at = directive.find("Write ")
        assert keep_at != -1, "keep-English clause missing"
        assert write_at != -1, "localize clause missing"
        assert keep_at < write_at

    def test_prose_suppresses_language_commentary(self) -> None:
        # Guards against trailing asides like "(Note: written in Japanese as
        # requested)" leaking into the summary.
        directive = language_directive(FEATURES[0], "Japanese")
        assert "no note or comment about the language" in directive

    # Ways a future edit might re-introduce the contradiction below.
    _NEGATIVE_ENGLISH = (
        "not in English",
        "not English",
        "never in English",
        "avoid English",
        "instead of English",
        "rather than English",
    )

    @pytest.mark.parametrize("index", _JSON_FEATURES)
    @pytest.mark.parametrize("language", [LANGUAGE_AUTO, "Japanese", "German"])
    def test_json_keeps_english_positively(self, index: int, language: str) -> None:
        directive = language_directive(FEATURES[index], language)
        # Non-vacuous: the keep-English clause must actually be present...
        assert "English" in directive
        # ...but only ever as a positive instruction about keys and enums. Under
        # "Match input" the target may itself be English, so telling the model
        # to avoid English would contradict the directive's own requirement.
        for phrase in self._NEGATIVE_ENGLISH:
            assert phrase not in directive

    @pytest.mark.parametrize("language", [LANGUAGE_AUTO, "Japanese", "German"])
    def test_prose_makes_no_claim_about_english(self, language: str) -> None:
        # Prose localizes the whole response, so it has no keys or enums to
        # exempt — any mention of English here could only be a prohibition.
        assert "English" not in language_directive(FEATURES[0], language)


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
        # Deliberate choices: 16K is the memory-safe default (~2.6 GB of KV cache
        # at ~160 KB/token, independent of the weights' quantization); 131072 is
        # Granite 4.1's 128K ceiling. Pinned so neither drifts silently.
        assert _DEFAULT_MAX_INPUT_TOKENS == 16384
        assert MODEL_MAX_TOKENS == 131072
        # The default must itself sit inside the clamp range.
        assert 1 <= _DEFAULT_MAX_INPUT_TOKENS <= MODEL_MAX_TOKENS


class TestEffectiveMaxTokens:
    @pytest.mark.parametrize("index", _ALL_FEATURES)
    def test_english_uses_base_budget(self, index: int) -> None:
        feature = FEATURES[index]
        assert _effective_max_tokens(feature, LANGUAGE_ENGLISH) == feature["max_tokens"]

    @pytest.mark.parametrize("index", _ALL_FEATURES)
    @pytest.mark.parametrize(
        "language",
        [
            pytest.param(lang, id=lang.replace(" ", "-").lower())
            for lang in LANGUAGES
            if lang != LANGUAGE_ENGLISH
        ],
    )
    def test_every_non_english_target_enlarges_budget(
        self, index: int, language: str
    ) -> None:
        # Covers LANGUAGE_AUTO too, since it is LANGUAGES[0]. Latin-script
        # targets are included deliberately: a localized German rationale
        # overran the sentiment feature's 128-token budget and truncated
        # mid-object, so restricting this to CJK/Arabic is not enough.
        feature = FEATURES[index]
        assert _effective_max_tokens(feature, language) == feature["max_tokens"] * 2


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

    def test_sentiment_colors_are_tuned_in_both_modes(self) -> None:
        # render_result colors the sentiment metric via `:color[...]` markdown,
        # which reads the theme's `<color>Color` keys. Every color the app emits
        # must be tuned in BOTH modes, else that sentiment falls back to
        # Streamlit's default hue — the orange-vs-yellow gap this guards against
        # (the code emitted `:orange[...]` while only yellowColor was defined).
        theme = self._theme()
        for color in sorted(set(_SENTIMENT_COLOR.values())):
            key = f"{color}Color"
            assert key in theme["light"], f"{key} missing from [theme.light]"
            assert key in theme["dark"], f"{key} missing from [theme.dark]"


CI_WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


def _load_ci_workflow() -> dict:
    """Parse .github/workflows/ci.yml.

    Shared by TestCIWorkflow and TestReleaseWorkflow, which guard two jobs of the
    same file: a second loader is exactly the drift these classes exist to catch.
    pyyaml is a dev-only dependency, imported lazily so a missing parser fails
    those classes rather than module collection (mirrors TestThemeConfig's lazy
    streamlit.config import).
    """
    import yaml

    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


class TestCIWorkflow:
    """The GitHub Actions CI workflow ships in .github/workflows/ci.yml.

    Like the theme config, CI fails *silently*: a dropped step or a runner
    swapped to Linux (where the darwin-only mlx can't install, so the suite can't
    even be collected) would still show a green check on whatever it does run.
    These assertions pin the invariants that keep CI a faithful mirror of the
    documented local gates.
    """

    WORKFLOW = CI_WORKFLOW

    def _workflow(self) -> dict:
        return _load_ci_workflow()

    def _run_commands(self) -> str:
        steps = self._workflow()["jobs"]["check"]["steps"]
        return "\n".join(step["run"] for step in steps if "run" in step)

    def test_workflow_exists_and_parses(self) -> None:
        assert self.WORKFLOW.is_file()
        self._workflow()  # raises yaml.YAMLError on a syntax error

    # The Apple Silicon runner labels GitHub currently supports. An allow-list
    # rather than the `major >= 14` floor this replaces, which was satisfiable by
    # two separate broken states: it stayed green on `macos-14` right through that
    # image's deprecation (brownout failures from 2026-10-05, retired 2026-11-02),
    # and it admits the x64 `macos-*-intel` / `macos-*-large` labels, where the
    # arm64-only mlx wheels don't install at all. GitHub keeps only the latest two
    # OS versions, so the valid set is a moving window: when an image is deprecated
    # (actions/runner-images issues), move this list forward — don't widen it back
    # into a floor, which is what let a retiring runner pass unnoticed.
    APPLE_SILICON_RUNNERS = frozenset({"macos-latest", "macos-15", "macos-26"})

    def test_runs_on_apple_silicon(self) -> None:
        # Load-bearing: uv.lock pins mlx/mlx-metal to sys_platform == 'darwin' and
        # streamlit_app.py imports mlx at module top, so a Linux runner can't even
        # collect the tests. But `sys_platform == 'darwin'` is true on Intel macOS
        # too, where the arm64-only mlx wheels won't install.
        runs_on = self._workflow()["jobs"]["check"]["runs-on"]
        assert runs_on in self.APPLE_SILICON_RUNNERS, (
            f"{runs_on!r} is not a supported Apple Silicon runner label; expected "
            f"one of {sorted(self.APPLE_SILICON_RUNNERS)}"
        )

    def test_delegates_to_the_shared_quality_gate(self) -> None:
        # The four commands documented in CLAUDE.md / README "Development" live in
        # scripts/gate.sh so that CI and the Claude Code Stop hook cannot drift
        # apart; TestHooksConfig pins what the script actually runs. Spelling them
        # out here again would reintroduce precisely that drift.
        assert "scripts/gate.sh" in self._run_commands()

    def test_install_is_lockfile_pinned(self) -> None:
        # --locked makes CI fail on a stale lockfile instead of silently resolving
        # a different dependency set than uv.lock records.
        assert "uv sync --locked" in self._run_commands()

    def test_triggers_on_push_to_main_and_pull_requests(self) -> None:
        # PyYAML (YAML 1.1) parses the bare `on:` key as the boolean True, so the
        # trigger block lands under the True key rather than the string "on".
        data = self._workflow()
        triggers = data.get("on", data.get(True)) or {}
        assert "main" in triggers["push"]["branches"]
        assert "pull_request" in triggers


class TestReleaseWorkflow:
    """Auto-tag-and-publish ships in .github/workflows/ci.yml + scripts/release.sh.

    The silent-degradation guard with the least forgiving failure mode, alongside
    TestThemeConfig, TestCIWorkflow and TestHooksConfig. Automation that stops
    firing is indistinguishable from a stretch with no version bumps, and
    automation that fires from the wrong place — a pull request, a red gate, an
    unmerged commit — has already published a tag by the time anyone looks. None
    of it is reachable by running the app.
    """

    ROOT = Path(__file__).parent.parent
    WORKFLOW = CI_WORKFLOW
    SCRIPT = ROOT / "scripts" / "release.sh"

    def _workflow(self) -> dict:
        return _load_ci_workflow()

    def _job(self) -> dict:
        return self._workflow()["jobs"]["release"]

    def test_release_script_is_executable(self) -> None:
        # The workflow invokes it directly, not through `sh`, so a dropped +x bit
        # breaks the release with a green gate above it.
        assert self.SCRIPT.is_file()
        assert os.access(self.SCRIPT, os.X_OK), "scripts/release.sh is not executable"

    def test_release_job_delegates_to_the_script(self) -> None:
        # Inlining it back into YAML would put every behavioral assertion below
        # out of reach: an escaped one-liner cannot be run or shellchecked.
        runs = "\n".join(step["run"] for step in self._job()["steps"] if "run" in step)
        assert "scripts/release.sh" in runs

    def test_release_requires_a_green_gate(self) -> None:
        # A tag is public the moment it exists and is what users install; cutting
        # one from a build that failed lint, types or tests is worse than not
        # cutting one at all.
        needs = self._job()["needs"]
        assert "check" in ([needs] if isinstance(needs, str) else needs)

    def test_release_only_fires_on_a_push_to_main(self) -> None:
        # Without the event guard the job would also run on pull_request, where a
        # fork could get a version bump tagged without review.
        condition = self._job()["if"]
        assert "github.event_name == 'push'" in condition
        assert "refs/heads/main" in condition

    def test_only_the_release_job_can_write(self) -> None:
        # Least privilege, and the reason the token grant sits on the job rather
        # than the workflow: `check` runs a pull request's code, including a
        # fork's, and must not be able to push a tag.
        workflow = self._workflow()
        assert workflow["permissions"]["contents"] == "read"
        assert workflow["jobs"]["release"]["permissions"]["contents"] == "write"
        assert "permissions" not in workflow["jobs"]["check"]

    def test_release_is_never_cancelled_in_flight(self) -> None:
        # Two pushes to main in quick succession race for the same tag. Cancelling
        # the older run is the wrong resolution: it can land between the draft and
        # the publish, stranding a draft that then blocks every later release.
        concurrency = self._job()["concurrency"]
        assert concurrency["group"]
        assert concurrency["cancel-in-progress"] is False

    def test_no_job_interpolates_into_the_shell(self) -> None:
        # ${{ }} inside `run:` is substituted textually before any shell sees it,
        # which is the whole of GitHub Actions script injection. Checked across
        # every job, not just this one: `release` is already confined to pushes to
        # main, while `check` is the job that runs a fork's pull-request code with
        # attacker-controlled github.head_ref / PR title in scope.
        for name, job in self._workflow()["jobs"].items():
            for step in job["steps"]:
                assert "${{" not in step.get("run", ""), (
                    f"interpolation inside run: in job {name!r}"
                )

    def test_release_job_is_given_a_token_and_a_repository(self) -> None:
        # GH_REPO names the repository outright instead of leaving gh to infer it
        # from the checkout's git remote; unset outside Actions, where inference is
        # what you want.
        step = next(
            step
            for step in self._job()["steps"]
            if step.get("run", "").strip().endswith("release.sh")
        )
        assert "GH_TOKEN" in step["env"]
        assert "GH_REPO" in step["env"]

    # There is deliberately no test asserting uv.lock's recorded project version
    # matches pyproject.toml's. uv.lock does carry that version, and CI's
    # `uv sync --locked` does fail on a stale one — but every gate command runs
    # under `uv run`, which re-locks before pytest is even imported (verified: a
    # bump to 0.2.0 left uv.lock reading 0.2.0 by the time the assertion ran, and
    # reverting pyproject.toml silently rewrote it back). Such an assertion can
    # therefore never fail from this side, and the real mistake it would be aimed
    # at — committing pyproject.toml without the regenerated uv.lock — is git
    # hygiene that no in-process test can observe. See the release procedure in
    # CLAUDE.md, which documents the coupling instead.

    def _run_release(
        self,
        tmp_path: Path,
        *,
        tag_exists: bool = False,
        probe_error: str = "gh: Not Found (HTTP 404)",
        existing_draft: bool = False,
        version: str = "9.9.9",
        uv_exit: int = 0,
        create_exit: int = 0,
        edit_exit: int = 0,
        list_exit: int = 0,
        sha: str | None = "c0ffeeb",
        actions: bool = True,
        omit: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        """Run release.sh in a throwaway repo with `gh` and `uv` stubbed.

        Source assertions are as weak here as they were for gate.sh, and for a
        sharper reason: the script spells `--draft` twice — once to create the
        draft, once as `--draft=false` to publish it — so a substring check
        cannot distinguish the two-step publish from a create that never
        publishes, which is exactly the mutation that would leave every release
        sitting unpublished. "Exits 0 when already tagged" is pure control flow
        with no distinctive string at all. Stubbing also keeps the test hermetic:
        the real script publishes a real release.

        Both stubs log their argv, so what was *invoked* is assertable and not
        just what came back — `uv version` and `uv version --short` return the
        same thing here but not in production.

        Returns the completed process and the stub calls it recorded, in order,
        each prefixed with the tool name.
        """
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        shutil.copy2(self.SCRIPT, scripts / "release.sh")

        log = tmp_path / "calls.log"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Dispatches on the subcommand to fake each outcome. `release list` emits
        # one draft tag per line, matching the --jq the script asks for.
        stubs = {
            "gh": (
                'case "$*" in\n'
                f"\"api \"*) printf '%s\\n' '{probe_error}' >&2; exit {0 if tag_exists else 1} ;;\n"
                f"\"release list\"*) printf '%s' '{f'v{version}' if existing_draft else ''}'; exit {list_exit} ;;\n"
                f'"release create "*) exit {create_exit} ;;\n'
                f'"release edit "*) exit {edit_exit} ;;\n'
                "esac\n"
            ),
            "uv": f"printf '%s\\n' '{version}'\nexit {uv_exit}\n",
        }
        for tool, body in stubs.items():
            if tool == omit:
                continue
            stub = bin_dir / tool
            stub.write_text(
                f"#!/bin/sh\nprintf '%s\\n' \"{tool} $*\" >>'{log}'\n{body}exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)

        # A stubbed-out tool has to be unreachable, not merely unstubbed, so the
        # PATH drops everything but the stubs and the system tools git needs.
        path = f"{bin_dir}" if omit else f"{bin_dir}:{os.environ['PATH']}"
        env: dict[str, str] = {**os.environ, "PATH": f"{path}:/usr/bin:/bin"}
        env.pop("GITHUB_SHA", None)
        env.pop("GITHUB_ACTIONS", None)
        if sha is not None:
            env["GITHUB_SHA"] = sha
        if actions:
            env["GITHUB_ACTIONS"] = "true"

        result = subprocess.run(
            [str(scripts / "release.sh")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,  # a non-zero exit is the thing under test
            env=env,
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    @staticmethod
    def _released(calls: list[str]) -> bool:
        return any(call.startswith("gh release create") for call in calls)

    def test_an_already_tagged_version_releases_nothing(self, tmp_path: Path) -> None:
        # The idempotence contract: every push to main runs this, and all but the
        # ones that bumped the version must be no-ops that still exit 0, or the
        # job goes red on ordinary commits.
        result, calls = self._run_release(tmp_path, tag_exists=True)
        assert result.returncode == 0
        assert not self._released(calls), "re-released an already-tagged version"

    def test_an_untagged_version_is_drafted_then_published(
        self, tmp_path: Path
    ) -> None:
        # Both halves matter. Draft-only leaves the release invisible and, because
        # a draft carries no git tag, re-attempts it forever; publishing without
        # drafting first gives up the property that a failure mid-release leaves
        # no tag behind to be mistaken for a completed one.
        result, calls = self._run_release(tmp_path)
        assert result.returncode == 0
        create = next(
            i for i, c in enumerate(calls) if c.startswith("gh release create")
        )
        publish = next(
            i for i, c in enumerate(calls) if c.startswith("gh release edit")
        )
        assert create < publish, "published before drafting"
        assert "--draft" in calls[create]
        assert "--draft=false" in calls[publish]

    def test_the_version_is_read_with_the_short_flag(self, tmp_path: Path) -> None:
        # Load-bearing and otherwise untestable: bare `uv version` prints
        # "<name> <version>", which the plausibility guard would then refuse,
        # bricking releases permanently. The stub returns the same string either
        # way, so only the recorded argv can tell them apart.
        _, calls = self._run_release(tmp_path)
        assert "uv version --short" in calls

    def test_notes_are_generated_and_the_title_matches_the_tag(
        self, tmp_path: Path
    ) -> None:
        # --generate-notes is what makes this a release rather than a bare tag;
        # the explicit title pins the v<version> convention every release so far
        # follows, which --generate-notes would otherwise synthesize for itself.
        _, calls = self._run_release(tmp_path, version="1.2.3")
        create = next(c for c in calls if c.startswith("gh release create"))
        assert "--generate-notes" in create
        assert "--title v1.2.3" in create

    def test_the_tag_is_the_pyproject_version_prefixed_with_v(
        self, tmp_path: Path
    ) -> None:
        _, calls = self._run_release(tmp_path, version="1.2.3")
        assert any(c.startswith("gh release create v1.2.3 ") for c in calls)

    def test_the_existing_tag_probe_is_an_exact_match(self, tmp_path: Path) -> None:
        # git/ref/tags/<tag> resolves one ref; the plural git/refs/tags/<tag> form
        # prefix-matches — verified against the live API, git/refs/tags/v0.1
        # answers 200 off the existing v0.1.0 — so a bump to 0.1 would report
        # itself already released and silently never ship.
        _, calls = self._run_release(tmp_path, version="1.2.3")
        probe = next(c for c in calls if c.startswith("gh api"))
        assert "git/ref/tags/v1.2.3" in probe
        assert "git/refs/tags/" not in probe

    def test_an_inconclusive_probe_refuses_to_release(self, tmp_path: Path) -> None:
        # Only a definite 404 means "not released yet". A 5xx, a rate limit or an
        # expired token fails the probe identically, and reading those as "absent"
        # would re-release a version that already shipped.
        result, calls = self._run_release(
            tmp_path, probe_error="gh: Bad credentials (HTTP 401)"
        )
        assert result.returncode != 0
        assert not self._released(calls)

    def test_the_release_targets_the_commit_ci_validated(self, tmp_path: Path) -> None:
        # Defaulting to the branch tip would tag whatever landed while the gate
        # was running — an untested commit, under a tag that says it passed.
        _, calls = self._run_release(tmp_path, sha="deadbee")
        create = next(c for c in calls if c.startswith("gh release create"))
        assert "--target deadbee" in create

    def test_a_stranded_draft_is_adopted_rather_than_duplicated(
        self, tmp_path: Path
    ) -> None:
        # A draft carries no tag, so the probe cannot see one and GitHub accepts a
        # *second* draft for the same tag_name. Creating a rival leaves two drafts
        # racing for one tag, with `gh release edit` resolving whichever it finds
        # first — so the tag can land on the abandoned run's commit. Adopt and
        # retarget instead.
        result, calls = self._run_release(
            tmp_path, existing_draft=True, version="1.2.3"
        )
        assert result.returncode == 0
        assert not self._released(calls), "created a rival draft for the same tag"
        publish = next(c for c in calls if c.startswith("gh release edit"))
        assert "--target c0ffeeb" in publish, "adopted the draft without retargeting it"
        assert "--draft=false" in publish

    def test_a_failed_publish_names_the_recovery(self, tmp_path: Path) -> None:
        # The one state needing a human. Asserting merely on "draft" would pass on
        # the *create* failure's message too — this pins the sentence that says
        # how to unstick the pipeline.
        result, _ = self._run_release(tmp_path, edit_exit=1)
        assert result.returncode != 0
        assert "publish or delete the draft" in result.stderr

    def test_a_failed_draft_fails(self, tmp_path: Path) -> None:
        result, _ = self._run_release(tmp_path, create_exit=1)
        assert result.returncode != 0

    def test_an_unlistable_release_set_refuses_to_release(self, tmp_path: Path) -> None:
        # Without the draft listing there is no way to tell a first attempt from a
        # retry, and guessing "first attempt" is what creates the rival draft.
        result, calls = self._run_release(tmp_path, list_exit=1)
        assert result.returncode != 0
        assert not self._released(calls)

    @pytest.mark.parametrize(
        "reported",
        [
            pytest.param("uv 0.6.14", id="uv-own-version"),
            pytest.param("granite-text-intelligence 9.9.9", id="name-and-version"),
            pytest.param("0.1.0 (from pyproject.toml)", id="decorated"),
            pytest.param("", id="empty"),
        ],
    )
    def test_a_version_it_does_not_recognise_is_refused(
        self, tmp_path: Path, reported: str
    ) -> None:
        # `uv version --short` is the contract; the first two are what the other
        # shapes of `uv version` print. Tagging one would cut a release named
        # after the toolchain. The third is why the guard cannot check only the
        # first character: it leads with a digit and would be tagged verbatim,
        # spaces and all.
        result, calls = self._run_release(tmp_path, version=reported)
        assert result.returncode != 0
        assert not self._released(calls)

    def test_an_unreadable_version_is_refused(self, tmp_path: Path) -> None:
        result, calls = self._run_release(tmp_path, uv_exit=1)
        assert result.returncode != 0
        assert not self._released(calls)

    @pytest.mark.parametrize("tool", ["gh", "uv"])
    def test_a_missing_tool_is_named(self, tmp_path: Path, tool: str) -> None:
        # Neither is a project dependency — both are ambient on the runner — so
        # the failure has to name the tool rather than surface as a mis-attributed
        # "could not read a version".
        # Asserting merely that the name appears is too weak: without the guard,
        # `sh` says "gh: not found" on its own and the probe then refuses as
        # inconclusive, so the tool name reaches stderr either way. This pins the
        # script's own diagnosis.
        result, calls = self._run_release(tmp_path, omit=tool)
        assert result.returncode != 0
        assert f"{tool} is required" in result.stderr
        assert not self._released(calls)

    def test_an_unresolvable_commit_is_refused(self, tmp_path: Path) -> None:
        # Outside Actions there is no GITHUB_SHA, and the HEAD fallback has
        # nothing to resolve in a repo without commits. `gh release create` reads
        # an empty --target as "use the default branch", so an unguarded fallback
        # would tag a commit nobody chose.
        result, calls = self._run_release(tmp_path, sha=None)
        assert result.returncode != 0
        assert not self._released(calls)

    def test_a_hand_run_off_main_is_refused(self, tmp_path: Path) -> None:
        # Outside Actions nothing confines this to main — the workflow's `if:` is
        # not in play and HEAD is whatever branch the caller is on. A hand-run on
        # a feature branch would otherwise publish a public tag pointing at an
        # unreviewed commit, which is precisely what the workflow guard prevents
        # for CI.
        result, calls = self._run_release(tmp_path, actions=False)
        assert result.returncode != 0
        assert "origin/main" in result.stderr
        assert not self._released(calls)


class TestHooksConfig:
    """Claude Code hooks ship in .claude/settings.json and scripts/gate.sh.

    The Stop gate fails the same way the theme config and CI do — silently. It
    spent its whole lifetime chaining the four gates with `&&` and letting the
    exit code fall through, but Claude Code treats *only* exit 2 as blocking
    (ruff, ty and pytest all exit 1, which is a non-blocking error), so a red
    gate printed a message and the turn ended anyway. Nothing caught that,
    because nothing asserted on it. These pin the invariants that keep it a real
    gate rather than a notification.
    """

    ROOT = Path(__file__).parent.parent
    SETTINGS = ROOT / ".claude" / "settings.json"
    GATE = ROOT / "scripts" / "gate.sh"

    def _settings(self) -> dict:
        return json.loads(self.SETTINGS.read_text(encoding="utf-8"))

    def _commands(self, event: str) -> str:
        return "\n".join(
            hook["command"]
            for matcher in self._settings()["hooks"][event]
            for hook in matcher["hooks"]
        )

    def _gate(self) -> str:
        return self.GATE.read_text(encoding="utf-8")

    def test_settings_exists_and_parses(self) -> None:
        assert self.SETTINGS.is_file()
        self._settings()  # raises JSONDecodeError on a syntax error

    def test_gate_script_is_executable(self) -> None:
        # CI and the Stop hook both invoke it directly rather than through `sh`,
        # so a dropped +x bit breaks both callers at once.
        assert self.GATE.is_file()
        assert os.access(self.GATE, os.X_OK), "scripts/gate.sh is not executable"

    def test_gate_script_runs_all_four_documented_gates(self) -> None:
        gate = self._gate()
        for command in (
            "uv run ruff check .",
            "uv run ruff format --check .",
            "uv run ty check",
            "uv run pytest",
        ):
            assert command in gate, f"missing gate: {command!r}"

    def _run_gate(
        self, tmp_path: Path, uv_exit: int, uv_output: str = "DIAGNOSTIC"
    ) -> subprocess.CompletedProcess[str]:
        """Run gate.sh in a throwaway repo with `uv` stubbed to a fixed outcome.

        Asserting on the script's *source* is too weak to be worth much: `exit 2`
        and `>&2` both appear in its unrelated `cd`-failure branch, so a substring
        check stays green even when the failure path itself is mutated to exit 1
        or to drop the redirection (verified — both mutations passed a substring
        assertion). Executing it is the only way these invariants can actually
        fail. The stub also keeps the test hermetic and instant: the real gate
        would run the suite recursively.
        """
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        shutil.copy2(self.GATE, scripts / "gate.sh")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "uv"
        emit = f"printf '%s\\n' '{uv_output}'\n" if uv_output else ""
        stub.write_text(f"#!/bin/sh\n{emit}exit {uv_exit}\n", encoding="utf-8")
        stub.chmod(0o755)

        return subprocess.run(
            [str(scripts / "gate.sh")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,  # a non-zero exit is the thing under test
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        )

    def test_gate_blocks_with_exit_2(self, tmp_path: Path) -> None:
        # 1 is the conventional Unix failure code, and the one every tool the gate
        # runs actually returns — but Claude Code reads it as a non-blocking error
        # and lets the turn end on a red gate. Only 2 blocks.
        assert self._run_gate(tmp_path, uv_exit=1).returncode == 2

    def test_gate_reports_failures_on_stderr(self, tmp_path: Path) -> None:
        # stderr is the stream fed back on a block, while ruff and pytest print
        # their diagnostics to stdout; an unredirected failure blocks with no
        # explanation attached.
        result = self._run_gate(tmp_path, uv_exit=1, uv_output="RUFF-SAYS-NO")
        assert "RUFF-SAYS-NO" in result.stderr
        assert "RUFF-SAYS-NO" not in result.stdout

    def test_gate_names_every_failing_gate(self, tmp_path: Path) -> None:
        # No short-circuit: one block should report everything that is wrong,
        # since Claude Code allows only 8 consecutive blocks to fix it all.
        stderr = self._run_gate(tmp_path, uv_exit=1).stderr
        for name in ("ruff check", "ruff format --check", "ty check", "pytest"):
            assert name in stderr, f"failure not attributed to {name!r}"

    def test_gate_passes_when_every_gate_passes(self, tmp_path: Path) -> None:
        assert self._run_gate(tmp_path, uv_exit=0).returncode == 0

    def test_gate_explains_a_silent_death(self, tmp_path: Path) -> None:
        # A gate killed without writing anything — OOM, hook timeout, a
        # half-created .venv — must not block with a blank message, which is the
        # "blocked with no reason" failure the exit-2 change exists to remove.
        result = self._run_gate(tmp_path, uv_exit=137, uv_output="")
        assert result.returncode == 2
        assert "no output" in result.stderr

    def test_stop_hook_delegates_to_the_gate_script(self) -> None:
        # Inlining the chain back into settings.json would put it beyond the reach
        # of every assertion above: a JSON-escaped one-liner cannot be run, linted
        # or shellchecked.
        assert "scripts/gate.sh" in self._commands("Stop")

    def test_stop_hook_runs_unconditionally(self) -> None:
        # Two short-circuits were tried and both lost more than they saved. A
        # `git status --porcelain` check skipped exactly the turns that had just
        # committed work (committing clears dirtiness without clearing risk), and
        # it failed open — a held index.lock silently disabled the gate. A
        # `stop_hook_active` check made the gate one-shot: it blocked once, then
        # let the next Stop through without re-running anything, so it never
        # verified its own fix. Claude Code already caps a Stop hook at 8
        # consecutive blocks, so neither guard was needed.
        stop = self._commands("Stop")
        assert "git status" not in stop
        assert "stop_hook_active" not in stop

    def test_format_hook_covers_python_and_markdown(self) -> None:
        # ruff >= 0.16 formats python fences inside Markdown and CI's
        # `ruff format --check .` reaches CLAUDE.md, so .md needs the same
        # exit-2 contract .py gets or its failures vanish.
        post = self._commands("PostToolUse")
        assert "*.py" in post
        assert "*.md" in post
        assert "exit 2" in post

    def test_secret_guard_covers_env_lockfile_and_secrets(self) -> None:
        # An accident guardrail, not a security boundary: it matches Edit/Write/
        # MultiEdit only, never Bash. The .env.example exemption keeps the
        # committed template editable.
        pre = self._commands("PreToolUse")
        for pattern in ("*/.env", "*/secrets.toml", "*/uv.lock"):
            assert pattern in pre, f"unguarded path: {pattern}"
        assert "*/.env.example" in pre
        assert "exit 2" in pre
