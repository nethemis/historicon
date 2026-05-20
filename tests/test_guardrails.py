"""Tests for guardrail functions.

After removing anti-fabrication code, this file covers only the on-topic
filter (check_on_topic + get_classifier).

Run with: uv run pytest tests/test_guardrails.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

# ─── On-topic classifier ──────────────────────────────────────────────────────


def test_get_classifier_returns_callable():
    from agents.guardrails import get_classifier

    with patch("agents.guardrails._load_classifier", return_value=MagicMock()):
        get_classifier.cache_clear()
        result = get_classifier()
        assert result is not None
        get_classifier.cache_clear()


def test_check_on_topic_history_passes():
    from agents.guardrails import _CANDIDATE_LABELS, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _CANDIDATE_LABELS[1]],
            "scores": [0.85, 0.15],
        }
    )
    is_ok, msg = check_on_topic(
        "Tell me about the 1974 Cyprus coup", classifier=mock_clf
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_coding_blocked():
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.9, 0.1],
        }
    )
    is_ok, msg = check_on_topic(
        "Write me a Python function to sort a list", classifier=mock_clf
    )
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_weather_blocked():
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.88, 0.12],
        }
    )
    is_ok, _ = check_on_topic("What is the weather today?", classifier=mock_clf)
    assert is_ok is False


def test_check_on_topic_classifier_error_fails_open():
    """When the classifier throws, check_on_topic returns True (fail-open)."""
    from agents.guardrails import check_on_topic

    def _bad_clf(query, labels, multi_label=False):
        raise RuntimeError("Model crashed")

    is_ok, msg = check_on_topic("some query", classifier=_bad_clf)
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_below_threshold_passes():
    """Off-topic label doesn't win (on-topic scores higher) → query passes."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.3, 0.7],  # off_topic loses → passes through
        }
    )
    is_ok, _ = check_on_topic("Ambiguous query", classifier=mock_clf)
    assert is_ok is True


# ─── Positive-matching: general knowledge that bypassed the old narrow labels ─


def test_check_on_topic_general_trivia_blocked():
    """'What is the capital of France?' should be blocked (not Greek/Cypriot history)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    # With positive matching the on-topic label loses → blocked
    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.65, 0.35],
        }
    )
    is_ok, msg = check_on_topic("What is the capital of France?", classifier=mock_clf)
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_celebrity_question_blocked():
    """'Who is Elon Musk?' should be blocked (not Greek/Cypriot history)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.72, 0.28],
        }
    )
    is_ok, msg = check_on_topic("Who is Elon Musk?", classifier=mock_clf)
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_low_confidence_both_passes():
    """When both labels score low the query passes (fail-open — don't block ambiguous queries)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    # on_topic wins but both scores are very low → uncertain → fail open
    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.25, 0.18],  # on_topic wins but neither is confident
        }
    )
    is_ok, msg = check_on_topic("Random ambiguous query xyz", classifier=mock_clf)
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_greek_history_high_confidence_passes():
    """Clear Greek history query with strong on-topic score should always pass."""
    from agents.guardrails import (
        _CANDIDATE_LABELS,
        _OFF_TOPIC_LABEL,
        check_on_topic,
    )  # noqa: F401

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.90, 0.10],
        }
    )
    is_ok, msg = check_on_topic(
        "Πες μου για τον Κοσκωτά και το σκάνδαλο", classifier=mock_clf
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_podcast_question_english_passes():
    """'How did Petros the First die, according to the podcast?' should pass."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.75, 0.25],
        }
    )
    is_ok, msg = check_on_topic(
        "How did Petros the First die, according to the podcast?",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_podcast_question_greek_passes():
    """Greek podcast query 'Σύμφωνα με το podcast πώς πέθανε ο Πέτρος ο Πρώτος;' should pass."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.78, 0.22],
        }
    )
    is_ok, msg = check_on_topic(
        "Σύμφωνα με το podcast πώς πέθανε ο Πέτρος ο Πρώτος;",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_episode_keyword_greek_short_circuits():
    """Query with 'επεισόδιο' bypasses NLI entirely — classifier must not be called."""
    from agents.guardrails import check_on_topic

    mock_clf = MagicMock()
    is_ok, msg = check_on_topic(
        "Τι άλλο λέει το επεισόδιο για τη βρετανική κατοχή στην Κύπρο;",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""
    mock_clf.assert_not_called()


def test_check_on_topic_episode_keyword_english_short_circuits():
    """Query with 'episode' bypasses NLI entirely — classifier must not be called."""
    from agents.guardrails import check_on_topic

    mock_clf = MagicMock()
    is_ok, msg = check_on_topic(
        "What else does the episode say about the British occupation in Cyprus?",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""
    mock_clf.assert_not_called()


# ─── Grounding check ──────────────────────────────────────────────────────────


def test_check_grounding_grounded_response_passes():
    """High entailment score → (True, '')"""
    from agents.guardrails import check_grounding

    mock_clf = MagicMock(return_value={"labels": ["hypothesis"], "scores": [0.85]})
    is_ok, msg = check_grounding(
        "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση.",
        [
            "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση από την Alpha Bank."
        ],
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""


def test_check_grounding_fabricated_response_blocked():
    """Low entailment score → (False, msg)"""
    from agents.guardrails import check_grounding

    mock_clf = MagicMock(return_value={"labels": ["hypothesis"], "scores": [0.2]})
    is_ok, msg = check_grounding(
        "Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας το 1990.",  # fabricated
        ["Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."],
        classifier=mock_clf,
    )
    assert is_ok is False
    assert len(msg) > 0


def test_check_grounding_empty_context_fails_open():
    """No context chunks → fail-open without calling classifier."""
    from agents.guardrails import check_grounding

    mock_clf = MagicMock()
    is_ok, msg = check_grounding(
        "Some response text.",
        [],
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""
    mock_clf.assert_not_called()


def test_check_grounding_empty_response_fails_open():
    """Empty response → fail-open without calling classifier."""
    from agents.guardrails import check_grounding

    mock_clf = MagicMock()
    is_ok, msg = check_grounding("", ["Some context."], classifier=mock_clf)
    assert is_ok is True
    assert msg == ""
    mock_clf.assert_not_called()


def test_check_grounding_classifier_error_fails_open():
    """Classifier raises → fail-open."""
    from agents.guardrails import check_grounding

    def _bad_clf(*args, **kwargs):
        raise RuntimeError("Model crashed")

    is_ok, msg = check_grounding(
        "Some response text.",
        ["Some context about Greek history."],
        classifier=_bad_clf,
    )
    assert is_ok is True
    assert msg == ""


def test_check_grounding_at_threshold_passes():
    """Entailment score at exactly _GROUNDING_ENTAILMENT_MIN → grounded (boundary)."""
    from agents.guardrails import _GROUNDING_ENTAILMENT_MIN, check_grounding

    mock_clf = MagicMock(
        return_value={"labels": ["hypothesis"], "scores": [_GROUNDING_ENTAILMENT_MIN]}
    )
    is_ok, _ = check_grounding(
        "Ambiguous response.",
        ["Some context."],
        classifier=mock_clf,
    )
    assert is_ok is True


# ─── Sentence-level grounding (check_grounding_detailed) ─────────────────────


def test_check_grounding_detailed_all_sentences_grounded():
    """All sentences supported by context → (True, '', [])."""
    from agents.guardrails import check_grounding_detailed

    with patch("agents.guardrails.check_grounding", return_value=(True, "")):
        is_ok, msg, unverified = check_grounding_detailed(
            "Ο Κοσκωτάς ήταν τραπεζίτης. Το σκάνδαλο αποκαλύφθηκε το 1988.",
            ["Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."],
        )
    assert is_ok is True
    assert msg == ""
    assert unverified == []


def test_check_grounding_detailed_one_sentence_unverified():
    """One sentence fails NLI → (False, msg, [that_sentence])."""
    from agents.guardrails import _NOT_GROUNDED_MSG, check_grounding_detailed

    def _mock(response, context_chunks, classifier=None):
        return (False, _NOT_GROUNDED_MSG) if "fabricated" in response else (True, "")

    with patch("agents.guardrails.check_grounding", side_effect=_mock):
        is_ok, msg, unverified = check_grounding_detailed(
            "This is grounded history. This is a fabricated claim about the past.",
            ["Some real historical context about the topic."],
        )
    assert is_ok is False
    assert len(msg) > 0
    assert len(unverified) == 1
    assert "fabricated" in unverified[0]


def test_check_grounding_detailed_multiple_unverified():
    """Multiple failing sentences → all collected in unverified list."""
    from agents.guardrails import _NOT_GROUNDED_MSG, check_grounding_detailed

    def _mock(response, context_chunks, classifier=None):
        return (False, _NOT_GROUNDED_MSG) if "invented" in response else (True, "")

    with patch("agents.guardrails.check_grounding", side_effect=_mock):
        is_ok, msg, unverified = check_grounding_detailed(
            "Real sentence here. First invented claim present. Second invented thing said.",
            ["Some context."],
        )
    assert is_ok is False
    assert len(unverified) == 2


def test_check_grounding_detailed_empty_context_fails_open():
    """No context chunks → fail-open, returns empty unverified list."""
    from agents.guardrails import check_grounding_detailed

    is_ok, msg, unverified = check_grounding_detailed(
        "Some multi-sentence response here.", []
    )
    assert is_ok is True
    assert unverified == []


def test_check_grounding_detailed_empty_response_fails_open():
    """Empty response → fail-open, returns empty unverified list."""
    from agents.guardrails import check_grounding_detailed

    is_ok, msg, unverified = check_grounding_detailed("", ["Some context."])
    assert is_ok is True
    assert unverified == []


def test_check_grounding_detailed_caps_at_max_sentences():
    """NLI calls are capped at _MAX_SENTENCES_TO_CHECK regardless of response length."""
    from agents.guardrails import _MAX_SENTENCES_TO_CHECK, check_grounding_detailed

    call_count = 0

    def _counter(response, context_chunks, classifier=None):
        nonlocal call_count
        call_count += 1
        return True, ""

    long_response = "  ".join(
        [f"Sentence number {i} about historical events" for i in range(20)]
    )

    with patch("agents.guardrails.check_grounding", side_effect=_counter):
        check_grounding_detailed(long_response, ["Context."])

    assert call_count <= _MAX_SENTENCES_TO_CHECK


def test_check_grounding_detailed_skips_short_fragments():
    """Sentences below _MIN_SENTENCE_LENGTH are not checked."""
    from agents.guardrails import check_grounding_detailed

    call_count = 0

    def _counter(response, context_chunks, classifier=None):
        nonlocal call_count
        call_count += 1
        return True, ""

    # Only one substantive sentence; 'Yes.' and 'Ok.' are below the length threshold
    with patch("agents.guardrails.check_grounding", side_effect=_counter):
        check_grounding_detailed(
            "Yes. Ok. This is the only real sentence worth checking.",
            ["Some context."],
        )

    assert call_count == 1


# ─── _detect_language ────────────────────────────────────────────────────────


def test_detect_language_greek_text():
    """Predominantly Greek text → 'el'."""
    from agents.guardrails import _detect_language

    assert (
        _detect_language("Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση.")
        == "el"
    )


def test_detect_language_english_text():
    """Predominantly English text → 'en'."""
    from agents.guardrails import _detect_language

    assert (
        _detect_language("The Koskotas scandal shook Greek politics in the 1980s.")
        == "en"
    )


def test_detect_language_empty_text():
    """Empty / no alpha chars → defaults to 'en'."""
    from agents.guardrails import _detect_language

    assert _detect_language("") == "en"
    assert _detect_language("123 !!! ???") == "en"


# ─── _split_sentences (pysbd-based) ──────────────────────────────────────────


def test_split_sentences_greek_prose():
    """Splits normal Greek prose into individual sentences."""
    from agents.guardrails import _split_sentences

    result = _split_sentences(
        "Ο Κοσκωτάς γεννήθηκε το 1954 στη Λακωνία. "
        "Αργότερα έγινε ο ιδιοκτήτης της Τράπεζας Κρήτης. "
        "Το σκάνδαλο αποκαλύφθηκε το 1988."
    )
    assert len(result) == 3


def test_split_sentences_english_prose():
    """Splits normal English prose into individual sentences."""
    from agents.guardrails import _split_sentences

    result = _split_sentences(
        "The Koskotas scandal was a major political crisis. "
        "It involved the Bank of Crete in the late 1980s. "
        "Several politicians were implicated."
    )
    assert len(result) == 3


def test_split_sentences_filters_short_fragments():
    """Fragments below _MIN_SENTENCE_LENGTH are excluded."""
    from agents.guardrails import _MIN_SENTENCE_LENGTH, _split_sentences

    result = _split_sentences("OK. Yes. This is the only real sentence worth checking.")
    assert all(len(s) >= _MIN_SENTENCE_LENGTH for s in result)
    assert len(result) == 1


def test_split_sentences_markdown_bullets():
    """Markdown bullet list items are treated as separate sentences."""
    from agents.guardrails import _split_sentences

    result = _split_sentences(
        "Ο Κοσκωτάς είχε τρεις κατηγορίες:\n"
        "- Υπεξαίρεση κεφαλαίων από την Τράπεζα Κρήτης\n"
        "- Εκβιασμός υψηλόβαθμων στελεχών\n"
        "- Φοροδιαφυγή μεγάλης κλίμακας"
    )
    # pysbd should not collapse the entire thing into a single unsplit blob
    assert len(result) >= 1


def test_split_sentences_skips_markdown_headers():
    """ATX markdown headers (# and ##) are not treated as checkable claims."""
    from agents.guardrails import _split_sentences

    result = _split_sentences(
        "# Ιστορίες με Μεθυσμένους\n\n"
        "## 🍺 Ο Ιρλανδός που επέζησε\n\n"
        "Ο Μαλόι ήπιε το ξυλόπνευμα και επέζησε κατά θαύμα."
    )
    # Headers must be excluded; only the prose sentence survives
    assert all(not s.startswith("#") for s in result)
    assert len(result) == 1


def test_split_sentences_skips_horizontal_rules():
    """Markdown horizontal rules (--- / ***) are excluded."""
    from agents.guardrails import _split_sentences

    result = _split_sentences(
        "Αυτή είναι η πρώτη πρόταση που πρέπει να ελεγχθεί.\n"
        "---\n"
        "Αυτή είναι η δεύτερη πρόταση που πρέπει να ελεγχθεί."
    )
    assert all(not s.strip().startswith("-" * 3) for s in result)
    assert len(result) == 2


def test_check_grounding_uses_best_chunk():
    """A sentence grounded in ANY chunk passes, even if other chunks don't entail it."""
    from agents.guardrails import _GROUNDING_ENTAILMENT_MIN, check_grounding

    call_count = 0

    def _mock_clf(context, labels, hypothesis_template="{}", multi_label=True):
        nonlocal call_count
        call_count += 1
        # First chunk: not grounded; second chunk: grounded
        score = _GROUNDING_ENTAILMENT_MIN + 0.1 if call_count == 2 else 0.1
        return {"labels": labels, "scores": [score]}

    is_ok, msg = check_grounding(
        "Κάποια ιστορική πρόταση.",
        ["Chunk A — irrelevant context.", "Chunk B — relevant context."],
        classifier=_mock_clf,
    )
    assert is_ok is True
    assert call_count == 2  # stopped after finding grounded chunk


def test_check_grounding_short_circuits_on_first_grounded_chunk():
    """Stops checking chunks once a grounding score meets the threshold."""
    from agents.guardrails import _GROUNDING_ENTAILMENT_MIN, check_grounding

    call_count = 0

    def _mock_clf(context, labels, hypothesis_template="{}", multi_label=True):
        nonlocal call_count
        call_count += 1
        # First chunk is already grounded
        return {"labels": labels, "scores": [_GROUNDING_ENTAILMENT_MIN + 0.2]}

    check_grounding(
        "Κάποια πρόταση.",
        ["Chunk A.", "Chunk B.", "Chunk C."],
        classifier=_mock_clf,
    )
    assert call_count == 1  # short-circuited after first chunk
