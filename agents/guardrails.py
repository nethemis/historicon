"""Guardrails for the HistoriCon podcast.

On-topic filter — zero-shot classifier (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
via transformers) that blocks queries unrelated to Greek/Cypriot history.
The model is multilingual and handles Greek and Cypriot dialect queries natively.

The classifier is loaded lazily on the first real query (~280 MB, cached).

Usage:
    is_ok, error_msg = check_on_topic(query)
    if not is_ok:
        return {"error": error_msg, ...}

In tests, pass a mock classifier to avoid triggering the real model load:
    check_on_topic(query, classifier=my_mock)
"""

from functools import lru_cache
import re

from opentelemetry import trace
import pysbd

from agents import otel_setup

# Get tracer for this module
_tracer = otel_setup.get_tracer("historicon.guardrails")

# ─── Candidate labels for zero-shot classification ────────────────────────────

# Short, clean labels give the mDeBERTa NLI model much better discrimination
# than long bilingual descriptions (which push scores toward 0.5/0.5).
# Combined with _ON_TOPIC_HYPOTHESIS_TEMPLATE the NLI hypotheses become e.g.
# "This question is about Greek or Cypriot history." — clear and concise.
_ON_TOPIC_LABEL = "Greek or Cypriot history"
_OFF_TOPIC_LABEL = "an unrelated topic"

_CANDIDATE_LABELS = [_ON_TOPIC_LABEL, _OFF_TOPIC_LABEL]

# Hypothesis template used when calling the zero-shot pipeline.
# Produces: "This question is about Greek or Cypriot history."
_ON_TOPIC_HYPOTHESIS_TEMPLATE = "This question is about {}."

# Minimum confidence required before blocking a query as off-topic.
# With short labels the model gives clearer signal so 0.52 is sufficient.
_OFF_TOPIC_MIN_SCORE = 0.52
# Alias kept for backward-compatibility with evals.
_OFF_TOPIC_THRESHOLD = _OFF_TOPIC_MIN_SCORE

_OFF_TOPIC_MSG = (
    "Η ερώτησή σου δεν σχετίζεται με το podcast HistoriCon. "
    "Παρακαλώ ρώτα για την ελληνική ή κυπριακή ιστορία, τα επεισόδια ή τους ομιλητές. "
    "/ This question is outside the scope of the HistoriCon podcast. "
    "Please ask about Greek or Cypriot history, the podcast episodes, or its speakers."
)

# ─── Podcast-content keyword pre-check ───────────────────────────────────────

# If the query contains any of these terms it explicitly references the podcast
# or its episodes and is always on-topic — NLI is skipped entirely.
# The list is intentionally short to avoid false positives.
_PODCAST_KEYWORDS: frozenset[str] = frozenset(
    {
        "επεισόδιο",  # Greek: episode (singular)
        "επεισόδια",  # Greek: episodes (plural)
        "podcast",
        "historicon",
        "episode",
    }
)

# ─── Grounding check constants (output validation) ──────────────────────────

# Direct NLI approach: context is the premise, each sentence/response is the
# hypothesis (hypothesis_template="{}"). The model returns a sigmoid-normalised
# entailment score via multi_label=True. Scores below this threshold mean the
# context does NOT entail the sentence → likely hallucinated.
_GROUNDING_ENTAILMENT_MIN = 0.45

# mDeBERTa max sequence length is 512 tokens, shared between premise (context)
# and hypothesis (response sentence) plus 3 special tokens.
# Greek/multilingual text ≈ 3 chars/token → 900 chars ≈ 300 tokens for context,
# leaving ~200 tokens for the hypothesis sentence, well within the 512-token limit.
# At 1500 chars the context alone consumed ~500 tokens, truncating the hypothesis
# to near-nothing and making every sentence appear ungrounded.
_MAX_CONTEXT_CHARS = 900

# ─── Sentence-level grounding constants ──────────────────────────────────────

# Sentences shorter than this (chars) are too ambiguous for NLI and are skipped.
_MIN_SENTENCE_LENGTH = 20

# Maximum number of sentences to check per response to cap latency.
_MAX_SENTENCES_TO_CHECK = 8

# Markdown structural patterns — lines matching these are formatting, not claims.
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s")
_MARKDOWN_HR_RE = re.compile(r"^[\-\*_]{3,}\s*$")

_NOT_GROUNDED_MSG = (
    "⚠️ Μέρος αυτής της απάντησης μπορεί να μην βασίζεται αποκλειστικά στο περιεχόμενο "
    "του podcast. Παρακαλώ επαλήθευσε τις πληροφορίες με το πρωτότυπο επεισόδιο. "
    "/ ⚠️ Part of this response may not be fully grounded in the retrieved podcast content. "
    "Please verify the information against the original episode."
)


def _contains_podcast_keyword(query: str) -> bool:
    """Return True if *query* explicitly references the podcast or its episodes.

    Case-insensitive substring match against _PODCAST_KEYWORDS. A match means
    the query is asking about podcast content and should always be allowed
    through — NLI classification is skipped.
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in _PODCAST_KEYWORDS)


def _load_classifier():
    """Load the zero-shot classification pipeline (~280 MB, called once)."""
    from transformers import pipeline  # type: ignore[import]

    with _tracer.start_as_current_span("load_classifier") as span:
        print(
            "Loading zero-shot classification model (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)"
        )
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
        span.set_attribute("model", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        span.set_attribute("status", "loaded")
        return classifier


@lru_cache(maxsize=1)
def get_classifier():
    """Lazy-loaded classifier singleton — heavy load deferred to first query."""
    return _load_classifier()


def check_on_topic(query: str, classifier=None) -> tuple[bool, str]:
    """Return (is_on_topic, error_message). Fails open on classifier errors.

    Pass a mock classifier= in tests to avoid loading the real model.

    Uses multi_label=False (softmax over both labels) so the winning label is
    always clear. A query is blocked only when the off-topic label wins AND
    its score meets _OFF_TOPIC_MIN_SCORE. Anything ambiguous passes through
    (fail-open) to avoid blocking legitimate on-topic questions.
    """
    with _tracer.start_as_current_span("check_on_topic") as span:
        span.set_attribute("query_preview", query[:60])

        if classifier is None:
            classifier = get_classifier()

        # Pre-check: queries explicitly referencing the podcast or its episodes are
        # always on-topic — skip NLI to avoid false negatives from the model.
        if _contains_podcast_keyword(query):
            span.set_attribute("result", "on_topic_keyword_match")
            return True, ""

        try:
            result = classifier(
                query,
                _CANDIDATE_LABELS,
                hypothesis_template=_ON_TOPIC_HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )

            # Build a score lookup by label name
            scores_by_label: dict[str, float] = dict(
                zip(result["labels"], result["scores"])
            )
            on_topic_score = scores_by_label.get(_ON_TOPIC_LABEL, 0.0)
            off_topic_score = scores_by_label.get(_OFF_TOPIC_LABEL, 0.0)

            span.set_attribute("on_topic_score", on_topic_score)
            span.set_attribute("off_topic_score", off_topic_score)

            # Block only when off-topic wins with sufficient confidence
            if off_topic_score > on_topic_score and off_topic_score >= _OFF_TOPIC_MIN_SCORE:
                span.set_attribute("result", "off_topic_blocked")
                return False, _OFF_TOPIC_MSG

            span.set_attribute("result", "on_topic_allowed")
            return True, ""

        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("result", "error_fail_open")
            return True, ""  # fail open — don't block legitimate queries on errors


def check_grounding(
    response: str, context_chunks: list[str], classifier=None
) -> tuple[bool, str]:
    """Return (is_grounded, warning_message). Fails open on errors.

    Direct NLI approach: the retrieved context is the *premise* and the
    response (or individual sentence) is the *hypothesis*. The zero-shot
    pipeline is called with hypothesis_template="{}" and multi_label=True so
    the model returns a sigmoid-normalised entailment probability in [0, 1].

    - High score (>= _GROUNDING_ENTAILMENT_MIN) → context entails the response
      → grounded.
    - Low score (< _GROUNDING_ENTAILMENT_MIN) → context does NOT entail the
      response → possibly hallucinated → (False, warning).

    Fails open (returns True) on empty context, empty response, or any error.
    Pass a mock classifier= in tests to avoid loading the real model.
    """
    with _tracer.start_as_current_span("check_grounding") as span:
        span.set_attribute("response_preview", response[:60])
        span.set_attribute("num_context_chunks", len(context_chunks))

        if not context_chunks:
            span.set_attribute("result", "no_context_fail_open")
            return True, ""  # no context to compare against — fail open

        if not response.strip():
            span.set_attribute("result", "empty_response_fail_open")
            return True, ""  # no response to evaluate — fail open

        if classifier is None:
            classifier = get_classifier()

        try:
            hypothesis = response.strip()

            # Check each context chunk independently and take the maximum entailment
            # score. A sentence is grounded if ANY retrieved chunk entails it —
            # this avoids penalising sentences whose evidence appears in a later
            # chunk that would otherwise be truncated when all chunks are joined.
            max_score = 0.0
            chunks_checked = 0
            for chunk in context_chunks:
                context = chunk[:_MAX_CONTEXT_CHARS]
                result = classifier(
                    context,
                    [hypothesis],
                    hypothesis_template="{}",
                    multi_label=True,
                )
                score: float = result["scores"][0]
                chunks_checked += 1
                if score > max_score:
                    max_score = score
                if max_score >= _GROUNDING_ENTAILMENT_MIN:
                    break  # already grounded — skip remaining chunks

            span.set_attribute("entailment_score", max_score)
            span.set_attribute("chunks_checked", chunks_checked)

            if max_score < _GROUNDING_ENTAILMENT_MIN:
                span.set_attribute("result", "not_grounded")
                return False, _NOT_GROUNDED_MSG

            span.set_attribute("result", "grounded")
            return True, ""

        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("result", "error_fail_open")
            return True, ""  # fail open


def _detect_language(text: str) -> str:
    """Return 'el' if text is predominantly Greek, else 'en'.

    Counts Greek Unicode characters (Basic Greek + Extended blocks) as a
    fraction of all alphabetic characters. Responses from this podcast are
    typically >80% Greek; English replies contain <5% Greek chars.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return "en"
    greek_chars = [
        c for c in alpha_chars if "\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff"
    ]
    return "el" if len(greek_chars) / len(alpha_chars) > 0.2 else "en"


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using pysbd.

    Auto-detects language (Greek vs English) so both podcast-language
    responses (Greek) and English replies are segmented correctly. Sentences
    shorter than _MIN_SENTENCE_LENGTH chars are discarded — they are too
    short for the NLI model to make a reliable entailment judgement.
    """
    lang = _detect_language(text)
    segmenter = pysbd.Segmenter(language=lang, clean=True)
    parts = segmenter.segment(text.strip())
    return [
        s.strip()
        for s in parts
        if len(s.strip()) >= _MIN_SENTENCE_LENGTH
        and not _MARKDOWN_HEADING_RE.match(s.strip())
        and not _MARKDOWN_HR_RE.match(s.strip())
    ]


def check_grounding_detailed(
    response: str, context_chunks: list[str], classifier=None
) -> tuple[bool, str, list[str]]:
    """Sentence-level grounding check. Returns (is_grounded, message, unverified_sentences).

    Splits the response into sentences and runs check_grounding on each one
    individually. Returns the list of sentences that could not be verified
    against the retrieved context. Falls back to whole-response check when no
    checkable sentences are found (all too short).

    Caps at _MAX_SENTENCES_TO_CHECK to limit latency on long responses.
    Fails open on empty context, empty response, or any error.
    """
    with _tracer.start_as_current_span("check_grounding_detailed") as span:
        span.set_attribute("response_preview", response[:60])
        span.set_attribute("num_context_chunks", len(context_chunks))

        if not context_chunks:
            span.set_attribute("result", "no_context_fail_open")
            return True, "", []

        if not response.strip():
            span.set_attribute("result", "empty_response_fail_open")
            return True, "", []

        if classifier is None:
            classifier = get_classifier()

        sentences = _split_sentences(response)[:_MAX_SENTENCES_TO_CHECK]

        if not sentences:
            # All sentences were too short — fall back to whole-response binary check
            is_ok, msg = check_grounding(response, context_chunks, classifier)
            span.set_attribute("result", "fallback_whole_response")
            return is_ok, msg, []

        unverified: list[str] = []
        for sentence in sentences:
            is_ok, _ = check_grounding(sentence, context_chunks, classifier)
            if not is_ok:
                unverified.append(sentence)

        span.set_attribute("sentences_checked", len(sentences))
        span.set_attribute("unverified_count", len(unverified))

        if unverified:
            span.set_attribute("result", "unverified_sentences_found")
            return False, _NOT_GROUNDED_MSG, unverified

        span.set_attribute("result", "all_sentences_grounded")
        return True, "", []
