"""
Tests for the revised decision policy.

Policy:
- CLIP is supporting information, not a hard 0.26 agreement gate.
- Strong, dominant textual entailment can support consistency
  with a generated description.
- Missing/invalid signals and model errors require abstention.
- Caption/NLI disagreement alone does not prove visual contradiction.

These tests use synthetic model scores.
They do NOT evaluate pretrained-model accuracy.
"""

import pytest

from src.decision import (
    AnalysisResult,
    NLI_ENTAIL_THRESHOLD,
    decide,
)


CLAIM = "A dog is standing on grass."
CAPTION = "A dog is standing on green grass."

STRONG_ENTAILMENT = {
    "entailment": 0.95,
    "contradiction": 0.02,
    "neutral": 0.03,
}

STRONG_CONTRADICTION = {
    "entailment": 0.02,
    "contradiction": 0.95,
    "neutral": 0.03,
}


def analyze(**overrides):
    inputs = {
        "claim": CLAIM,
        "clip_score": 0.2812,
        "caption": CAPTION,
        "nli_scores": dict(STRONG_ENTAILMENT),
    }
    inputs.update(overrides)
    return decide(**inputs)


def test_result_schema():
    result = analyze()

    assert isinstance(result, AnalysisResult)

    expected = {
        "label",
        "explanation",
        "claim",
        "generated_description",
        "evidence",
        "signals",
        "abstention_reasons",
        "limitations",
        "review_recommended",
    }

    assert expected.issubset(result.to_dict())
    assert result.claim == CLAIM
    assert result.generated_description == CAPTION
    assert result.review_recommended is True
    assert result.limitations


def test_strong_supported_entailment_is_consistent():
    result = analyze()

    assert result.label == "CONSISTENT"
    assert not result.abstention_reasons
    assert "model-supported" in result.explanation.lower()


@pytest.mark.parametrize("clip_score", [0.21, 0.259, 0.26, 0.2812])
def test_old_clip_cutoff_is_not_an_agreement_gate(clip_score):
    """
    Tests the revised policy, not model accuracy.

    Strong textual entailment is not rejected solely because
    CLIP is slightly below the former 0.26 boundary.
    """
    result = analyze(clip_score=clip_score)

    assert result.label == "CONSISTENT"


@pytest.mark.parametrize(
    "clip_score",
    [None, -0.1, 0.0, float("nan"), float("inf"), 1.1],
)
def test_missing_invalid_or_nonpositive_clip_abstains(clip_score):
    result = analyze(clip_score=clip_score)

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert result.abstention_reasons


@pytest.mark.parametrize("caption", [None, "", "   "])
def test_missing_description_abstains(caption):
    result = analyze(caption=caption)

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert result.abstention_reasons


@pytest.mark.parametrize(
    "scores",
    [
        None,
        {},
        {"entailment": 0.95},
        {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        },
        {
            "entailment": float("nan"),
            "contradiction": 0.02,
            "neutral": 0.03,
        },
        {
            "entailment": 1.2,
            "contradiction": -0.2,
            "neutral": 0.0,
        },
    ],
)
def test_invalid_nli_abstains(scores):
    result = analyze(nli_scores=scores)

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert result.abstention_reasons


def test_high_clip_alone_does_not_establish_consistency():
    result = analyze(
        clip_score=0.40,
        nli_scores=None,
    )

    assert result.label == "INSUFFICIENT_EVIDENCE"


def test_neutral_textual_comparison_abstains():
    result = analyze(
        nli_scores={
            "entailment": 0.20,
            "contradiction": 0.05,
            "neutral": 0.75,
        }
    )

    assert result.label == "INSUFFICIENT_EVIDENCE"


def test_entailment_at_configured_boundary():
    entailment = NLI_ENTAIL_THRESHOLD

    result = analyze(
        nli_scores={
            "entailment": entailment,
            "contradiction": 0.05,
            "neutral": 1.0 - entailment - 0.05,
        }
    )

    assert result.label == "CONSISTENT"


def test_entailment_below_configured_boundary_abstains():
    entailment = NLI_ENTAIL_THRESHOLD - 0.001

    result = analyze(
        nli_scores={
            "entailment": entailment,
            "contradiction": 0.05,
            "neutral": 1.0 - entailment - 0.05,
        }
    )

    assert result.label == "INSUFFICIENT_EVIDENCE"


def test_textual_conflict_is_not_visual_proof():
    result = analyze(
        caption="A red bus is parked on a city street.",
        nli_scores=dict(STRONG_CONTRADICTION),
    )

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert any(
        "visual" in reason.lower()
        for reason in result.abstention_reasons
    )


def test_low_similarity_does_not_prove_contradiction():
    result = analyze(
        clip_score=0.05,
        nli_scores=dict(STRONG_CONTRADICTION),
    )

    assert result.label == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "claim",
    [
        "This dog won a competition yesterday.",
        "This photograph was taken in 2024.",
        "This is the original source of the photograph.",
    ],
)
def test_recognized_external_context_abstains(claim):
    """
    Tests selected heuristic flags only.
    Does not establish comprehensive external-context detection.
    """
    result = analyze(claim=claim)

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert any(
        "external evidence" in reason.lower()
        for reason in result.abstention_reasons
    )


@pytest.mark.parametrize(
    "error_key, stage",
    [
        ("clip_error", "CLIP"),
        ("caption_error", "BLIP"),
        ("nli_error", "NLI"),
    ],
)
def test_errors_override_otherwise_strong_signals(error_key, stage):
    result = analyze(**{error_key: "Simulated inference failure"})

    assert result.label == "INSUFFICIENT_EVIDENCE"
    assert any(
        stage in reason
        for reason in result.abstention_reasons
    )


def test_empty_claim_abstains():
    result = analyze(claim="   ")

    assert result.label == "INSUFFICIENT_EVIDENCE"


def test_actual_signals_are_preserved():
    result = analyze()

    assert result.signals["clip_cosine_similarity"] == 0.2812
    assert result.signals["blip_caption"] == CAPTION
    assert result.signals["nli_scores"] == STRONG_ENTAILMENT


def test_no_measurements_are_fabricated():
    result = decide(
        claim=CLAIM,
        clip_score=None,
        caption=None,
        nli_scores=None,
    )

    assert "clip_cosine_similarity" not in result.signals
    assert "blip_caption" not in result.signals
    assert "nli_scores" not in result.signals


def test_evidence_records_include_user_claim():
    result = analyze()

    assert any(
        record["source"] == "user_claim"
        and record["content"] == CLAIM
        for record in result.evidence
    )


def test_calls_do_not_share_mutable_limitations():
    first = analyze()
    second = analyze()

    first.limitations.append("Test-only modification")

    assert "Test-only modification" not in second.limitations
