"""
CrossCheck AI — transparent evidence fusion.

Important:
- CLIP similarity is not a truth score.
- NLI compares a generated description with the user claim.
- Generated descriptions can hallucinate or omit details.
- Agreement is model-supported, not independently verified truth.
- Specific grounded visual-conflict detection is not implemented.
- Thresholds below are unvalidated prototype heuristics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Retained for compatibility/documentation.
# HIGH_CLIP_THRESHOLD is informational, not an agreement gate.
HIGH_CLIP_THRESHOLD = 0.26
MIN_CLIP_FOR_CONTRADICTION = 0.18

NLI_ENTAIL_THRESHOLD = 0.80
NLI_CONTRA_THRESHOLD = 0.80
NLI_REQUIRED_MARGIN = 0.20


@dataclass
class AnalysisResult:
    label: str
    explanation: str
    claim: str
    generated_description: Optional[str]

    evidence: List[Dict[str, Any]] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    abstention_reasons: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    review_recommended: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "explanation": self.explanation,
            "claim": self.claim,
            "generated_description": self.generated_description,
            "evidence": self.evidence,
            "signals": self.signals,
            "abstention_reasons": self.abstention_reasons,
            "limitations": self.limitations,
            "review_recommended": self.review_recommended,
        }


LIMITATIONS = [
    "CLIP cosine similarity measures semantic alignment, not factual truth.",
    "BLIP descriptions can hallucinate or omit relevant details.",
    "NLI compares text; it does not directly inspect the image.",
    "Agreement applies only to model-described content.",
    "Thresholds are unvalidated prototype heuristics.",
    "External-context detection is limited and may miss unsupported claims.",
    "Specific grounded visual-conflict verification is not implemented.",
    "No benchmark accuracy or calibrated confidence is claimed.",
]


def external_context_flags(claim: str) -> List[str]:
    """
    Limited abstention safeguard, not a complete claim parser.

    A match means extra context may be needed; it does not mean
    the claim is false.
    """
    patterns = {
        "relative date or event history": (
            r"\b(yesterday|tomorrow|last week|last month|last year)\b"
        ),
        "specific calendar year": r"\b(?:19|20)\d{2}\b",
        "event outcome": (
            r"\b(won|winner|champion|competition|award|elected)\b"
        ),
        "provenance or authenticity": (
            r"\b(original source|authentic|fake|manipulated|photoshopped)\b"
        ),
        "causal assertion": r"\b(because|caused by|responsible for)\b",
    }

    return [
        name
        for name, pattern in patterns.items()
        if re.search(pattern, claim, flags=re.IGNORECASE)
    ]


def valid_nli(scores: Optional[Dict[str, float]]) -> bool:
    required = {"entailment", "contradiction", "neutral"}

    if not isinstance(scores, dict) or not required.issubset(scores):
        return False

    values = [scores[key] for key in required]

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
        for value in values
    ):
        return False

    return abs(sum(values) - 1.0) <= 0.02


def decide(
    claim: str,
    clip_score: Optional[float],
    caption: Optional[str],
    nli_scores: Optional[Dict[str, float]] = None,
    clip_error: Optional[str] = None,
    caption_error: Optional[str] = None,
    nli_error: Optional[str] = None,
) -> AnalysisResult:

    claim = claim.strip()
    caption = caption.strip() if isinstance(caption, str) else None
    caption = caption or None

    result = AnalysisResult(
        label="INSUFFICIENT_EVIDENCE",
        explanation="The available evidence does not support a decision.",
        claim=claim,
        generated_description=caption,
        limitations=list(LIMITATIONS),
    )

    result.evidence.append({
        "source": "user_claim",
        "type": "user_provided",
        "content": claim,
    })

    for stage, error in (
        ("CLIP", clip_error),
        ("BLIP", caption_error),
        ("NLI", nli_error),
    ):
        if error:
            result.abstention_reasons.append(
                f"{stage} unavailable: {error}"
            )

    clip_valid = (
        isinstance(clip_score, (int, float))
        and not isinstance(clip_score, bool)
        and math.isfinite(clip_score)
        and -1 <= clip_score <= 1
    )

    if clip_valid:
        result.signals["clip_cosine_similarity"] = float(clip_score)

        result.evidence.append({
            "source": "CLIP",
            "type": "model_signal",
            "content": (
                f"Image–claim cosine similarity: {clip_score:.4f}. "
                "Recorded as supporting information, not a truth score "
                "or standalone agreement gate."
            ),
        })
    else:
        result.abstention_reasons.append(
            "A valid CLIP alignment measurement is unavailable."
        )

    if caption:
        result.signals["blip_caption"] = caption

        result.evidence.append({
            "source": "BLIP",
            "type": "model_inferred_observation",
            "content": caption,
        })
    else:
        result.abstention_reasons.append(
            "No model-generated visual description is available."
        )

    scores_valid = valid_nli(nli_scores)

    if scores_valid:
        result.signals["nli_scores"] = {
            key: float(nli_scores[key])
            for key in ("entailment", "contradiction", "neutral")
        }

        result.signals["nli_thresholds"] = {
            "entailment_for_consistent": NLI_ENTAIL_THRESHOLD,
            "contradiction_for_contradictory": NLI_CONTRA_THRESHOLD,
            "required_margin": NLI_REQUIRED_MARGIN,
        }

        result.evidence.append({
            "source": "NLI",
            "type": "textual_comparison",
            "content": (
                f"Entailment={nli_scores['entailment']:.4f}; "
                f"contradiction={nli_scores['contradiction']:.4f}; "
                f"neutral={nli_scores['neutral']:.4f}. "
                "Premise: generated description. Hypothesis: user claim."
            ),
        })
    else:
        result.abstention_reasons.append(
            "Valid entailment, contradiction, and neutral scores "
            "are unavailable."
        )

    context_flags = external_context_flags(claim)

    if context_flags:
        result.abstention_reasons.append(
            "The claim may require external evidence: "
            + ", ".join(context_flags)
            + "."
        )

    if not claim:
        result.abstention_reasons.append("The claim is empty.")

    # Stop before making a stronger assessment if required stages
    # failed, inputs are missing, or external context is flagged.
    if result.abstention_reasons:
        result.explanation = (
            "The system cannot complete a supported comparison. "
            "This does not establish that the claim is wrong. "
            "See the specific abstention reasons."
        )
        return result

    entailment = nli_scores["entailment"]
    contradiction = nli_scores["contradiction"]
    neutral = nli_scores["neutral"]

    margin = entailment - max(contradiction, neutral)
    result.signals["entailment_margin"] = float(margin)

    # Strongly negative image alignment is treated conservatively.
    # This is a safety rule, not a calibrated contradiction boundary.
    if clip_score <= 0:
        result.abstention_reasons.append(
            "Non-positive CLIP alignment conflicts with issuing "
            "a supported agreement assessment."
        )
        result.explanation = (
            "Image alignment is weak or conflicting. "
            "The system abstains rather than using textual agreement alone."
        )
        return result

    if (
        entailment >= NLI_ENTAIL_THRESHOLD
        and margin >= NLI_REQUIRED_MARGIN
    ):
        result.label = "CONSISTENT"
        result.explanation = (
            "1. A visual description was generated from the uploaded image.\n\n"
            f"2. NLI supports that description entailing the claim "
            f"(raw entailment score {entailment:.4f}). Exact wording "
            "is not required.\n\n"
            f"3. Entailment exceeds both competing textual signals "
            f"by {margin:.4f}, meeting the configured heuristic margin.\n\n"
            f"4. CLIP alignment was measured at {clip_score:.4f}; "
            "it is supporting information, not factual confidence.\n\n"
            "This is model-supported consistency with the generated "
            "description—not independent verification of the full claim."
        )
        return result

    if (
        contradiction >= NLI_CONTRA_THRESHOLD
        and contradiction > max(entailment, neutral)
    ):
        result.abstention_reasons.append(
            "NLI detected a possible conflict between the description "
            "and claim, but specific visual-conflict verification "
            "is not implemented."
        )
        result.explanation = (
            "A possible textual conflict needs human investigation. "
            "The system does not treat generated-caption disagreement "
            "as proof of a visual contradiction."
        )
        return result

    result.abstention_reasons.append(
        "Textual entailment was not sufficiently strong and dominant: "
        f"entailment={entailment:.4f}, "
        f"contradiction={contradiction:.4f}, "
        f"neutral={neutral:.4f}, "
        f"entailment margin={margin:.4f}."
    )
    result.explanation = (
        "The generated description does not sufficiently support "
        "the complete claim under the configured rules. "
        "It may omit relevant details; the claim is not established "
        "as wrong."
    )

    return result