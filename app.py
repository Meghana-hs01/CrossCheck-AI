"""
CrossCheck AI — Evidence Before Verdict.

Developer: MEGHANA H S
Team: NeuralForge
Institution: REVA University, CSE

Run:
python -m streamlit run app.py --server.fileWatcherType none
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

import streamlit as st

from src.analysis import run_blip_caption, run_clip, run_nli
from src.decision import decide
from src.validation import validate_and_decode_image, validate_claim


CLIP_ID = "openai/clip-vit-base-patch32"
BLIP_ID = "Salesforce/blip-image-captioning-base"
NLI_ID = "typeform/distilbert-base-uncased-mnli"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="CrossCheck AI",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------
# Model loading
# Failed loads are not cached as successful return values.
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading image–text alignment model...")
def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        CLIP_ID,
        use_fast=False,
    )
    model = CLIPModel.from_pretrained(
        CLIP_ID,
        use_safetensors=True,
    )
    model.eval()

    return processor, model


@st.cache_resource(show_spinner="Loading visual-description model...")
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(
        BLIP_ID,
        use_fast=False,
    )
    model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)
    model.eval()

    return processor, model


@st.cache_resource(show_spinner="Loading caption-comparison model...")
def load_nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLI_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_ID)
    model.eval()

    mapping = getattr(model.config, "id2label", {}) or {}
    labels = {
        str(value).strip().lower()
        for value in mapping.values()
    }

    required = {"entailment", "contradiction", "neutral"}

    if not required.issubset(labels):
        raise ValueError(
            "Unsupported NLI label mapping. "
            f"Expected {sorted(required)}, received {mapping}."
        )

    return tokenizer, model


# ---------------------------------------------------------
# Input and state helpers
# ---------------------------------------------------------

def exception_message(exc):
    return f"{type(exc).__name__}: {exc}"


def make_fingerprint(image_data, claim):
    digest = hashlib.sha256()
    digest.update(image_data)
    digest.update(b"\x00")
    digest.update(claim.encode("utf-8"))

    return digest.hexdigest()


def clear_analysis():
    for key in (
        "analysis_bundle",
        "review_assessment",
        "review_notes",
    ):
        st.session_state.pop(key, None)


def reset_inputs():
    clear_analysis()
    st.session_state["input_generation"] = (
        st.session_state.get("input_generation", 0) + 1
    )


def valid_nli_scores(scores):
    required = {"entailment", "contradiction", "neutral"}

    if not isinstance(scores, dict):
        return False

    if not required.issubset(scores):
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


# ---------------------------------------------------------
# Real inference pipeline
# ---------------------------------------------------------

def analyze_image_claim(image, claim):
    clip_score = None
    description = None
    nli_scores = None

    clip_error = None
    description_error = None
    nli_error = None

    with st.status(
        "Analyzing the image and caption...",
        expanded=True,
    ) as progress:

        st.write("1 / 3 — Measuring image–caption alignment")

        try:
            processor, model = load_clip()

            clip_score, clip_error = run_clip(
                image,
                claim,
                processor,
                model,
            )

            if clip_score is not None:
                if (
                    not isinstance(clip_score, (int, float))
                    or not math.isfinite(clip_score)
                    or not -1 <= clip_score <= 1
                ):
                    clip_score = None
                    clip_error = "CLIP returned an invalid similarity score."

            if clip_score is None and clip_error is None:
                clip_error = "CLIP returned no similarity score."

        except Exception as exc:
            clip_error = exception_message(exc)

        st.write("2 / 3 — Generating a description of visible content")

        try:
            processor, model = load_blip()

            description, description_error = run_blip_caption(
                image,
                processor,
                model,
                max_new_tokens=50,
            )

            if isinstance(description, str):
                description = description.strip() or None
            else:
                description = None

            if description is None and description_error is None:
                description_error = "BLIP returned an empty description."

        except Exception as exc:
            description_error = exception_message(exc)

        st.write("3 / 3 — Comparing the description with your caption")

        if description is None:
            nli_error = (
                "Comparison skipped because the visual description "
                "was unavailable."
            )
        else:
            try:
                tokenizer, model = load_nli()

                nli_scores, nli_error = run_nli(
                    premise=description,
                    hypothesis=claim,
                    tokenizer=tokenizer,
                    model=model,
                )

                if not valid_nli_scores(nli_scores):
                    nli_scores = None
                    nli_error = (
                        nli_error
                        or "NLI returned missing or invalid comparison scores."
                    )

            except Exception as exc:
                nli_error = exception_message(exc)

        stage_errors = {
            "Image–caption alignment": clip_error,
            "Visual description": description_error,
            "Description–caption comparison": nli_error,
        }

        has_errors = any(stage_errors.values())

        progress.update(
            label=(
                "Analysis completed with unavailable stages"
                if has_errors
                else "All three model stages completed"
            ),
            state="error" if has_errors else "complete",
            expanded=has_errors,
        )

    result = decide(
        claim=claim,
        clip_score=clip_score,
        caption=description,
        nli_scores=nli_scores,
        clip_error=clip_error,
        caption_error=description_error,
        nli_error=nli_error,
    )

    result.signals["pipeline_status"] = {
        stage: error or "Completed"
        for stage, error in stage_errors.items()
    }

    result.signals["models"] = {
        "alignment": CLIP_ID,
        "description": BLIP_ID,
        "comparison": NLI_ID,
    }

    # A failed stage must not produce a stronger verdict.
    if has_errors:
        result.label = "INSUFFICIENT_EVIDENCE"
        result.explanation = (
            "A required analysis stage could not complete. "
            "This does not establish that your caption is wrong."
        )

        for stage, error in stage_errors.items():
            if error:
                reason = f"{stage}: {error}"
                if reason not in result.abstention_reasons:
                    result.abstention_reasons.append(reason)

    # Current pipeline does not independently verify a specific
    # visual conflict. Caption/NLI conflict alone is not enough.
    if result.label == "CONTRADICTORY":
        result.label = "INSUFFICIENT_EVIDENCE"
        result.explanation = (
            "The models detected a possible disagreement, but a specific "
            "visual conflict has not been verified. Human review is needed."
        )
        result.abstention_reasons.append(
            "Generated-description/NLI disagreement alone does not "
            "establish a visual contradiction."
        )

    if result.label == "CONSISTENT" and nli_scores:
        entailment = nli_scores["entailment"]
        competing = max(
            nli_scores["contradiction"],
            nli_scores["neutral"],
        )

        if competing >= entailment:
            result.label = "INSUFFICIENT_EVIDENCE"
            result.explanation = (
                "The comparison signals do not clearly favor agreement."
            )
            result.abstention_reasons.append(
                "A competing textual signal was at least as strong "
                "as entailment."
            )

    if (
        result.label == "INSUFFICIENT_EVIDENCE"
        and not result.abstention_reasons
    ):
        result.abstention_reasons.append(
            "The available signals did not satisfy the configured "
            "agreement rules."
        )

    return result


# ---------------------------------------------------------
# Clear user-facing answer
# ---------------------------------------------------------

def verdict_text(label):
    return {
        "CONSISTENT": "CORRECT — Caption matches the image",
        "CONTRADICTORY": "WRONG — Caption conflicts with the image",
        "INSUFFICIENT_EVIDENCE": (
            "CANNOT DETERMINE — Not enough evidence"
        ),
    }.get(label, "CANNOT DETERMINE — Not enough evidence")


def render_verdict(result):
    st.header("Final assessment")

    if result.label == "CONSISTENT":
        st.success(f"✅ {verdict_text(result.label)}")
        st.write(
            "The models support agreement between your caption and "
            "the generated image description."
        )

    elif result.label == "CONTRADICTORY":
        st.error(f"❌ {verdict_text(result.label)}")
        st.write(
            "The checked evidence supports a conflict with your caption."
        )

    else:
        st.warning(f"⚠️ {verdict_text(result.label)}")
        st.write(
            "The system cannot responsibly label this caption correct "
            "or wrong from the available evidence."
        )

    st.caption(
        "Model assessment of image–caption agreement only—not independent "
        "verification of factual truth. Models can make mistakes."
    )

    st.markdown("**Caption analyzed**")
    st.write(result.claim)


def render_three_points(result):
    st.subheader("Why the system gave this answer")

    description = result.generated_description
    clip_score = result.signals.get("clip_cosine_similarity")
    nli = result.signals.get("nli_scores")

    st.markdown("**1. What the image-description model reported**")

    if description:
        st.write(description)
        st.caption(
            "Generated by BLIP. This is not a verified observation; "
            "the description can hallucinate or omit details."
        )
    else:
        st.write("No visual description was available.")

    st.markdown("**2. How the description compares with your caption**")

    if nli:
        strongest = max(
            ("entailment", "contradiction", "neutral"),
            key=lambda key: nli[key],
        )

        if strongest == "entailment":
            st.write(
                "The strongest textual signal supports the generated "
                "description entailing your caption."
            )
        elif strongest == "contradiction":
            st.write(
                "The strongest textual signal suggests a possible "
                "conflict. This does not independently prove a "
                "visual contradiction."
            )
        else:
            st.write(
                "The strongest textual signal is neutral: the description "
                "does not clearly establish or contradict the full caption."
            )

        st.caption(
            f"Raw NLI scores — entailment: {nli['entailment']:.4f}; "
            f"contradiction: {nli['contradiction']:.4f}; "
            f"neutral: {nli['neutral']:.4f}. "
            "These are not factual-confidence percentages."
        )
    else:
        st.write("The textual comparison was unavailable.")

    st.markdown("**3. Image–caption alignment and remaining uncertainty**")

    if clip_score is not None:
        st.write(f"CLIP cosine similarity: **{clip_score:.4f}**.")
    else:
        st.write("CLIP alignment was unavailable.")

    st.caption(
        "Similarity is supporting information, not proof. "
        "Dates, identities, precise locations, and event history "
        "may require independent evidence."
    )


def render_diagnostics(result):
    statuses = result.signals.get("pipeline_status", {})

    with st.expander(
        "Model status and errors",
        expanded=any(value != "Completed" for value in statuses.values()),
    ):
        for stage, status in statuses.items():
            if status == "Completed":
                st.success(f"{stage}: completed")
            else:
                st.error(f"{stage}: {status}")

    if result.abstention_reasons:
        with st.expander(
            "Why the system could not decide",
            expanded=result.label == "INSUFFICIENT_EVIDENCE",
        ):
            for reason in dict.fromkeys(result.abstention_reasons):
                st.write(f"• {reason}")

    with st.expander("Detailed model evidence and decision rule"):
        st.write(result.explanation)

        for record in result.evidence:
            st.markdown(f"**{record.get('source', 'Unknown')}**")
            st.caption(record.get("type", ""))
            st.write(record.get("content", ""))

        st.json(result.signals)

    with st.expander("Limitations"):
        for limitation in result.limitations:
            st.write(f"• {limitation}")

        st.write(
            "• This version does not verify specific visual conflicts. "
            "Possible contradiction signals therefore trigger abstention."
        )


# ---------------------------------------------------------
# Human review and report export
# ---------------------------------------------------------

def build_report(bundle, assessment, notes):
    analysis = bundle["result"].to_dict()

    return {
        "application": "CrossCheck AI",
        "metadata": {
            "image_filename": bundle["image_name"],
            "analysis_timestamp_utc": bundle["timestamp"],
            "models": {
                "clip": CLIP_ID,
                "blip": BLIP_ID,
                "nli": NLI_ID,
            },
        },
        "displayed_assessment": verdict_text(analysis["label"]),
        "assessment_scope": (
            "Model-supported image–caption agreement, "
            "not independent factual verification."
        ),
        "implementation_scope": (
            "Agreement and abstention. Specific visual-conflict "
            "verification is not implemented."
        ),
        "analysis": analysis,
        "human_review": {
            "assessment": assessment,
            "notes": notes,
        },
    }


def readable_report(report):
    analysis = report["analysis"]

    lines = [
        "CrossCheck AI — Evidence Before Verdict",
        "=" * 48,
        f"Assessment: {report['displayed_assessment']}",
        f"Technical label: {analysis['label']}",
        f"Image: {report['metadata']['image_filename']}",
        f"Time (UTC): {report['metadata']['analysis_timestamp_utc']}",
        "",
        "Analyzed caption:",
        analysis["claim"],
        "",
        "Model-generated description:",
        analysis["generated_description"] or "Unavailable",
        "",
        "Decision explanation:",
        analysis["explanation"],
        "",
        "Abstention reasons:",
        *analysis["abstention_reasons"],
        "",
        "Raw model signals:",
        json.dumps(analysis["signals"], indent=2, ensure_ascii=False),
        "",
        "Limitations:",
        *analysis["limitations"],
        report["implementation_scope"],
        "",
        "Human review:",
        report["human_review"]["assessment"],
        report["human_review"]["notes"],
        "",
        report["assessment_scope"],
    ]

    return "\n".join(lines)


def render_review_and_downloads(bundle):
    st.divider()
    st.subheader("Human review")

    st.caption(
        "Your review is recorded separately. It does not change "
        "the automated assessment."
    )

    assessment = st.selectbox(
        "Reviewer assessment",
        [
            "Not reviewed",
            "Caption appears to match",
            "Possible conflict — investigate",
            "More evidence needed",
        ],
        key="review_assessment",
    )

    notes = st.text_area(
        "Reviewer notes",
        max_chars=3000,
        key="review_notes",
    )

    report = build_report(bundle, assessment, notes)

    st.subheader("Download analysis")

    left, right = st.columns(2)

    with left:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="crosscheck-evidence.json",
            mime="application/json",
        )

    with right:
        st.download_button(
            "Download readable report",
            data=readable_report(report),
            file_name="crosscheck-evidence.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("🔎 CrossCheck AI")
        st.caption("Evidence Before Verdict")
        st.divider()

        st.write("**MEGHANA H S**")
        st.write("Solo Participant · Team NeuralForge")
        st.write("REVA University · CSE")
        st.write("Challenge 13 — MultimodalFact AI")
        st.write("Track: AI/ML Methodology")

        st.divider()
        st.markdown("**Real pretrained models**")
        st.caption(f"Alignment: {CLIP_ID}")
        st.caption(f"Description: {BLIP_ID}")
        st.caption(f"Comparison: {NLI_ID}")

        st.warning(
            "Experimental decision-support tool. "
            "Thresholds are unvalidated. Human review is recommended."
        )

        st.caption(
            "Current scope: model-supported agreement and abstention. "
            "Specific visual-conflict verification remains incomplete."
        )

        st.caption(
            "Models run on the Python host. First use downloads weights. "
            "This app does not explicitly save uploaded images to disk."
        )


def main():
    render_sidebar()

    st.title("🔎 CrossCheck AI")
    st.subheader("Evidence Before Verdict")

    st.info(
        "Check whether an English caption agrees with an image. "
        "The result concerns visual consistency—not independent "
        "factual verification."
    )

    st.caption(
        "Use natural wording. You do not need to copy a model-generated "
        "description exactly."
    )

    st.button(
        "Reset inputs and analysis",
        on_click=reset_inputs,
    )

    generation = st.session_state.get("input_generation", 0)

    image_column, claim_column = st.columns(2)

    with image_column:
        uploaded = st.file_uploader(
            "Upload one image",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"upload_{generation}",
        )
        st.caption("JPEG, PNG, or WebP · Maximum 10 MB.")

    with claim_column:
        claim_raw = st.text_area(
            "English caption or claim",
            placeholder="Example: A dog is standing outdoors.",
            max_chars=1500,
            height=140,
            key=f"claim_{generation}",
        )

    image = None
    image_data = b""
    image_name = ""

    if uploaded is not None:
        image_name = uploaded.name

        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("Please upload an image no larger than 10 MB.")
        else:
            image_data = uploaded.getvalue()

            try:
                image, validation_error = validate_and_decode_image(
                    image_data
                )

                if validation_error:
                    image = None
                    st.error(f"Image validation failed: {validation_error}")

            except Exception as exc:
                image = None
                st.error(exception_message(exc))

    if image is not None:
        st.image(
            image,
            caption=f"Submitted image: {image_name}",
            width=360,
        )

    current_fingerprint = (
        make_fingerprint(image_data, claim_raw)
        if image is not None and claim_raw.strip()
        else None
    )

    existing = st.session_state.get("analysis_bundle")

    if (
        existing is not None
        and existing["fingerprint"] != current_fingerprint
    ):
        clear_analysis()
        st.info("Inputs changed. Analyze again for an updated result.")

    if st.button("Analyze Consistency", type="primary"):
        clear_analysis()

        if image is None:
            st.warning("Upload a valid image first.")
            return

        clean_claim, claim_error = validate_claim(claim_raw)

        if claim_error:
            st.warning(claim_error)
            return

        try:
            result = analyze_image_claim(image, clean_claim)

            st.session_state["analysis_bundle"] = {
                "result": result,
                "fingerprint": current_fingerprint,
                "image_name": image_name,
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }

        except Exception as exc:
            st.error(
                "Analysis could not complete. "
                "No completed assessment was generated."
            )
            st.code(exception_message(exc), language="text")

    bundle = st.session_state.get("analysis_bundle")

    if (
        bundle is not None
        and bundle["fingerprint"] == current_fingerprint
    ):
        st.divider()

        render_verdict(bundle["result"])
        render_three_points(bundle["result"])
        render_diagnostics(bundle["result"])
        render_review_and_downloads(bundle)

    st.divider()
    st.caption(
        "Research prototype · Real model inference · "
        "No benchmark accuracy claimed · Human review recommended"
    )


if __name__ == "__main__":
    main()