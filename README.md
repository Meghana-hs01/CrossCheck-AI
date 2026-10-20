# CrossCheck AI
## Evidence Before Verdict

**Developer:** MEGHANA H S  
**Team:** NeuralForge · Solo Participant  
**Institution:** REVA University, CSE  
**Event:** AI Innovation Hackathon 2026  
**Challenge:** 13 — MultimodalFact AI  
**Track:** AI/ML Methodology  

## Overview

CrossCheck AI is an experimental image–English-claim consistency
application. It combines real pretrained model inference with
transparent decision rules and human-review controls.

It evaluates cross-modal consistency. It does not independently
establish factual truth, verify sources, or detect image manipulation.

## Actual implemented pipeline

1. Validate and decode the uploaded image.
2. Compute CLIP image–claim cosine similarity.
3. Generate a BLIP visual description.
4. Compare that description with the claim using an NLI model.
5. Apply a transparent, heuristic agreement/abstention policy.
6. Display model signals, evidence, limitations, and review controls.
7. Export JSON and readable evidence reports.

Model-generated descriptions can hallucinate or omit details.
NLI compares text; it does not directly inspect the image.

## Current verdict scope

- **CONSISTENT:** model-supported agreement with the generated
  description under the configured heuristic rules.
- **INSUFFICIENT_EVIDENCE:** unavailable stages, invalid or weak signals,
  selected external-context flags, ambiguity, or unsupported conflict.
- **CONTRADICTORY:** reserved in the result schema. Specific grounded
  visual-conflict verification is not implemented, so the current
  application does not issue this verdict.

This is not yet a complete three-way visual contradiction detector.

## Architecture

```text
app.py
requirements.txt
README.md
.gitignore
src/
    __init__.py
    validation.py
    analysis.py
    decision.py
    models.py
tests/
    test_validation.py
    test_analysis.py
    test_decision.py
```

`app.py` contains the Streamlit interface and cached model loaders.
`analysis.py` wraps inference.
`decision.py` contains the pure decision policy.
`validation.py` validates uploads and claims.
`models.py` contains additional loaders for non-Streamlit callers.

## Models and acknowledgements

- Alignment: `openai/clip-vit-base-patch32`
  https://huggingface.co/openai/clip-vit-base-patch32
- Description: `Salesforce/blip-image-captioning-base`
  https://huggingface.co/Salesforce/blip-image-captioning-base
- Text comparison: `typeform/distilbert-base-uncased-mnli`
  https://huggingface.co/typeform/distilbert-base-uncased-mnli

Models and dependencies retain their respective licenses.
Consult the linked model repositories for model cards and license terms.
No project-wide license grant is implied by these acknowledgements.

Weights are downloaded on first use and cached separately.
They are not included in this repository.

## Local setup — Windows CMD

Use Python 3.10.

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.fileWatcherType none
```

Open http://localhost:8501.

Internet access is required for initial model downloads.
All three models require memory and storage; suitability for a hosting
platform must be checked separately.

## Decision policy

All thresholds are unvalidated prototype heuristics.

- CLIP similarity is supporting information, not a factual-confidence
  score or a hard 0.26 agreement gate.
- Agreement requires valid CLIP, BLIP, and NLI outputs.
- Non-positive CLIP alignment triggers abstention.
- NLI entailment must be at least 0.80.
- Entailment must exceed the strongest competing NLI signal by at
  least 0.20.
- Model-stage errors trigger abstention.
- Selected external-context patterns trigger abstention.
- Generated-caption/NLI contradiction alone does not establish a
  supported visual contradiction.

External-context detection is limited and may miss unsupported claims.
Do not interpret agreement as validation of every part of a claim.

## Automated tests

Run:

```bat
.venv\Scripts\python.exe -m pytest -q --tb=short
```

Latest user-observed local result on 20 October 2026:

```text
63 passed in 2.47s
```

The decision tests were revised to match the current policy:
the former hard CLIP gate was removed, and unsupported textual
contradictions now require abstention.

Tests cover validation, mocked inference, result structure, missing
signals, error handling, and decision-policy behavior.

Passing unit tests does not establish real-image accuracy.
No benchmark accuracy, precision, recall, or F1 is claimed.

## Observed real-model smoke test

For one dog-on-grass image and the claim:

> A dog is standing on grass.

The user observed:

- CLIP cosine similarity: 0.2812
- BLIP description: "a golden retrieve dog standing in a field"
- NLI entailment: 0.9919
- NLI contradiction: 0.0054
- NLI neutral: 0.0027
- Assessment: CONSISTENT

This is one observed example, not a benchmark.
The latest decision-policy changes require further real-image testing.

## Suggested manual checks

1. Submit missing image/text and inspect validation.
2. Analyze a clear image with a matching claim.
3. Test several natural paraphrases using the same image.
4. Test unsupported dates or event history.
5. Edit inputs and confirm old results disappear.
6. Reset inputs and analysis.
7. Add human-review notes and inspect both report formats.
8. Inspect model-stage errors if a model cannot load.

Record actual observations rather than assuming expected results.

## Limitations and responsible use

- Generated descriptions can hallucinate or omit content.
- NLI may misinterpret paraphrases, negation, or fine-grained details.
- CLIP can be unreliable for counts, attributes, and precise relations.
- The decision thresholds are not calibrated on a labelled dataset.
- False positives and false negatives remain possible.
- No OCR, source retrieval, geolocation, or image-forensics capability
  is implemented.
- Specific visual-conflict verification remains incomplete.
- Human review is required; no consequential action is automated.

## Privacy

The application code processes uploaded images in memory and does not
explicitly save them to disk.

Inference runs within the Python hosting process, not through a paid
remote inference API. Initial model downloads require network access.

For a public deployment, uploaded inputs are sent to the hosting server
for processing. Host-specific logging, retention, and privacy settings
must be reviewed separately.

## Deployment status

Local model inference has been observed.
Public deployment has not yet been verified.

A Python application host is required; GitHub Pages cannot run this
backend. Deployment must provide sufficient memory, storage, network
access for model downloads, and CPU resources.

Only publish a live-demo URL after verifying an actual inference request.

## Repository exclusions

Do not commit:

- `.venv/` or other virtual environments
- Model weights and caches
- Secrets or `.env` files
- Private uploads
- Local backups or generated reports