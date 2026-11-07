# CrossCheck AI

### Evidence Before Verdict

CrossCheck AI is an experimental Streamlit app that compares an image with
an English caption or claim. It combines three pretrained models with
transparent decision rules, human-review controls, and downloadable reports.

> **Important:** This prototype evaluates image–text consistency. It does not
> independently establish factual truth, verify sources, or prove that an
> image is authentic.

## What it does

1. Validates and decodes an uploaded image.
2. Uses CLIP to measure image–claim embedding similarity.
3. Uses BLIP to generate a description of the image.
4. Uses an NLI model to compare that description with the claim.
5. Applies a heuristic policy to report supported consistency or abstain.
6. Shows model signals, evidence, limitations, and review controls.
7. Exports the analysis and human-review notes as JSON or text.

## Try it

Upload an image and enter a plain-English claim, for example:

- “A dog is standing on grass.”
- “A red car is parked beside a building.”
- “Two people are riding bicycles.”

Click **Analyze Consistency**. The first analysis downloads the model weights,
so an internet connection is required. Depending on your connection and
computer, the first run may take several minutes.

## Run locally on Windows

Use Python 3.10 and PowerShell from the project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Keep the
terminal open while using the app; press **Ctrl+C** there to stop it.

The CPU installation works without a CUDA-compatible GPU. The three models
need several gigabytes of disk space and may require substantial memory while
running. Their weights are downloaded and cached locally on first use; they
are not included in this repository.

## Run tests

The automated tests use mocked model outputs for inference-stage tests and do
not download model weights:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
```

The tests cover image and claim validation, inference wrappers, result
structure, errors, and decision-policy behavior. Passing tests does not
demonstrate model accuracy on real-world images.

## Models

| Stage | Model |
|---|---|
| Image–claim alignment | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| Image description | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Description–claim comparison | [typeform/distilbert-base-uncased-mnli](https://huggingface.co/typeform/distilbert-base-uncased-mnli) |

Review the linked model cards and licenses before redistributing or deploying
the models. Project dependencies and model weights retain their own licenses;
this repository does not grant a project-wide license.

## How to interpret results

- **CONSISTENT** means the model signals meet the prototype's rules for
  agreement with the generated description. It is not proof that the full
  claim is true.
- **INSUFFICIENT_EVIDENCE** means the app abstains because evidence is missing,
  weak, ambiguous, or requires context not available to the models.
- **CONTRADICTORY** is reserved in the result schema. The app does not issue
  this verdict because specific, grounded visual-conflict verification is not
  implemented.

CLIP similarity is not a probability or calibrated confidence score. NLI
compares text and does not directly inspect the image. Generated descriptions
may omit details or hallucinate. All decision thresholds are experimental,
unvalidated heuristics; human review is recommended.

## Project structure

```text
app.py                  Streamlit interface and cached model loaders
requirements.txt        Runtime and test dependencies
src/
  analysis.py            CLIP, BLIP, and NLI inference wrappers
  decision.py            Evidence and abstention policy
  validation.py          Image and claim validation
tests/
  test_analysis.py
  test_decision.py
  test_validation.py
```

## Privacy and deployment

The app processes uploaded images in memory and does not explicitly save them
to disk. Inference runs in the Python process hosting the app. A public
deployment receives uploaded images on its server, so review that host's
logging, retention, privacy, and resource policies before sharing it.

GitHub Pages cannot run this Python backend. A live deployment needs a Python
host with enough memory, storage, and network access to download the models.
No public deployment is provided or claimed by this repository.

## Push to GitHub

Install [Git for Windows](https://git-scm.com/download/win), create an empty
repository on GitHub, then run these commands from the project folder. Replace
the URL with your repository's HTTPS URL:

```powershell
git init
git add .
git commit -m "Prepare CrossCheck AI project"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
git push -u origin main
```

The `.gitignore` excludes virtual environments, caches, model weights,
secrets, uploads, and local backups. Do not force-add those files or commit
credentials.

---

**Project:** Challenge 13 — MultimodalFact AI · AI/ML Methodology  
**Team:** NeuralForge · Solo Participant  
**Institution:** REVA University · Computer Science and Engineering
