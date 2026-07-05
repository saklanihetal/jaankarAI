# JaankaarAI — Multimodal Multilingual Regional Fake News Detection

JaankaarAI is a multimodal, multilingual fake news verification system built for Indian regional-language content. It goes beyond plain text classification by combining **text analysis, live evidence retrieval, natural language inference, and image forensics** to decide whether a news claim is `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIED`.

Misinformation on social media rarely comes as text alone — it's often paired with unrelated, out-of-context, or manipulated images to make the claim more convincing. JaankaarAI is designed to catch both angles at once: what the text says, and whether the accompanying image actually backs it up.

## Description

Automated fact-checking tools overwhelmingly target English and a handful of high-resource languages, leaving a large gap for the regional-language content that circulates widely on Indian social media. JaankaarAI addresses that gap with an end-to-end verification pipeline: a user submits a claim (text, and optionally an image or audio clip) in any supported language, the system translates and retrieves live, related news evidence, cross-checks the claim against that evidence with NLI models, verifies the image separately for manipulation and text-image consistency, and fuses all of these signals through trained gating classifiers into a final, explainable verdict. The goal is not just a fake/real label but a verdict a user can audit, backed by the actual articles retrieved and the individual model scores that produced it.

## Features

- **Multilingual input** — language detection and translation (NLLB / Google Translate) so regional-language claims are verified in a common pipeline alongside English.
- **Live evidence retrieval** — automatic keyword/query extraction and retrieval of related, real-time news from NewsAPI, EventRegistry, NewsData, and Google RSS.
- **Semantic ranking & NLI** — evidence is ranked with LaBSE embeddings and checked for entailment/contradiction against the claim using BART-Large-MNLI.
- **Multimodal verification** — CLIP-based image-text consistency scoring, a CNN deepfake detector, and AWS Rekognition for entity/celebrity/label recognition.
- **Gated decision fusion** — trained support and contradiction gates (scikit-learn) combine all signals into a single confidence-scored verdict, with a TF-IDF stylometry model as an auxiliary text-only signal.
- **Audio support** — Whisper-based transcription/translation for audio claims, and gTTS for reading results aloud.
- **Streamlit UI** — a two-page app (Landing + Detector) with a full explainability panel showing every underlying signal.

## How the Pipeline Works

A claim submitted to JaankaarAI (text, optionally with an image and/or audio) moves through six stages before a verdict is shown:

### 1. Input Ingestion & Preprocessing
The claim text (or Whisper-transcribed audio) is language-detected. If it isn't already in English, it's translated using NLLB / Google Translate so every downstream model operates on a common language, while the original regional-language text is preserved and shown back to the user for transparency.

### 2. Query Generation & Evidence Retrieval
Keywords and named entities are extracted from the (translated) claim to build one or more search queries. These queries are sent out in parallel to live news sources — NewsAPI, EventRegistry, NewsData, and Google News RSS — to pull back a set of recent, related articles. This step grounds the verdict in current, real-world reporting instead of relying only on a static training set.

### 3. Evidence Ranking & Natural Language Inference
Each retrieved article is embedded with LaBSE and ranked by semantic similarity (relevance) to the claim. The most relevant articles are then passed, together with the claim, into a BART-Large-MNLI natural language inference model, which scores each claim-evidence pair as entailment (supports), contradiction, or neutral. This produces a distribution of relevance, entailment, and contradiction scores across all retrieved evidence rather than a single article's opinion.

### 4. Multimodal Verification
If an image is attached, CLIP measures how consistent the image is with the claim text (catching cases where a real photo is reused with an unrelated caption). A CNN-based deepfake detector separately scores the likelihood the image itself has been manipulated. AWS Rekognition identifies recognizable faces, celebrities, and objects in the image, which helps flag identity mismatches (for example, a claim about one public figure illustrated with a photo of someone else).

### 5. Decision Fusion & Gated Classification
All of the signals gathered so far — relevance/entailment/contradiction statistics from stage 3, CLIP and deepfake scores from stage 4, plus TF-IDF stylometry and lexicon-based writing-style signals — are combined into a feature vector. Two trained scikit-learn models, the support gate and the contradiction gate, each independently decide whether the aggregated evidence is strong enough to call the claim supported or contradicted, using thresholds tuned during training (see the evaluation table below).

### 6. Final Verdict & UI
The gate outputs, together with the raw evidence and signal telemetry, are passed to a backend reasoning engine (Gemini) that applies the final decision rules and produces one of three verdicts — SUPPORTED, CONTRADICTED, or UNVERIFIED — along with a short explanation. Everything is rendered in the Streamlit UI: the verdict itself, the evidence snippets it was based on, and a full "Signal Telemetry" panel exposing every underlying score (TF-IDF, NLI entailment/contradiction, CLIP relevance, deepfake probability, recognized entities) so the result isn't a black box.

### Outcome
Rather than returning a single fake/real label from text alone, JaankaarAI produces a verdict that is traceable back to actual retrieved evidence and, where relevant, cross-checked against the accompanying image — which is particularly useful for regional-language misinformation that often pairs a plausible-sounding local claim with a repurposed photo or video still. The explainability panel means a user (or moderator) reviewing a verdict can see exactly which articles were used, how similar or contradictory they were, and whether the image itself raised any red flags, rather than having to trust the verdict blindly.

## Repository Structure

```
jaankarAI/
├── app/
│   ├── app.py            # Current production app (final decision engine + UI)
│   ├── app1.py – app4.py  # Iterative development snapshots (kept for reference)
│   └── cache_key.py       # Retrieval cache key helper
├── archive/
│   ├── old_main/          # Earlier standalone pipeline versions (main_v1–v5)
│   ├── old_ui/             # Earlier Streamlit UI iterations
│   └── tests/              # Ad-hoc test/debug scripts
├── data/
│   ├── raw/                 # Training feature CSVs (v1–v3, RSS variant)
│   ├── cache/                # Retrieval cache
│   └── lexicon_out/           # Lexicon resources
├── models/
│   ├── weights/                # Trained model artifacts (.joblib)
│   └── config/                  # Model metadata: features, thresholds, metrics
├── plots/                        # Evaluation plots (ROC, PR, confusion, prob histograms)
├── scripts/
│   ├── preprocessing/             # Retrieval cache building, entity extraction
│   └── training/                    # Feature building and model training scripts
└── README.md
```

## Models & Evaluation

Three trained classifiers drive the decision layer (see `models/config/*_meta.json` and `plots/metrics_summary.json` for full details):

| Model | Purpose | Accuracy | F1 | AUC | Threshold |
|---|---|---|---|---|---|
| **Contradiction Gate** | Detects when evidence contradicts the claim | 0.991 | 0.944 | 0.998 | 0.70 |
| **Support Gate** | Detects when evidence is strong enough to assert `SUPPORTED` | 0.963 | 0.00* | 0.716 | 0.35 |
| **TF-IDF Style Model** | Character n-gram stylometry baseline | 0.419 | 0.165 | 0.321 | — |

\* The support gate is trained on a highly imbalanced auto-labeled set (55 positives / 1445 negatives); precision/recall at the default threshold are near-zero, so it is best treated as an auxiliary signal rather than a standalone classifier. Corresponding ROC, PR, confusion matrix, and probability histogram plots for each model are in `plots/`.

## Getting Started

### Prerequisites

- Python 3.9+
- API keys for the services you want to enable (all are optional and fail gracefully if missing):
  - `NEWS_API_KEY`, `NEWSDATA_KEY`, `EVENTREGISTRY_KEY` — evidence retrieval
  - `HF_TOKEN` — Hugging Face models (CLIP, BART-MNLI, LaBSE, NLLB)
  - `AWS_KEY`, `AWS_SECRET`, `AWS_REGION` — AWS Rekognition
  - `GEMINI_API_KEY` — backend reasoning/decision engine

### Installation

```bash
git clone https://github.com/<your-username>/jaankarAI.git
cd jaankarAI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** This repo doesn't yet ship a `requirements.txt`. Core dependencies used across `app/app.py` include: `streamlit`, `torch`, `transformers`, `sentence-transformers` (LaBSE), `scikit-learn`, `joblib`, `numpy`, `pandas`, `Pillow`, `requests`, `feedparser`, `deep-translator`, `boto3`, `google-generativeai`, `gTTS`, and `openai-whisper`/`tensorflow` for audio and deepfake components. Pin versions as you finalize your environment and add a `requirements.txt` to the repo root.

### Configuration

Add your API keys either to Streamlit secrets (`.streamlit/secrets.toml`) or as environment variables:

```toml
NEWS_API_KEY = "..."
NEWSDATA_KEY = "..."
EVENTREGISTRY_KEY = "..."
HF_TOKEN = "..."
AWS_KEY = "..."
AWS_SECRET = "..."
AWS_REGION = "..."
GEMINI_API_KEY = "..."
```

### Running the App

```bash
streamlit run app/app.py
```

Navigate between the **Home** page and the **Run Detector** page from the sidebar. On the detector page, submit a text claim (optionally with an image or audio) to get a verdict along with a full signal telemetry breakdown (TF-IDF scores, NLI entailment/contradiction, CLIP relevance, deepfake probability, and recognized entities).

## Training Your Own Models

Feature-building and training scripts are in `scripts/`:

- `scripts/preprocessing/` — builds retrieval caches (`build_retrieval_cache_v2.py`, `build_retrieval_cache_rss.py`) and offline entity lexicons (`offline_build_india_entities.py`).
- `scripts/training/` — builds feature tables (`build_training_features*.py`) and trains each model (`train_support_gate.py`, `train_contradiction_gate_v1.py`, `train_tfidf_style_model.py`, `train_fake_real_mlp_v3.py`).

Trained artifacts are saved to `models/weights/`, and metadata (feature lists, thresholds, label stats) to `models/config/`.

## Roadmap / Notes

- `app/app1.py`–`app4.py` are retained development snapshots showing the system's evolution (from a text-only baseline to multilingual query generation with Whisper transcription and retrieval telemetry). `app/app.py` is the current entry point.
- `archive/` holds older pipeline (`main_v1`–`v5`) and UI iterations for historical reference — not required to run the current app.
- The support gate would benefit from a more balanced training set; contributions to relabeling or rebalancing are welcome.

## Acknowledgements

Built on top of open models and libraries including CLIP, BART-Large-MNLI, LaBSE, NLLB, Whisper, AWS Rekognition, and Google Gemini.
