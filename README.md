# PostMortem

**Personalized Content Intelligence & Counterfactual Performance Simulator**

PostMortem learns from *your own* historical content performance and predicts
how a new draft is likely to do, before you publish it — with an honest
uncertainty estimate, and a "what if I changed X" counterfactual simulator.
It does not claim to predict universal virality. It only knows what it has
seen in your account's own history.

> "Based on this account's own history, how is this draft likely to perform,
> how uncertain is that prediction, and how would the prediction change under
> different controllable scenarios?"

---

## Architecture at a glance

```
CSV upload → validation → normalization → SQL (SQLite)
                                              │
                          ┌───────────────────┼────────────────────┐
                          ▼                                        ▼
                 Feature pipeline                          Analytics API
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                     ▼
  Baseline models                    PyTorch fusion model
  (dummy / HGB / TF-IDF)             (text + metadata → 3 heads)
        └─────────────────┬─────────────────┘
                          ▼
                 Model registry (checksum-verified)
                          ▼
                   Inference API (FastAPI)
                          │
              ┌────────────┴────────────┐
              ▼                          ▼
        Prediction UI          Counterfactual Lab
             (Streamlit, talks to the API over HTTP)
```

- **Language:** Python 3.11+
- **Deep learning:** PyTorch multi-task fusion model (regression + 2 classification heads)
- **Text embeddings:** local sentence-transformers encoder, with a deterministic
  dependency-free hashing fallback if the model can't be downloaded (fully offline-capable)
- **Baselines:** dummy, HistGradientBoosting, TF-IDF + logistic regression
- **Database:** SQLite (zero-cost, file-based; swap `POSTMORTEM_DATABASE_URL` for Postgres if needed)
- **API:** FastAPI + Pydantic
- **UI:** Streamlit
- **Uncertainty:** Monte Carlo Dropout
- **Cost:** no paid API, no required GPU, no required managed database

## Zero-cost guarantee

Nothing in this system requires a paid API key, a GPU, or a managed cloud
service. Training and inference run on CPU. The database is a local SQLite
file. Any free-tier cloud host is treated as an optional distribution layer
on top of a system that works fully locally and in Docker.

---

## Quickstart (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate a synthetic demo dataset (reproducible, no real/scraped data)
python scripts/generate_sample_data.py --rows 300 --out tests/fixtures/sample_posts.csv

# Terminal 1: API
uvicorn app.main:app --reload --port 8000

# Terminal 2: UI
POSTMORTEM_API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501, create a project, upload
`tests/fixtures/sample_posts.csv` on the **Upload** page, train a model on
the **Training** page, and try **Predict** / **Counterfactual Lab**.

## Quickstart (Docker)

```bash
docker compose up --build
```

- API: http://localhost:8000/health
- UI: http://localhost:8501

Both services share a persistent Docker volume for the SQLite database and
model artifacts, so data survives container restarts.

## Running tests

```bash
pip install -r requirements.txt
pytest -q
```

28 tests cover ingestion validation/quarantine, chronological-split leakage
prevention, feature pipeline determinism, fusion model shapes/losses/MC
Dropout, artifact checksum verification (including a tamper-detection test),
and a full integration test that exercises upload → normalize → train →
registry → predict → simulate end to end through the real FastAPI app.

---

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /projects` | Create an isolated project |
| `POST /projects/{id}/datasets` | Upload a CSV, get a validation report |
| `GET /datasets/{id}/validation` | Re-fetch a dataset's validation report |
| `POST /projects/{id}/train` | Start a background training job |
| `GET /training-jobs/{id}` | Poll job status (`queued/running/completed/failed`) |
| `GET /projects/{id}/models` | List trained model versions |
| `POST /projects/{id}/predict` | Score a single draft |
| `POST /projects/{id}/simulate` | Counterfactual scenario comparison |
| `GET /projects/{id}/analytics` | Historical summary stats |
| `GET /health` | Service health |

Interactive docs at `http://localhost:8000/docs` once the API is running.

## Canonical CSV schema

| Field | Required | Notes |
|---|---|---|
| `caption` | Yes | Free text |
| `platform` | Yes | Normalized to lowercase |
| `post_type` | Yes | e.g. image, video, carousel, reel |
| `posted_at` | Optional | Any parseable timestamp |
| `hashtags` | Optional | Raw string, parsed downstream |
| `impressions` | Preferred | Needed to compute `engagement_rate` |
| `likes` / `comments` / `shares` / `saves` | Optional | Non-negative |

Common aliases (`text`→`caption`, `views`→`impressions`, etc.) are mapped
automatically. Rows that fail validation are **quarantined with a reason**,
never silently dropped — see the validation report returned by the upload
endpoint.

---

## Key engineering guarantees

- **Leakage prevention:** chronological train/val/test split; classification
  thresholds (median, top-20%) and all scalers/encoders are fit on the
  training split only, then applied unchanged everywhere else.
- **Consistent feature pipeline:** the exact same `FeaturePipeline` object
  used at training time is persisted and reused at inference/simulation
  time — there is no second, drifted implementation in the API layer.
- **Artifact integrity:** every model artifact directory is SHA-256
  checksummed at save time; loading recomputes and verifies the checksum
  and refuses to load on any mismatch (see `tests/integration/test_registry.py`
  for a tamper-detection test).
- **No untrusted deserialization:** only artifacts created and registered by
  this application are ever loaded. Model weights are saved as PyTorch
  `state_dict`s, not arbitrary pickled objects.
- **Uncertainty as a warning signal:** Monte Carlo Dropout gives a mean and
  std per prediction; the UI surfaces a plain-language "low confidence"
  note above a configurable std threshold, not a false precision number.
- **Non-causal counterfactuals:** every simulation response carries an
  explicit disclaimer that changes are model-based correlational output,
  not causal proof.

## Known limitations

- The historical-stats features (average engagement by platform/post_type)
  are frozen from the training split at fit time and then applied
  identically at inference — this is a coarser simplification than a
  strictly point-in-time expanding-window feature, chosen to keep the
  pipeline simple and leakage-safe. If per-post-in-time historical
  features become important, replace `FeaturePipeline._numeric_block`'s
  lookup with a proper expanding aggregate.
- The hashing-fallback text encoder (used when `sentence-transformers` /
  its model weights aren't available) is meaningfully weaker than a real
  pretrained sentence encoder — it exists to keep the system fully
  functional offline, not as a replacement for it.
- Metrics on small synthetic datasets (e.g. the bundled 300-row generator)
  will look noisy; this is expected and documented, not a bug. Always
  compare fusion-model metrics against the reported baselines rather than
  reading them in isolation (see `metrics.json` in each model run).
- SQLite is fine for a single local/demo deployment; for concurrent
  multi-user production use, point `POSTMORTEM_DATABASE_URL` at Postgres.

## What this project deliberately does not do

- Does not call any paid LLM API for core functionality.
- Does not require a GPU.
- Does not promise universal virality prediction — only personalized,
  account-specific pattern modeling.
- Does not introduce Kubernetes/Celery/Redis — the background job runner
  is an in-process thread + semaphore, sufficient for a single-node,
  zero-cost deployment.

---

## Project layout

```
postmortem/
├── app/            FastAPI service: routes, schemas, DB models, services
├── ml/              feature engineering, models, training, evaluation, registry
├── ui/              Streamlit product interface
├── tests/           unit + integration tests, fixtures
├── scripts/         synthetic sample-data generator
├── docker/          Dockerfile.api, Dockerfile.ui
├── docker-compose.yml
├── requirements.txt / pyproject.toml
└── .env.example
```
