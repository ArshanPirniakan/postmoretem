# PostMortem

**Personalized content intelligence and a counterfactual performance simulator.**

PostMortem learns from *your own* historical social content performance and predicts how a new draft is likely to do before you publish it. Each prediction comes with an uncertainty estimate, and you can run "what if I changed X" scenarios against the same model.

It does **not** claim to predict universal virality. It only knows what it has seen in the history you upload.

> "Based on this account's own history, how is this draft likely to perform, how uncertain is that prediction, and how would the prediction change under different controllable scenarios?"

---

## What's in this repo

The whole application is a single Python module by design.

| File | Purpose |
| --- | --- |
| `postmortem.py` | Everything: config, DB access, CSV ingestion, feature engineering, PyTorch model, training, model registry, inference, simulation, FastAPI routes, and a minimal built-in web UI |
| `schema.sql` | SQLite schema, loaded at startup (idempotent). Must sit next to `postmortem.py` |
| `requirements.txt` / `pyproject.toml` | Dependencies |
| `Dockerfile` | Container build |
| `.env.example` | Example environment variables |

## How it works

```
CSV upload -> validation/quarantine -> normalization -> SQLite
                                                          |
                                                  FeaturePipeline
                                       (text hashing embedding + metadata)
                                                          |
                          +-------------------------------+---------------+
                          v                                               v
               HistGradientBoosting baseline              PyTorch fusion model
                                                    (text + metadata -> 3 heads)
                          +-------------------------------+---------------+
                                                          v
                                     Checksum-verified model registry
                                                          v
                                        FastAPI: predict / simulate
                                                          v
                                     Built-in web UI at http://localhost:8000
```

- **Language:** Python 3.11+
- **Model:** PyTorch multi-task fusion model with one regression head (engagement rate) and two classification heads (above median, top 20%)
- **Baseline:** scikit-learn `HistGradientBoostingRegressor` on the metadata features, reported next to the fusion model so you can see whether the neural model actually helps
- **Text features:** a deterministic hashing-based embedding. It needs no model download and works fully offline (see [Limitations](#known-limitations))
- **Uncertainty:** Monte Carlo Dropout (mean and standard deviation per prediction)
- **Database:** SQLite, a local file
- **API/UI:** FastAPI, plus a single-page HTML UI served at `/`
- **Cost:** no paid API, no GPU required, no managed database

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python postmortem.py
```

Then open <http://localhost:8000>:

1. **Create a project**
2. **Upload a CSV** (see the format below)
3. **Train**: copy the dataset ID into the form, start training, and use "Check job status" until it reads `completed`
4. **Predict** a draft, then run a **counterfactual** (for example, change `post_type`)

Interactive API docs are at <http://localhost:8000/docs>.

### Docker

<!-- VERIFY: confirm the Dockerfile's exposed port and data path match these commands -->

```bash
docker build -t postmortem .
docker run -p 8000:8000 -v postmortem-data:/app/data -v postmortem-artifacts:/app/artifacts postmortem
```

The volumes keep the SQLite database and trained models across restarts.

## CSV format

| Column | Required | Notes |
| --- | --- | --- |
| `caption` | Yes | Free text |
| `platform` | Yes | Lowercased automatically |
| `post_type` | Yes | For example `image`, `video`, `carousel`, `reel` |
| `posted_at` | Optional | Any parseable timestamp. Rows without one are used for training only |
| `hashtags` | Optional | Raw string. `#tags` in the caption are counted too |
| `impressions` | Needed for a target | Engagement rate can't be computed without it |
| `likes`, `comments`, `shares`, `saves` | Optional | Non-negative numbers |

Common aliases are mapped automatically (`text`/`content` -> `caption`, `views`/`reach` -> `impressions`, `date`/`timestamp` -> `posted_at`, and a few more).

**Engagement rate** is `(likes + 3*comments + 5*shares + 4*saves) / max(impressions, 1)`.

Rows that fail validation (missing required fields, non-numeric or negative counts, unparseable timestamps) are **quarantined with a reason** and reported in the upload response, not silently dropped. Exact duplicates are removed. You need at least 5 valid posts to train, and in practice far more for the results to mean anything.

Minimal example:

```csv
caption,platform,post_type,posted_at,impressions,likes,comments,shares,saves
"Behind the scenes of today's shoot #bts",instagram,reel,2026-01-05T18:30:00Z,5200,310,22,14,40
"New blog post is live, link in bio",instagram,image,2026-01-08T09:10:00Z,1800,45,3,1,6
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health |
| `POST /projects` | Create a project |
| `GET /projects` | List projects |
| `POST /projects/{id}/datasets` | Upload a CSV; returns a validation report |
| `POST /projects/{id}/train` | Start a background training job |
| `GET /training-jobs/{id}` | Poll job status (`queued`/`running`/`completed`/`failed`) |
| `GET /projects/{id}/models` | List trained model runs and their metrics |
| `POST /projects/{id}/predict` | Score a single draft |
| `POST /projects/{id}/simulate` | Compare a baseline draft against scenarios |
| `GET /projects/{id}/analytics` | Historical summary stats |

Simulation scenarios may only override `caption`, `platform`, `post_type`, `posted_at` and `hashtags`.

## Configuration

All environment variables are optional.

| Variable | Default |
| --- | --- |
| `POSTMORTEM_DATA_DIR` | `./data` |
| `POSTMORTEM_ARTIFACTS_DIR` | `./artifacts` |
| `POSTMORTEM_DB_PATH` | `./data/postmortem.db` |
| `POSTMORTEM_MAX_UPLOAD_BYTES` | 25 MB |
| `POSTMORTEM_MAX_UPLOAD_ROWS` | 200000 |
| `POSTMORTEM_SEED` | 42 |
| `POSTMORTEM_MAX_EPOCHS` | 60 |
| `POSTMORTEM_MAX_TRAIN_SECONDS` | 600 |
| `POSTMORTEM_MAX_CONCURRENT_JOBS` | 1 |
| `POSTMORTEM_MC_PASSES` | 30 |
| `POSTMORTEM_TEXT_DIM` | 256 |
| `PORT` | 8000 |

## Engineering notes

- **Leakage prevention:** the split is chronological. Classification thresholds (median, top 20%), scalers, encoders and the per-platform/post-type averages are all fit on the training split only, then applied unchanged to validation, test and inference.
- **One feature pipeline:** the same fitted `FeaturePipeline` is saved with the model and reused for inference and simulation. There is no second implementation of the feature logic.
- **Artifact integrity:** each model directory is SHA-256 checksummed when saved. Loading recomputes the checksum and refuses to load on any mismatch with the stored value.
- **Masked losses:** rows with unknown targets are excluded from the relevant loss term instead of being given fake labels.
- **Uncertainty as a warning:** the standard deviation from MC Dropout is shown as a plain-language "low confidence" note above a threshold. It is not a calibrated interval.
- **Non-causal counterfactuals:** every simulation response includes a disclaimer that the output is model-based and correlational, not proof of a real-world effect.
- **Background jobs:** training runs in an in-process thread guarded by a semaphore. There is no Celery, Redis or Kubernetes.

## Known limitations

- **No authentication.** Anyone who can reach the server can create projects, upload data and train models. Run it locally or behind your own access control. By default it binds to `0.0.0.0`, so change the host in the last line of `postmortem.py` to `127.0.0.1` if you don't want it reachable from your network.
- **Pickle is used for the fitted feature pipeline and baseline model** (the fusion model weights are saved as a PyTorch `state_dict`). The checksum detects corruption or edits to a model directory, but it is not a signature. Only point `POSTMORTEM_ARTIFACTS_DIR` at a directory you trust.
- **The text encoder captures which words appear, not what they mean.** It averages per-token hashed vectors, so it can't generalize across synonyms or phrasing.
- **Small or undated datasets give optimistic metrics.** Rows without `posted_at` go to training only. If the validation or test split ends up empty, evaluation falls back to the training data. Always read the fusion metrics next to the baseline metrics in each run's `metrics_json`.
- **Per-platform and per-post-type average engagement** are computed from the training split and also used as features for those same training rows, a simplification rather than a strict point-in-time feature.
- **Only the regression and "above median" classifier are evaluated.** The top-20% head is trained but not reported in the metrics.
- **SQLite with a single shared connection** is fine for local or demo use, not for concurrent multi-user production.

## Running tests

There is no automated test suite yet.

## License

<!-- Add a license (for example MIT) and a LICENSE file. -->
