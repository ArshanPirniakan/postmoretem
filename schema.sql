-- schema.sql
-- PostMortem: Personalized Content Intelligence & Counterfactual Performance Simulator
-- SQLite schema. Loaded automatically by postmortem.py on startup (idempotent: IF NOT EXISTS).
--
-- Design notes:
--   * Model binaries are NEVER stored here -- only filesystem paths + SHA-256
--     checksums, verified before load (see ModelRegistry in postmortem.py).
--   * All IDs are server-generated UUID4 strings, never client-supplied.
--   * Every upload row is either accepted into `posts` or recorded in
--     `dataset_issues` with a reason -- never silently dropped.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS datasets (
    id                     TEXT PRIMARY KEY,
    project_id             TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    original_filename      TEXT NOT NULL,
    stored_path            TEXT NOT NULL,
    checksum_sha256        TEXT NOT NULL,
    row_count              INTEGER NOT NULL DEFAULT 0,
    quarantined_row_count  INTEGER NOT NULL DEFAULT 0,
    status                 TEXT NOT NULL DEFAULT 'uploaded',   -- uploaded|validated|failed
    created_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_issues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_index   INTEGER NOT NULL,
    reason      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id               TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id       TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    caption          TEXT NOT NULL DEFAULT '',
    platform         TEXT NOT NULL,
    post_type        TEXT NOT NULL,
    posted_at        TEXT,                 -- ISO-8601 UTC, nullable
    hashtags         TEXT NOT NULL DEFAULT '',
    impressions      REAL,
    likes            REAL,
    comments         REAL,
    shares           REAL,
    saves            REAL,
    engagement_rate  REAL,                 -- weighted(likes,comments,shares,saves) / max(impressions, 1)
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_posts_project ON posts(project_id);
CREATE INDEX IF NOT EXISTS idx_posts_dataset ON posts(dataset_id);
CREATE INDEX IF NOT EXISTS idx_posts_posted_at ON posts(posted_at);

CREATE TABLE IF NOT EXISTS training_jobs (
    id             TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id     TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    status         TEXT NOT NULL DEFAULT 'queued',   -- queued|running|completed|failed
    model_run_id   TEXT,
    error_message  TEXT,
    started_at     TEXT,
    finished_at    TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    id                       TEXT PRIMARY KEY,
    project_id               TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id               TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    model_kind               TEXT NOT NULL DEFAULT 'fusion',
    config_json              TEXT NOT NULL DEFAULT '{}',
    metrics_json             TEXT NOT NULL DEFAULT '{}',
    feature_schema_version   TEXT NOT NULL DEFAULT 'v1',
    random_seed              INTEGER NOT NULL DEFAULT 42,
    median_threshold         REAL NOT NULL DEFAULT 0,
    top20_threshold          REAL NOT NULL DEFAULT 0,
    status                   TEXT NOT NULL DEFAULT 'completed',
    created_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model binaries live on disk; this row is only a checksum-verified pointer.
CREATE TABLE IF NOT EXISTS model_artifacts (
    id                   TEXT PRIMARY KEY,
    model_run_id         TEXT NOT NULL UNIQUE REFERENCES model_runs(id) ON DELETE CASCADE,
    artifact_dir         TEXT NOT NULL,
    model_state_path     TEXT NOT NULL,
    preprocessing_path   TEXT NOT NULL,
    checksum_sha256      TEXT NOT NULL,
    is_active            INTEGER NOT NULL DEFAULT 0,   -- 0/1, one active artifact per project
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_active ON model_artifacts(is_active);

CREATE TABLE IF NOT EXISTS predictions (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_run_id      TEXT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    scenario_label    TEXT NOT NULL DEFAULT 'baseline',
    input_json        TEXT NOT NULL,
    regression_pred   REAL NOT NULL,
    median_prob       REAL NOT NULL,
    top20_prob        REAL NOT NULL,
    uncertainty_std   REAL NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
