#!/usr/bin/env python3
"""
PostMortem -- Personalized Content Intelligence & Counterfactual Performance Simulator
=========================================================================================
Single-file application. Everything (config, DB access, ingestion, feature
engineering, PyTorch model, training, model registry, inference,
counterfactual simulation, FastAPI routes, and a minimal built-in web UI)
lives in this one module by design, so the whole system can be read,
forked, and run from a single file.

The database schema lives in the sibling file `schema.sql` (loaded and
executed once at startup -- idempotent, safe to re-run).

Run:
    pip install fastapi uvicorn[standard] pandas numpy scikit-learn torch python-multipart
    python postmortem.py
    # then open http://localhost:8000

Environment variables (all optional, sensible local-first defaults):
    POSTMORTEM_DATA_DIR          default: ./data
    POSTMORTEM_ARTIFACTS_DIR     default: ./artifacts
    POSTMORTEM_DB_PATH           default: ./data/postmortem.db
    POSTMORTEM_MAX_UPLOAD_BYTES  default: 25 MB
    POSTMORTEM_MAX_UPLOAD_ROWS   default: 200000
    POSTMORTEM_SEED              default: 42
    POSTMORTEM_MAX_EPOCHS        default: 60
    POSTMORTEM_MC_PASSES         default: 30
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================================================
# 1. CONFIG
# ==========================================================================

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"


class Settings:
    data_dir = Path(os.getenv("POSTMORTEM_DATA_DIR", str(BASE_DIR / "data")))
    artifacts_dir = Path(os.getenv("POSTMORTEM_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
    db_path = Path(os.getenv("POSTMORTEM_DB_PATH", str(BASE_DIR / "data" / "postmortem.db")))
    uploads_dir = data_dir / "uploads"

    max_upload_bytes = int(os.getenv("POSTMORTEM_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    max_upload_rows = int(os.getenv("POSTMORTEM_MAX_UPLOAD_ROWS", 200_000))

    text_embedding_dim = int(os.getenv("POSTMORTEM_TEXT_DIM", 256))
    random_seed = int(os.getenv("POSTMORTEM_SEED", 42))

    max_training_epochs = int(os.getenv("POSTMORTEM_MAX_EPOCHS", 60))
    max_training_seconds = int(os.getenv("POSTMORTEM_MAX_TRAIN_SECONDS", 600))
    max_concurrent_training_jobs = int(os.getenv("POSTMORTEM_MAX_CONCURRENT_JOBS", 1))
    mc_dropout_passes = int(os.getenv("POSTMORTEM_MC_PASSES", 30))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | postmortem | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("postmortem")


def new_id() -> str:
    """Server-side resource ID. Never accept one supplied by a client."""
    return str(uuid.uuid4())


def truncate_for_log(text: str | None, max_chars: int = 40) -> str:
    """Caption/free-text content must never be logged in full."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars] + "\u2026"


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Guards against path traversal / null-byte tricks in uploaded filenames."""
    name = Path(filename).name.replace("\x00", "")
    name = _SAFE_NAME_RE.sub("_", name)
    return (name or "upload.csv")[:255]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dir_checksum(directory: Path, exclude: set[str] = frozenset()) -> str:
    h = hashlib.sha256()
    for file in sorted(directory.rglob("*")):
        if file.is_file() and file.name not in exclude:
            h.update(file.name.encode("utf-8"))
            h.update(file.read_bytes())
    return h.hexdigest()


# ==========================================================================
# 2. DATABASE (raw sqlite3, schema loaded from schema.sql)
# ==========================================================================

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_db_lock = threading.Lock()
_conn = get_connection()


def init_db() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"schema.sql not found at {SCHEMA_PATH}. It must sit next to postmortem.py."
        )
    with _db_lock:
        _conn.executescript(SCHEMA_PATH.read_text())
        _conn.commit()
    logger.info("Database ready at %s", settings.db_path)


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    """Serializes all writes with a lock; sqlite3 in this process is single-connection."""
    with _db_lock:
        cur = _conn.cursor()
        try:
            yield cur
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


# ==========================================================================
# 3. INGESTION: CSV validation, normalization, quarantine
# ==========================================================================

REQUIRED_COLUMNS = ["caption", "platform", "post_type"]
OPTIONAL_NUMERIC = ["impressions", "likes", "comments", "shares", "saves"]
COLUMN_ALIASES = {
    "text": "caption", "content": "caption", "post_caption": "caption",
    "channel": "platform", "network": "platform",
    "type": "post_type", "content_type": "post_type",
    "created_at": "posted_at", "timestamp": "posted_at", "date": "posted_at",
    "tags": "hashtags", "views": "impressions", "reach": "impressions",
}
ENGAGEMENT_WEIGHTS = {"likes": 1.0, "comments": 3.0, "shares": 5.0, "saves": 4.0}
EPSILON = 1.0


class IngestionError(Exception):
    pass


@dataclass
class QuarantineIssue:
    row_index: int
    reason: str


@dataclass
class ValidationResult:
    normalized: pd.DataFrame
    quarantined: list[QuarantineIssue] = field(default_factory=list)
    total_rows: int = 0


def _read_csv_safely(raw_bytes: bytes) -> pd.DataFrame:
    if len(raw_bytes) > settings.max_upload_bytes:
        raise IngestionError(f"Upload exceeds max size of {settings.max_upload_bytes} bytes")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise IngestionError(f"Malformed CSV: {e}") from e
    if len(df) > settings.max_upload_rows:
        raise IngestionError(f"Upload has {len(df)} rows, exceeding max of {settings.max_upload_rows}")
    return df


def validate_and_normalize(raw_bytes: bytes) -> ValidationResult:
    df = _read_csv_safely(raw_bytes)

    rename_map = {c: COLUMN_ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in COLUMN_ALIASES}
    df = df.rename(columns=rename_map)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionError(f"Missing required columns: {missing}")

    quarantined: list[QuarantineIssue] = []
    keep = pd.Series(True, index=df.index)

    for col in REQUIRED_COLUMNS:
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for idx in df.index[empty & keep]:
            quarantined.append(QuarantineIssue(int(idx), f"missing required field '{col}'"))
        keep &= ~empty

    for col in OPTIONAL_NUMERIC:
        if col not in df.columns:
            df[col] = pd.NA
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & coerced.isna()
        for idx in df.index[bad & keep]:
            quarantined.append(QuarantineIssue(int(idx), f"non-numeric value in '{col}'"))
        keep &= ~bad
        df[col] = coerced
        negative = df[col].notna() & (df[col] < 0)
        for idx in df.index[negative & keep]:
            quarantined.append(QuarantineIssue(int(idx), f"negative value in '{col}'"))
        keep &= ~negative

    if "posted_at" in df.columns:
        parsed = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)
        bad_ts = df["posted_at"].notna() & (df["posted_at"].astype(str).str.strip() != "") & parsed.isna()
        for idx in df.index[bad_ts & keep]:
            quarantined.append(QuarantineIssue(int(idx), "unparseable posted_at timestamp"))
        keep &= ~bad_ts
        df["posted_at"] = parsed
    else:
        df["posted_at"] = pd.NaT

    df["hashtags"] = df.get("hashtags", "").fillna("").astype(str) if "hashtags" in df.columns else ""
    df["caption"] = df["caption"].astype(str)
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["post_type"] = df["post_type"].astype(str).str.strip().str.lower()

    normalized = df.loc[keep].copy()
    before = len(normalized)
    normalized = normalized.drop_duplicates(subset=["caption", "platform", "post_type", "posted_at"], keep="first")
    if before - len(normalized):
        logger.info("Dropped %d exact duplicate rows", before - len(normalized))

    return ValidationResult(normalized=normalized, quarantined=quarantined, total_rows=len(df))


def compute_engagement_rate(row: pd.Series) -> float | None:
    impressions = row.get("impressions")
    if pd.isna(impressions):
        return None
    numerator = sum(
        ENGAGEMENT_WEIGHTS.get(f, 0.0) * (row.get(f) or 0.0)
        for f in ("likes", "comments", "shares", "saves") if not pd.isna(row.get(f))
    )
    return float(numerator / max(float(impressions), EPSILON))


# ==========================================================================
# 4. FEATURE ENGINEERING
# ==========================================================================

FEATURE_SCHEMA_VERSION = "v1"
_HASHTAG_RE = re.compile(r"#\w+")
_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"[.!?]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_#@]+")


class HashingTextEncoder:
    """Deterministic, dependency-free text embedding (no network/model download needed)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _token_vector(self, token: str) -> np.ndarray:
        h = hashlib.sha256(token.lower().encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text or "")
            if not tokens:
                continue
            vecs = np.stack([self._token_vector(t) for t in tokens])
            mean = vecs.mean(axis=0)
            norm = np.linalg.norm(mean)
            out[i] = mean / norm if norm > 0 else mean
        return out


_TEXT_ENCODER = HashingTextEncoder(dim=settings.text_embedding_dim)


def _text_stats(captions: pd.Series) -> pd.DataFrame:
    c = captions.fillna("").astype(str)
    return pd.DataFrame({
        "caption_char_len": c.str.len(),
        "caption_word_count": c.apply(lambda t: len(_WORD_RE.findall(t))),
        "caption_sentence_count": c.apply(lambda t: max(1, len([s for s in _SENTENCE_RE.split(t) if s.strip()]))),
        "caption_question_count": c.str.count(r"\?"),
        "caption_exclaim_count": c.str.count("!"),
        "caption_punct_density": c.apply(lambda t: (sum(ch in ".,!?;:" for ch in t) / len(t)) if len(t) else 0.0),
    })


def _hashtag_stats(hashtags: pd.Series, caption: pd.Series) -> pd.DataFrame:
    combined = hashtags.fillna("").astype(str) + " " + caption.fillna("").astype(str)
    return pd.DataFrame({"hashtag_count": combined.apply(lambda t: len(_HASHTAG_RE.findall(t)))})


def _cyclical(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * series / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(radians), f"{prefix}_cos": np.cos(radians)})


def _time_features(posted_at: pd.Series) -> pd.DataFrame:
    posted_at = pd.to_datetime(posted_at, utc=True, errors="coerce")
    hour = posted_at.dt.hour.fillna(12).astype(float)
    weekday = posted_at.dt.weekday.fillna(3).astype(float)
    month = posted_at.dt.month.fillna(6).astype(float)
    out = pd.concat([_cyclical(hour, 24, "hour"), _cyclical(weekday, 7, "weekday"), _cyclical(month, 12, "month")], axis=1)
    out["has_timestamp"] = posted_at.notna().astype(float)
    return out.reset_index(drop=True)


@dataclass
class HistoricalStats:
    """Account-level stats frozen from the TRAINING split only (leakage prevention)."""
    global_mean: float = 0.0
    by_platform: dict = field(default_factory=dict)
    by_post_type: dict = field(default_factory=dict)

    def lookup(self, platform: str, post_type: str) -> tuple[float, float]:
        return (self.by_platform.get(platform, self.global_mean),
                self.by_post_type.get(post_type, self.global_mean))


class FeaturePipeline:
    """
    Fit once on the training split, then reused unchanged for validation,
    test, inference, and simulation -- never re-fit or re-implemented
    elsewhere (this is the single source of truth for feature logic).
    """

    def __init__(self, text_dim: int = 256):
        self.text_dim = text_dim
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.num_scaler = StandardScaler()
        self.historical_stats = HistoricalStats()
        self._fitted = False
        self._num_columns: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, engagement_col: str = "engagement_rate") -> "FeaturePipeline":
        train_df = train_df.reset_index(drop=True)
        self.cat_encoder.fit(train_df[["platform", "post_type"]].astype(str))

        valid = train_df[engagement_col].dropna()
        self.historical_stats.global_mean = float(valid.mean()) if len(valid) else 0.0
        self.historical_stats.by_platform = (
            train_df.assign(_y=train_df[engagement_col]).groupby("platform")["_y"].mean().dropna().to_dict()
        )
        self.historical_stats.by_post_type = (
            train_df.assign(_y=train_df[engagement_col]).groupby("post_type")["_y"].mean().dropna().to_dict()
        )

        numeric_block = self._numeric_block(train_df)
        self._num_columns = list(numeric_block.columns)
        self.num_scaler.fit(numeric_block.values)
        self._fitted = True
        return self

    def _numeric_block(self, df: pd.DataFrame) -> pd.DataFrame:
        text_stats = _text_stats(df["caption"])
        hashtag_stats = _hashtag_stats(df.get("hashtags", pd.Series([""] * len(df))), df["caption"])
        time_feats = _time_features(df["posted_at"]) if "posted_at" in df.columns else _time_features(pd.Series([pd.NaT] * len(df)))

        hist_platform, hist_post_type = [], []
        for _, row in df[["platform", "post_type"]].astype(str).iterrows():
            p, t = self.historical_stats.lookup(row["platform"], row["post_type"])
            hist_platform.append(p)
            hist_post_type.append(t)
        hist_df = pd.DataFrame({"hist_avg_engagement_platform": hist_platform, "hist_avg_engagement_post_type": hist_post_type})

        out = pd.concat([text_stats.reset_index(drop=True), hashtag_stats.reset_index(drop=True),
                          time_feats.reset_index(drop=True), hist_df.reset_index(drop=True)], axis=1)
        if self._num_columns is not None:
            out = out.reindex(columns=self._num_columns, fill_value=0.0)
        return out.fillna(0.0)

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fit before transform")
        df = df.reset_index(drop=True)
        text_emb = _TEXT_ENCODER.encode(df["caption"].fillna("").astype(str).tolist())
        numeric_scaled = self.num_scaler.transform(self._numeric_block(df).values)
        cat_encoded = self.cat_encoder.transform(df[["platform", "post_type"]].astype(str))
        metadata = np.concatenate([numeric_scaled, cat_encoded], axis=1).astype(np.float32)
        return {"text": text_emb.astype(np.float32), "metadata": metadata}

    @property
    def metadata_dim(self) -> int:
        return len(self._num_columns) + len(self.cat_encoder.get_feature_names_out())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Undated rows go to train only, never val/test -- their order can't be verified."""
    df = df.copy()
    undated = df[df["posted_at"].isna()]
    dated = df[df["posted_at"].notna()].sort_values("posted_at")
    n = len(dated)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    n_train = n - n_val - n_test
    train = pd.concat([undated, dated.iloc[:n_train]], axis=0).reset_index(drop=True)
    val = dated.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = dated.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


def build_targets(train, val, test, engagement_col="engagement_rate", top_pct=0.8):
    """Median/top-20% thresholds computed from TRAIN split only, applied unchanged elsewhere."""
    train_targets = train[engagement_col].dropna()
    median_threshold = float(train_targets.median()) if len(train_targets) else 0.0
    top20_threshold = float(np.quantile(train_targets, top_pct)) if len(train_targets) else 0.0
    for split in (train, val, test):
        split["label_above_median"] = (split[engagement_col] >= median_threshold).astype(float)
        split["label_top20"] = (split[engagement_col] >= top20_threshold).astype(float)
        missing = split[engagement_col].isna()
        split.loc[missing, ["label_above_median", "label_top20"]] = np.nan
    return median_threshold, top20_threshold


# ==========================================================================
# 5. MODELS: baseline + PyTorch multi-task fusion model
# ==========================================================================

@dataclass
class FusionConfig:
    text_dim: int
    metadata_dim: int
    text_proj_dim: int = 64
    metadata_hidden_dim: int = 64
    trunk_hidden_dim: int = 64
    dropout: float = 0.3


class FusionModel(nn.Module):
    """
    text_embedding --[proj MLP]--\\
                                   +--> concat --> trunk --> 3 heads
    metadata        --[enc MLP]--/

    Heads: regression (engagement score), median classifier, top-20% classifier.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.text_proj_dim), nn.LayerNorm(config.text_proj_dim),
            nn.ReLU(), nn.Dropout(config.dropout),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(config.metadata_dim, config.metadata_hidden_dim), nn.LayerNorm(config.metadata_hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout),
        )
        fused_dim = config.text_proj_dim + config.metadata_hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, config.trunk_hidden_dim), nn.LayerNorm(config.trunk_hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout),
        )
        self.regression_head = nn.Linear(config.trunk_hidden_dim, 1)
        self.median_head = nn.Linear(config.trunk_hidden_dim, 1)
        self.top20_head = nn.Linear(config.trunk_hidden_dim, 1)

    def forward(self, text: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = torch.cat([self.text_proj(text), self.metadata_encoder(metadata)], dim=-1)
        trunk_out = self.trunk(fused)
        return {
            "regression": self.regression_head(trunk_out).squeeze(-1),
            "median_logit": self.median_head(trunk_out).squeeze(-1),
            "top20_logit": self.top20_head(trunk_out).squeeze(-1),
        }


def multi_task_loss(outputs, y_reg, y_median, y_top20, reg_mask, cls_mask,
                     lambda_reg=1.0, lambda_median=1.0, lambda_top20=1.0):
    """Masks exclude rows with unknown targets from the relevant loss term (no fake labels)."""
    huber, bce = nn.HuberLoss(), nn.BCEWithLogitsLoss()
    zero = torch.tensor(0.0)
    l_reg = huber(outputs["regression"][reg_mask], y_reg[reg_mask]) if reg_mask.any() else zero
    l_med = bce(outputs["median_logit"][cls_mask], y_median[cls_mask]) if cls_mask.any() else zero
    l_top = bce(outputs["top20_logit"][cls_mask], y_top20[cls_mask]) if cls_mask.any() else zero
    total = lambda_reg * l_reg + lambda_median * l_med + lambda_top20 * l_top
    return total, {"loss_total": float(total.detach()), "loss_regression": float(l_reg.detach()),
                    "loss_median": float(l_med.detach()), "loss_top20": float(l_top.detach())}


@torch.no_grad()
def mc_dropout_predict(model: FusionModel, text: torch.Tensor, metadata: torch.Tensor, n_passes: int = 30) -> dict:
    """MC-Dropout uncertainty: std is a warning signal of low confidence, not a calibrated interval."""
    was_training = model.training
    model.train()
    reg, med, top = [], [], []
    for _ in range(n_passes):
        out = model(text, metadata)
        reg.append(out["regression"])
        med.append(torch.sigmoid(out["median_logit"]))
        top.append(torch.sigmoid(out["top20_logit"]))
    model.train(was_training)
    reg_s, med_s, top_s = torch.stack(reg), torch.stack(med), torch.stack(top)
    return {
        "regression_mean": reg_s.mean(0), "regression_std": reg_s.std(0),
        "median_prob_mean": med_s.mean(0), "top20_prob_mean": top_s.mean(0),
    }


def train_baseline(metadata_X: np.ndarray, y_reg: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    """The fusion model's regression head must beat this to be worth its complexity."""
    mask = ~np.isnan(y_reg)
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(metadata_X[mask], y_reg[mask])
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    return {"mae": float(mean_absolute_error(yt, yp)), "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))), "n": int(mask.sum())}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    mask = ~np.isnan(y_true)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return {"pr_auc": None, "roc_auc": None, "f1": None, "n": int(mask.sum())}
    yt, yp = y_true[mask], y_prob[mask]
    preds = (yp >= 0.5).astype(int)
    return {"pr_auc": float(average_precision_score(yt, yp)), "roc_auc": float(roc_auc_score(yt, yp)),
            "f1": float(f1_score(yt, preds, zero_division=0)), "n": int(mask.sum())}


# ==========================================================================
# 6. TRAINING PIPELINE
# ==========================================================================

@dataclass
class TrainingResult:
    feature_pipeline: FeaturePipeline
    fusion_model: FusionModel
    baseline_model: HistGradientBoostingRegressor
    metrics: dict
    config: dict
    median_threshold: float
    top20_threshold: float


def _to_tensors(features: dict, df: pd.DataFrame) -> dict:
    y_reg = df["engagement_rate"].astype(float).values
    y_med = df["label_above_median"].astype(float).values
    y_top = df["label_top20"].astype(float).values
    return {
        "text": torch.tensor(features["text"], dtype=torch.float32),
        "metadata": torch.tensor(features["metadata"], dtype=torch.float32),
        "y_reg": torch.tensor(np.nan_to_num(y_reg), dtype=torch.float32),
        "y_median": torch.tensor(np.nan_to_num(y_med), dtype=torch.float32),
        "y_top20": torch.tensor(np.nan_to_num(y_top), dtype=torch.float32),
        "reg_mask": torch.tensor(~np.isnan(y_reg)),
        "cls_mask": torch.tensor(~np.isnan(y_med)),
    }


def run_training_pipeline(posts_df: pd.DataFrame, epochs: int | None = None, seed: int | None = None) -> TrainingResult:
    seed = seed if seed is not None else settings.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    epochs = epochs or settings.max_training_epochs

    train_df, val_df, test_df = chronological_split(posts_df)
    median_threshold, top20_threshold = build_targets(train_df, val_df, test_df)

    pipeline = FeaturePipeline(text_dim=settings.text_embedding_dim).fit(train_df)
    train_feat = pipeline.transform(train_df)
    val_feat = pipeline.transform(val_df) if len(val_df) else train_feat
    test_feat = pipeline.transform(test_df) if len(test_df) else train_feat

    baseline_model = train_baseline(train_feat["metadata"], train_df["engagement_rate"].astype(float).values, seed)

    config = FusionConfig(text_dim=train_feat["text"].shape[1], metadata_dim=pipeline.metadata_dim)
    model = FusionModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_t = _to_tensors(train_feat, train_df)
    val_t = _to_tensors(val_feat, val_df) if len(val_df) else train_t
    dataset = TensorDataset(train_t["text"], train_t["metadata"], train_t["y_reg"], train_t["y_median"],
                             train_t["y_top20"], train_t["reg_mask"], train_t["cls_mask"])
    loader = DataLoader(dataset, batch_size=min(32, max(1, len(dataset))), shuffle=True)

    best_val_loss, best_state, patience_left = float("inf"), None, 8
    start = time.time()
    for epoch in range(epochs):
        if time.time() - start > settings.max_training_seconds:
            logger.warning("Training stopped early: max_training_seconds reached")
            break
        model.train()
        for text, meta, y_reg, y_med, y_top, reg_mask, cls_mask in loader:
            optimizer.zero_grad()
            out = model(text, meta)
            loss, _ = multi_task_loss(out, y_reg, y_med, y_top, reg_mask, cls_mask)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(val_t["text"], val_t["metadata"])
            val_loss, _ = multi_task_loss(val_out, val_t["y_reg"], val_t["y_median"], val_t["y_top20"],
                                           val_t["reg_mask"], val_t["cls_mask"])
        if float(val_loss) < best_val_loss - 1e-5:
            best_val_loss, best_state, patience_left = float(val_loss), {k: v.clone() for k, v in model.state_dict().items()}, 8
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_t = _to_tensors(test_feat, test_df) if len(test_df) else val_t
    model.eval()
    mc = mc_dropout_predict(model, test_t["text"], test_t["metadata"], n_passes=settings.mc_dropout_passes)

    y_reg_true = test_t["y_reg"].numpy().copy()
    y_reg_true[~test_t["reg_mask"].numpy()] = np.nan
    y_med_true = test_t["y_median"].numpy().copy()
    y_med_true[~test_t["cls_mask"].numpy()] = np.nan

    metrics = {
        "fusion_regression": regression_metrics(y_reg_true, mc["regression_mean"].numpy()),
        "fusion_median_classification": classification_metrics(y_med_true, mc["median_prob_mean"].numpy()),
        "baseline_regression": regression_metrics(y_reg_true, baseline_model.predict(test_feat["metadata"])),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    logger.info("Training complete | fusion MAE=%s | baseline MAE=%s",
                metrics["fusion_regression"]["mae"], metrics["baseline_regression"]["mae"])

    return TrainingResult(
        feature_pipeline=pipeline, fusion_model=model, baseline_model=baseline_model, metrics=metrics,
        config={"epochs_requested": epochs, "random_seed": seed, "fusion_config": config.__dict__},
        median_threshold=median_threshold, top20_threshold=top20_threshold,
    )


# ==========================================================================
# 7. MODEL REGISTRY (checksum-verified artifact save/load)
# ==========================================================================

class ArtifactIntegrityError(Exception):
    pass


@dataclass
class LoadedArtifact:
    model_run_id: str
    fusion_model: FusionModel
    feature_pipeline: FeaturePipeline
    median_threshold: float
    top20_threshold: float


def save_artifact(project_id: str, model_run_id: str, result: TrainingResult) -> dict:
    artifact_dir = settings.artifacts_dir / project_id / model_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_state_path = artifact_dir / "fusion_model_state.pt"
    torch.save(result.fusion_model.state_dict(), model_state_path)

    preprocessing_path = artifact_dir / "feature_pipeline.pkl"
    result.feature_pipeline.save(preprocessing_path)

    with open(artifact_dir / "baseline_model.pkl", "wb") as f:
        pickle.dump(result.baseline_model, f)

    meta = {
        "config": result.config, "metrics": result.metrics,
        "median_threshold": result.median_threshold, "top20_threshold": result.top20_threshold,
        "fusion_config": result.config["fusion_config"],
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    checksum = dir_checksum(artifact_dir)
    (artifact_dir / "CHECKSUM.sha256").write_text(checksum)

    return {"artifact_dir": str(artifact_dir), "model_state_path": str(model_state_path),
            "preprocessing_path": str(preprocessing_path), "checksum_sha256": checksum}


def load_artifact(artifact_dir: str, expected_checksum: str, model_run_id: str) -> LoadedArtifact:
    """Refuses to load if the recomputed checksum doesn't match -- guards against tampering/corruption."""
    directory = Path(artifact_dir)
    checksum_file = directory / "CHECKSUM.sha256"
    if not directory.exists() or not checksum_file.exists():
        raise ArtifactIntegrityError(f"Artifact or checksum missing at {artifact_dir}")

    recomputed = dir_checksum(directory, exclude={"CHECKSUM.sha256"})
    stored = checksum_file.read_text().strip()
    if recomputed != stored or recomputed != expected_checksum:
        raise ArtifactIntegrityError("Artifact checksum mismatch -- possible tampering or corruption. Refusing to load.")

    meta = json.loads((directory / "meta.json").read_text())
    feature_pipeline = FeaturePipeline.load(directory / "feature_pipeline.pkl")
    fusion_config = FusionConfig(**meta["fusion_config"])
    fusion_model = FusionModel(fusion_config)
    fusion_model.load_state_dict(torch.load(directory / "fusion_model_state.pt", map_location="cpu"))
    fusion_model.eval()

    return LoadedArtifact(model_run_id=model_run_id, fusion_model=fusion_model, feature_pipeline=feature_pipeline,
                           median_threshold=meta["median_threshold"], top20_threshold=meta["top20_threshold"])


def resolve_active_artifact(project_id: str, model_run_id: str | None) -> LoadedArtifact:
    with db_cursor() as cur:
        if model_run_id:
            cur.execute(
                "SELECT a.artifact_dir, a.checksum_sha256, r.id as run_id FROM model_artifacts a "
                "JOIN model_runs r ON a.model_run_id = r.id WHERE r.project_id=? AND r.id=?",
                (project_id, model_run_id),
            )
        else:
            cur.execute(
                "SELECT a.artifact_dir, a.checksum_sha256, r.id as run_id FROM model_artifacts a "
                "JOIN model_runs r ON a.model_run_id = r.id WHERE r.project_id=? AND a.is_active=1",
                (project_id,),
            )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=409, detail="No trained model for this project yet. Train one first.")
    return load_artifact(row["artifact_dir"], row["checksum_sha256"], row["run_id"])


# ==========================================================================
# 8. INFERENCE + COUNTERFACTUAL SIMULATION
# ==========================================================================

HIGH_UNCERTAINTY_STD = 0.15
ALLOWED_OVERRIDE_FIELDS = {"caption", "platform", "post_type", "posted_at", "hashtags"}


def predict_draft(artifact: LoadedArtifact, draft: "DraftInput", scenario_label: str = "baseline") -> "PredictionOut":
    df = pd.DataFrame([{
        "caption": draft.caption, "platform": draft.platform.strip().lower(),
        "post_type": draft.post_type.strip().lower(), "posted_at": draft.posted_at, "hashtags": draft.hashtags,
    }])
    features = artifact.feature_pipeline.transform(df)
    text = torch.tensor(features["text"], dtype=torch.float32)
    metadata = torch.tensor(features["metadata"], dtype=torch.float32)
    mc = mc_dropout_predict(artifact.fusion_model, text, metadata, n_passes=settings.mc_dropout_passes)

    reg_pred, reg_std = float(mc["regression_mean"][0]), float(mc["regression_std"][0])
    note = ("Low confidence: this draft looks unlike the account's historical evidence."
            if reg_std >= HIGH_UNCERTAINTY_STD else
            "Prediction is within the range of patterns seen in this account's history.")

    return PredictionOut(
        scenario_label=scenario_label, regression_pred=reg_pred,
        median_prob=float(mc["median_prob_mean"][0]), top20_prob=float(mc["top20_prob_mean"][0]),
        uncertainty_std=reg_std, confidence_note=note,
    )


def run_simulation(artifact: LoadedArtifact, baseline: "DraftInput", scenarios: list["ScenarioOverride"]) -> "SimulateResponse":
    baseline_pred = predict_draft(artifact, baseline, "baseline")
    results = []
    for scenario in scenarios:
        disallowed = set(scenario.overrides) - ALLOWED_OVERRIDE_FIELDS
        if disallowed:
            raise HTTPException(status_code=400, detail=f"Cannot override non-controllable fields: {disallowed}")
        data = baseline.model_dump()
        data.update(scenario.overrides)
        scenario_draft = DraftInput(**data)
        scenario_pred = predict_draft(artifact, scenario_draft, scenario.label)
        abs_diff = scenario_pred.regression_pred - baseline_pred.regression_pred
        pct_diff = (abs_diff / abs(baseline_pred.regression_pred) * 100.0) if baseline_pred.regression_pred else None
        results.append(ScenarioResult(label=scenario.label, prediction=scenario_pred,
                                       abs_diff_vs_baseline=abs_diff, pct_diff_vs_baseline=pct_diff))
    return SimulateResponse(baseline=baseline_pred, scenarios=results)


# ==========================================================================
# 9. BACKGROUND TRAINING ORCHESTRATION
# ==========================================================================

_training_semaphore = threading.Semaphore(settings.max_concurrent_training_jobs)


def start_training_job(project_id: str, dataset_id: str, epochs: int | None, seed: int | None) -> str:
    job_id = new_id()
    with db_cursor() as cur:
        cur.execute("INSERT INTO training_jobs (id, project_id, dataset_id, status) VALUES (?,?,?,'queued')",
                    (job_id, project_id, dataset_id))
    threading.Thread(target=_run_training_job, args=(job_id, project_id, dataset_id, epochs, seed), daemon=True).start()
    return job_id


def _run_training_job(job_id: str, project_id: str, dataset_id: str, epochs: int | None, seed: int | None) -> None:
    if not _training_semaphore.acquire(timeout=1):
        _set_job(job_id, status="failed", error_message="Too many concurrent training jobs; try again shortly.")
        return
    try:
        _set_job(job_id, status="running", started_at=True)
        with db_cursor() as cur:
            cur.execute("SELECT * FROM posts WHERE project_id=? AND dataset_id=?", (project_id, dataset_id))
            rows = [dict(r) for r in cur.fetchall()]
        if len(rows) < 5:
            raise ValueError("Not enough posts to train on (need at least 5).")

        df = pd.DataFrame(rows)
        df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True, errors="coerce")
        result = run_training_pipeline(df, epochs=epochs, seed=seed)

        model_run_id = new_id()
        artifact_info = save_artifact(project_id, model_run_id, result)

        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO model_runs (id, project_id, dataset_id, model_kind, config_json, metrics_json, "
                "feature_schema_version, random_seed, median_threshold, top20_threshold, status) "
                "VALUES (?,?,?,'fusion',?,?,?,?,?,?,'completed')",
                (model_run_id, project_id, dataset_id, json.dumps(result.config, default=str),
                 json.dumps(result.metrics, default=str), FEATURE_SCHEMA_VERSION,
                 result.config["random_seed"], result.median_threshold, result.top20_threshold),
            )
            cur.execute(
                "UPDATE model_artifacts SET is_active=0 WHERE model_run_id IN "
                "(SELECT id FROM model_runs WHERE project_id=?)", (project_id,),
            )
            cur.execute(
                "INSERT INTO model_artifacts (id, model_run_id, artifact_dir, model_state_path, "
                "preprocessing_path, checksum_sha256, is_active) VALUES (?,?,?,?,?,?,1)",
                (new_id(), model_run_id, artifact_info["artifact_dir"], artifact_info["model_state_path"],
                 artifact_info["preprocessing_path"], artifact_info["checksum_sha256"]),
            )
            cur.execute(
                "UPDATE training_jobs SET status='completed', model_run_id=?, finished_at=datetime('now') WHERE id=?",
                (model_run_id, job_id),
            )
        logger.info("Training job %s completed -> model_run %s", job_id, model_run_id)
    except Exception as e:
        logger.exception("Training job %s failed", job_id)
        _set_job(job_id, status="failed", error_message=str(e))
    finally:
        _training_semaphore.release()


def _set_job(job_id: str, status: str, error_message: str | None = None, started_at: bool = False) -> None:
    with db_cursor() as cur:
        if started_at:
            cur.execute("UPDATE training_jobs SET status=?, started_at=datetime('now') WHERE id=?", (status, job_id))
        elif status == "failed":
            cur.execute("UPDATE training_jobs SET status=?, error_message=?, finished_at=datetime('now') WHERE id=?",
                        (status, error_message, job_id))
        else:
            cur.execute("UPDATE training_jobs SET status=? WHERE id=?", (status, job_id))


# ==========================================================================
# 10. API SCHEMAS
# ==========================================================================

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class TrainRequest(BaseModel):
    dataset_id: str
    epochs: int | None = Field(default=None, ge=1, le=200)
    random_seed: int | None = None


class DraftInput(BaseModel):
    caption: str = Field(default="", max_length=10_000)
    platform: str
    post_type: str
    posted_at: Optional[str] = None
    hashtags: str = ""

    @field_validator("caption")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class PredictRequest(BaseModel):
    model_run_id: str | None = None
    draft: DraftInput


class PredictionOut(BaseModel):
    scenario_label: str
    regression_pred: float
    median_prob: float
    top20_prob: float
    uncertainty_std: float
    confidence_note: str


class ScenarioOverride(BaseModel):
    label: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class SimulateRequest(BaseModel):
    model_run_id: str | None = None
    baseline: DraftInput
    scenarios: list[ScenarioOverride]


class ScenarioResult(BaseModel):
    label: str
    prediction: PredictionOut
    abs_diff_vs_baseline: float
    pct_diff_vs_baseline: float | None


class SimulateResponse(BaseModel):
    baseline: PredictionOut
    scenarios: list[ScenarioResult]
    disclaimer: str = (
        "Counterfactual output is model-based, not causal proof. Under this model and this "
        "account's historical data, the predicted outcome changes as shown -- this is not a "
        "guarantee of real-world effect."
    )


# ==========================================================================
# 11. FASTAPI APP
# ==========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("PostMortem started")
    yield


app = FastAPI(title="PostMortem", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/projects")
def create_project(payload: ProjectCreate):
    project_id = new_id()
    with db_cursor() as cur:
        cur.execute("INSERT INTO projects (id, name) VALUES (?, ?)", (project_id, payload.name))
        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = cur.fetchone()
    return row_to_dict(row)


@app.get("/projects")
def list_projects():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


@app.post("/projects/{project_id}/datasets")
async def upload_dataset(project_id: str, file: UploadFile):
    with db_cursor() as cur:
        cur.execute("SELECT id FROM projects WHERE id=?", (project_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    raw_bytes = await file.read()
    try:
        result = validate_and_normalize(raw_bytes)
    except IngestionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    dataset_id = new_id()
    project_dir = settings.uploads_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    stored_path = project_dir / f"{dataset_id}_{sanitize_filename(file.filename)}"
    stored_path.write_bytes(raw_bytes)
    checksum = sha256_of_bytes(raw_bytes)

    normalized = result.normalized.copy()
    normalized["engagement_rate"] = normalized.apply(compute_engagement_rate, axis=1)

    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO datasets (id, project_id, original_filename, stored_path, checksum_sha256, "
            "row_count, quarantined_row_count, status) VALUES (?,?,?,?,?,?,?,'validated')",
            (dataset_id, project_id, file.filename, str(stored_path), checksum,
             len(normalized), len(result.quarantined)),
        )
        for issue in result.quarantined[:200]:
            cur.execute("INSERT INTO dataset_issues (dataset_id, row_index, reason) VALUES (?,?,?)",
                        (dataset_id, issue.row_index, issue.reason))
        for _, row in normalized.iterrows():
            posted_at = row["posted_at"].isoformat() if pd.notna(row["posted_at"]) else None
            cur.execute(
                "INSERT INTO posts (id, project_id, dataset_id, caption, platform, post_type, posted_at, "
                "hashtags, impressions, likes, comments, shares, saves, engagement_rate) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id(), project_id, dataset_id, row["caption"], row["platform"], row["post_type"], posted_at,
                 row["hashtags"], _n(row.get("impressions")), _n(row.get("likes")), _n(row.get("comments")),
                 _n(row.get("shares")), _n(row.get("saves")), _n(row.get("engagement_rate"))),
            )

    return {"dataset_id": dataset_id, "total_rows": result.total_rows, "valid_rows": len(normalized),
            "quarantined_rows": len(result.quarantined),
            "issues_sample": [{"row_index": q.row_index, "reason": q.reason} for q in result.quarantined[:50]]}


def _n(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


@app.post("/projects/{project_id}/train")
def start_training(project_id: str, payload: TrainRequest):
    with db_cursor() as cur:
        cur.execute("SELECT id FROM projects WHERE id=?", (project_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")
    job_id = start_training_job(project_id, payload.dataset_id, payload.epochs, payload.random_seed)
    with db_cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=?", (job_id,))
        return row_to_dict(cur.fetchone())


@app.get("/training-jobs/{job_id}")
def get_training_job(job_id: str):
    with db_cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return row_to_dict(row)


@app.get("/projects/{project_id}/models")
def list_models(project_id: str):
    with db_cursor() as cur:
        cur.execute("SELECT id, model_kind, feature_schema_version, status, created_at, metrics_json "
                    "FROM model_runs WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        return [dict(r) for r in cur.fetchall()]


@app.post("/projects/{project_id}/predict", response_model=PredictionOut)
def predict(project_id: str, payload: PredictRequest):
    artifact = resolve_active_artifact(project_id, payload.model_run_id)
    prediction = predict_draft(artifact, payload.draft)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (id, project_id, model_run_id, scenario_label, input_json, "
            "regression_pred, median_prob, top20_prob, uncertainty_std) VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), project_id, artifact.model_run_id, "baseline",
             payload.draft.model_dump_json(), prediction.regression_pred, prediction.median_prob,
             prediction.top20_prob, prediction.uncertainty_std),
        )
    logger.info("Prediction served | project=%s caption='%s'", project_id, truncate_for_log(payload.draft.caption))
    return prediction


@app.post("/projects/{project_id}/simulate", response_model=SimulateResponse)
def simulate(project_id: str, payload: SimulateRequest):
    artifact = resolve_active_artifact(project_id, payload.model_run_id)
    return run_simulation(artifact, payload.baseline, payload.scenarios)


@app.get("/projects/{project_id}/analytics")
def analytics(project_id: str):
    with db_cursor() as cur:
        cur.execute("SELECT engagement_rate, post_type, platform FROM posts WHERE project_id=?", (project_id,))
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return {"project_id": project_id, "post_count": 0, "avg_engagement_rate": None,
                "median_engagement_rate": None, "top20_threshold": None, "by_post_type": {}, "by_platform": {}}
    df = pd.DataFrame(rows)
    rates = df["engagement_rate"].dropna()
    return {
        "project_id": project_id, "post_count": len(rows),
        "avg_engagement_rate": float(rates.mean()) if len(rates) else None,
        "median_engagement_rate": float(rates.median()) if len(rates) else None,
        "top20_threshold": float(np.quantile(rates, 0.8)) if len(rates) else None,
        "by_post_type": df.groupby("post_type")["engagement_rate"].mean().dropna().to_dict(),
        "by_platform": df.groupby("platform")["engagement_rate"].mean().dropna().to_dict(),
    }


# ==========================================================================
# 12. MINIMAL BUILT-IN UI (single page, no separate frontend process needed)
# ==========================================================================

_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>PostMortem</title>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:880px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:2rem;border-top:1px solid #eee;padding-top:1rem}
input,select,textarea,button{font:inherit;padding:.4rem;margin:.2rem 0;width:100%;box-sizing:border-box}
button{background:#111;color:#fff;border:0;border-radius:4px;cursor:pointer;width:auto;padding:.5rem 1rem}
pre{background:#f6f6f6;padding:.75rem;border-radius:4px;overflow:auto;font-size:.82rem}
.row{display:flex;gap:.5rem} .row>*{flex:1}
label{font-size:.82rem;color:#555;display:block;margin-top:.5rem}
</style></head>
<body>
<h1>PostMortem</h1>
<p>Personalized content intelligence, not universal virality prediction.</p>

<h2>1. Project</h2>
<label>Name</label><input id="projName" value="My Account">
<button onclick="createProject()">Create project</button>
<div id="projectId"></div>

<h2>2. Upload CSV (required columns: caption, platform, post_type)</h2>
<input type="file" id="csvFile" accept=".csv">
<button onclick="uploadCsv()">Upload &amp; validate</button>
<pre id="uploadResult"></pre>

<h2>3. Train</h2>
<label>Dataset ID</label><input id="datasetId">
<button onclick="train()">Start training</button>
<button onclick="checkJob()">Check job status</button>
<pre id="trainResult"></pre>

<h2>4. Predict</h2>
<label>Caption</label><textarea id="caption" rows="3">a great training update</textarea>
<div class="row">
  <div><label>Platform</label><input id="platform" value="instagram"></div>
  <div><label>Post type</label>
    <select id="postType"><option>image</option><option>video</option><option>carousel</option><option selected>reel</option></select>
  </div>
</div>
<button onclick="predict()">Predict</button>
<pre id="predictResult"></pre>

<h2>5. Counterfactual: change post type</h2>
<label>Alternative post type</label>
<select id="altPostType"><option>image</option><option selected>video</option><option>carousel</option><option>reel</option></select>
<button onclick="simulate()">Run simulation</button>
<pre id="simResult"></pre>

<h2>Analytics</h2>
<button onclick="loadAnalytics()">Refresh</button>
<pre id="analyticsResult"></pre>

<script>
let projectId = null, datasetId = null, jobId = null;

async function api(method, path, body, isForm) {
  const opts = {method, headers: isForm ? {} : {'Content-Type':'application/json'}};
  if (body) opts.body = isForm ? body : JSON.stringify(body);
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

async function createProject() {
  const p = await api('POST', '/projects', {name: document.getElementById('projName').value});
  projectId = p.id;
  document.getElementById('projectId').innerText = 'Project ID: ' + projectId;
}

async function uploadCsv() {
  if (!projectId) return alert('Create a project first');
  const file = document.getElementById('csvFile').files[0];
  if (!file) return alert('Choose a CSV file');
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await api('POST', `/projects/${projectId}/datasets`, fd, true);
    datasetId = res.dataset_id;
    document.getElementById('datasetId').value = datasetId;
    document.getElementById('uploadResult').innerText = JSON.stringify(res, null, 2);
  } catch (e) { document.getElementById('uploadResult').innerText = 'Error: ' + e.message; }
}

async function train() {
  if (!projectId) return alert('Create a project first');
  const dsId = document.getElementById('datasetId').value;
  try {
    const res = await api('POST', `/projects/${projectId}/train`, {dataset_id: dsId, epochs: 20});
    jobId = res.id;
    document.getElementById('trainResult').innerText = JSON.stringify(res, null, 2);
  } catch (e) { document.getElementById('trainResult').innerText = 'Error: ' + e.message; }
}

async function checkJob() {
  if (!jobId) return alert('Start training first');
  const res = await api('GET', `/training-jobs/${jobId}`);
  document.getElementById('trainResult').innerText = JSON.stringify(res, null, 2);
}

async function predict() {
  if (!projectId) return alert('Create a project first');
  const draft = {caption: document.getElementById('caption').value, platform: document.getElementById('platform').value,
                 post_type: document.getElementById('postType').value};
  try {
    const res = await api('POST', `/projects/${projectId}/predict`, {draft});
    document.getElementById('predictResult').innerText = JSON.stringify(res, null, 2);
  } catch (e) { document.getElementById('predictResult').innerText = 'Error: ' + e.message; }
}

async function simulate() {
  if (!projectId) return alert('Create a project first');
  const baseline = {caption: document.getElementById('caption').value, platform: document.getElementById('platform').value,
                     post_type: document.getElementById('postType').value};
  const alt = document.getElementById('altPostType').value;
  try {
    const res = await api('POST', `/projects/${projectId}/simulate`,
      {baseline, scenarios: [{label: 'post_type=' + alt, overrides: {post_type: alt}}]});
    document.getElementById('simResult').innerText = JSON.stringify(res, null, 2);
  } catch (e) { document.getElementById('simResult').innerText = 'Error: ' + e.message; }
}

async function loadAnalytics() {
  if (!projectId) return alert('Create a project first');
  const res = await api('GET', `/projects/${projectId}/analytics`);
  document.getElementById('analyticsResult').innerText = JSON.stringify(res, null, 2);
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui_home():
    return _UI_HTML


# ==========================================================================
# 13. ENTRYPOINT
# ==========================================================================

if __name__ == "__main__":
    uvicorn.run("postmortem:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
