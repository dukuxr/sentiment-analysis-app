"""
Persistence helpers for automation/testing evidence.
Stores run-level and prediction-level data in SQLite.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import uuid

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DB_PATH = RESULTS_DIR / "sentiment_app.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def generate_run_id(prefix: str) -> str:
    token = uuid.uuid4().hex[:8].upper()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{ts}-{token}"


def initialize_database():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                mode TEXT NOT NULL,
                source TEXT,
                model_type TEXT,
                model_accuracy_pct REAL,
                total_records INTEGER,
                processed_records INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                avg_confidence REAL,
                duration_seconds REAL,
                throughput_rows_per_sec REAL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1 REAL,
                tp INTEGER,
                tn INTEGER,
                fp INTEGER,
                fn INTEGER,
                notes TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                row_index INTEGER,
                input_text_preview TEXT,
                input_text_hash TEXT,
                predicted_sentiment TEXT NOT NULL,
                confidence REAL,
                actual_sentiment TEXT,
                is_correct INTEGER,
                error_type TEXT,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_run_id
            ON predictions(run_id)
            """
        )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_run(run_data: dict, prediction_rows: list[dict] | None = None):
    if not run_data.get("run_id"):
        raise ValueError("run_data must include run_id")

    prediction_rows = prediction_rows or []

    run_payload = {
        "run_id": run_data.get("run_id"),
        "created_at_utc": run_data.get("created_at_utc") or _utc_now_iso(),
        "mode": run_data.get("mode"),
        "source": run_data.get("source"),
        "model_type": run_data.get("model_type"),
        "model_accuracy_pct": _to_float(run_data.get("model_accuracy_pct")),
        "total_records": _to_int(run_data.get("total_records")),
        "processed_records": _to_int(run_data.get("processed_records")),
        "positive_count": _to_int(run_data.get("positive_count")),
        "negative_count": _to_int(run_data.get("negative_count")),
        "avg_confidence": _to_float(run_data.get("avg_confidence")),
        "duration_seconds": _to_float(run_data.get("duration_seconds")),
        "throughput_rows_per_sec": _to_float(run_data.get("throughput_rows_per_sec")),
        "accuracy": _to_float(run_data.get("accuracy")),
        "precision": _to_float(run_data.get("precision")),
        "recall": _to_float(run_data.get("recall")),
        "f1": _to_float(run_data.get("f1")),
        "tp": _to_int(run_data.get("tp")),
        "tn": _to_int(run_data.get("tn")),
        "fp": _to_int(run_data.get("fp")),
        "fn": _to_int(run_data.get("fn")),
        "notes": run_data.get("notes"),
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, created_at_utc, mode, source, model_type, model_accuracy_pct,
                total_records, processed_records, positive_count, negative_count,
                avg_confidence, duration_seconds, throughput_rows_per_sec,
                accuracy, precision, recall, f1, tp, tn, fp, fn, notes
            ) VALUES (
                :run_id, :created_at_utc, :mode, :source, :model_type, :model_accuracy_pct,
                :total_records, :processed_records, :positive_count, :negative_count,
                :avg_confidence, :duration_seconds, :throughput_rows_per_sec,
                :accuracy, :precision, :recall, :f1, :tp, :tn, :fp, :fn, :notes
            )
            """,
            run_payload,
        )

        conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_payload["run_id"],))

        prediction_payload = []
        for item in prediction_rows:
            raw_text = str(item.get("input_text", ""))
            preview = raw_text if len(raw_text) <= 180 else raw_text[:177] + "..."
            prediction_payload.append(
                (
                    run_payload["run_id"],
                    _to_int(item.get("row_index")),
                    preview,
                    _text_hash(raw_text),
                    item.get("predicted_sentiment"),
                    _to_float(item.get("confidence")),
                    item.get("actual_sentiment"),
                    _to_int(item.get("is_correct")),
                    item.get("error_type"),
                    _utc_now_iso(),
                )
            )

        if prediction_payload:
            conn.executemany(
                """
                INSERT INTO predictions (
                    run_id, row_index, input_text_preview, input_text_hash,
                    predicted_sentiment, confidence, actual_sentiment,
                    is_correct, error_type, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prediction_payload,
            )


def get_recent_runs(limit: int = 100) -> pd.DataFrame:
    query = """
        SELECT *
        FROM runs
        ORDER BY created_at_utc DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=[limit])


def get_predictions_for_run(run_id: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM predictions
        WHERE run_id = ?
        ORDER BY id ASC
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=[run_id])

