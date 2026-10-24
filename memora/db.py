"""SQLite storage. One database file per deployment; every patient-scoped row carries patient_id."""
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from flask import g

from memora import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS caregivers (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, name TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata', home_description TEXT DEFAULT '',
    home_lat REAL, home_lon REAL, home_label TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patient_caregivers (
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    caregiver_id INTEGER NOT NULL REFERENCES caregivers(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, PRIMARY KEY (patient_id, caregiver_id));
CREATE TABLE IF NOT EXISTS link_codes (
    code TEXT PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    name TEXT NOT NULL, relationship TEXT DEFAULT '', voice_file TEXT, cloud_consent INTEGER NOT NULL DEFAULT 0,
    rekognition_face_ids TEXT DEFAULT '[]', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedder TEXT NOT NULL, embedding BLOB NOT NULL, quality TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recognition_events (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL, decision TEXT NOT NULL, provider TEXT NOT NULL,
    similarity REAL, probability REAL, detail TEXT, latency_ms REAL, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    type TEXT NOT NULL, title TEXT NOT NULL, time TEXT NOT NULL, days TEXT NOT NULL, notes TEXT DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reminder_log (
    id INTEGER PRIMARY KEY, reminder_id INTEGER NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    patient_id INTEGER NOT NULL, due_at TEXT NOT NULL, status TEXT NOT NULL, acted_at TEXT,
    UNIQUE (reminder_id, due_at));

CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    source TEXT NOT NULL, text TEXT DEFAULT '', scale INTEGER, emotion TEXT, scores TEXT, distress REAL,
    flagged INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assistant_log (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    text TEXT NOT NULL, intent TEXT, confidence REAL, slot TEXT, answer TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, severity TEXT NOT NULL, message TEXT NOT NULL, detail TEXT,
    created_at TEXT NOT NULL, resolved_at TEXT);

CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    name TEXT NOT NULL, text_query TEXT NOT NULL, text_embedding BLOB, kind TEXT NOT NULL DEFAULT 'item',
    notes TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS object_refs (
    id INTEGER PRIMARY KEY, object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    file TEXT NOT NULL, embedding BLOB, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY, object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    frame_file TEXT NOT NULL, crop_file TEXT NOT NULL, box TEXT NOT NULL, score REAL NOT NULL, method TEXT NOT NULL,
    description TEXT, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    title TEXT NOT NULL, caption TEXT DEFAULT '', photo_file TEXT, voice_file TEXT, added_by TEXT DEFAULT '',
    created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS safe_zones (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    name TEXT NOT NULL, polygon TEXT NOT NULL, aws_synced INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS location_reports (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    lat REAL NOT NULL, lon REAL NOT NULL, accuracy_m REAL, inside INTEGER, zone_id INTEGER, provider TEXT,
    detail TEXT, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS push_subscriptions (
    id INTEGER PRIMARY KEY, caregiver_id INTEGER NOT NULL REFERENCES caregivers(id) ON DELETE CASCADE,
    endpoint TEXT UNIQUE NOT NULL, keys TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage_counters (
    provider TEXT NOT NULL, month TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (provider, month));
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ago(minutes):
    """UTC timestamp `minutes` ago, in the same ISO format as now() so string comparison works."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def connect(path=None):
    conn = sqlite3.connect(str(path or config.DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path=None):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path or config.DB_PATH)
    if path.exists():
        conn = connect(path)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(patients)").fetchall()}
        conn.close()
        if cols and ("username" not in cols or "caregiver_id" in cols):  # before patient accounts / several relatives
            backup = path.with_suffix(".pre-accounts.bak")
            path.replace(backup)
            logging.getLogger(__name__).warning("old database schema: moved to %s and started a fresh one", backup)
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def get():
    if "db" not in g:
        g.db = connect()
    return g.db


def close(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def query(sql, args=(), one=False, conn=None):
    rows = (conn or get()).execute(sql, args).fetchall()
    rows = [dict(r) for r in rows]
    return (rows[0] if rows else None) if one else rows


def execute(sql, args=(), conn=None):
    conn = conn or get()
    cur = conn.execute(sql, args)
    conn.commit()
    return cur.lastrowid


def to_blob(vector):
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)


def loads(value, default=None):
    if value in (None, ""):
        return default
    return json.loads(value)


def bump_usage(provider, cap, conn=None):
    """Atomically count one paid API call for this month; returns False (and does not count) at the cap."""
    conn = conn or get()
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with conn:
        conn.execute("INSERT OR IGNORE INTO usage_counters (provider, month, count) VALUES (?, ?, 0)", (provider, month))
        cur = conn.execute(
            "UPDATE usage_counters SET count = count + 1 WHERE provider = ? AND month = ? AND count < ?",
            (provider, month, cap))
    return cur.rowcount == 1


def usage(provider, conn=None):
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    row = query("SELECT count FROM usage_counters WHERE provider = ? AND month = ?", (provider, month), one=True, conn=conn)
    return row["count"] if row else 0
