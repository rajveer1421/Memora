"""Create a demo caregiver linked to a demo patient account, with example reminders, a home and a safe zone,
all clearly labelled "(demo)". No faces or photos are created: add people with your own (consented) photos.

    python scripts/seed_demo.py --email demo@memora.app --password "choose-one" --patient-password "choose-another"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from werkzeug.security import generate_password_hash  # noqa: E402

from memora import db  # noqa: E402

REMINDERS = [
    ("medication", "Morning medicine (demo)", "08:00", "Donepezil 10 mg with breakfast"),
    ("meal", "Lunch (demo)", "13:00", "Drink a glass of water too"),
    ("activity", "Evening walk (demo)", "17:30", "Around the garden with Priya"),
    ("checkin", "Daily check-in (demo)", "18:00", ""),
    ("medication", "Evening medicine (demo)", "20:00", "Memantine 5 mg after dinner"),
]
ALL_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--patient-username", default="demo.patient")
    ap.add_argument("--patient-password", required=True)
    ap.add_argument("--lat", type=float, default=21.1458)
    ap.add_argument("--lon", type=float, default=79.0882)
    a = ap.parse_args()
    db.init_db()
    conn = db.connect()
    if db.query("SELECT 1 FROM caregivers WHERE email = ?", (a.email,), one=True, conn=conn) or \
            db.query("SELECT 1 FROM patients WHERE username = ?", (a.patient_username,), one=True, conn=conn):
        sys.exit("That demo account already exists.")
    cid = db.execute("INSERT INTO caregivers (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                     (a.email, "Demo caregiver", generate_password_hash(a.password), db.now()), conn=conn)
    pid = db.execute("INSERT INTO patients (username, password_hash, name, timezone, home_description, home_lat, home_lon, "
                     "home_label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (a.patient_username, generate_password_hash(a.patient_password), "Demo Patient", "Asia/Kolkata",
                      "at home (demo)", a.lat, a.lon, "Home (demo)", db.now()), conn=conn)
    db.execute("INSERT INTO patient_caregivers (patient_id, caregiver_id, created_at) VALUES (?, ?, ?)", (pid, cid, db.now()), conn=conn)
    for rtype, title, time_, notes in REMINDERS:
        db.execute("INSERT INTO reminders (patient_id, type, title, time, days, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (pid, rtype, title, time_, json.dumps(ALL_DAYS), notes, db.now()), conn=conn)
    d = 0.0008
    square = {"type": "Polygon", "coordinates": [[[a.lon - d, a.lat - d], [a.lon + d, a.lat - d], [a.lon + d, a.lat + d],
                                                  [a.lon - d, a.lat + d], [a.lon - d, a.lat - d]]]}
    db.execute("INSERT INTO safe_zones (patient_id, name, polygon, created_at) VALUES (?, ?, ?, ?)",
               (pid, "Home (demo)", json.dumps(square), db.now()), conn=conn)
    print(f"Demo caregiver {a.email} linked to patient @{a.patient_username}: {len(REMINDERS)} reminders, a home and a safe zone.")


if __name__ == "__main__":
    main()
