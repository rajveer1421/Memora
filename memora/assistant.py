"""Ask Memora: turn a classified question into an answer built only from the patient's own records,
and raise confusion / repeated-question / distress flags for the caregiver.

No language model writes the answers: every reply is a template filled from the database, so the
assistant can say "I don't know" but cannot make things up."""
import json
import re
from datetime import datetime, timedelta

from rapidfuzz import fuzz, process

from memora import config, db, storage
from memora.alerts import raise_flag
from memora.ml import distress, intent

HELP_TEXT = "You can ask me where your things are, about your medicines, what's on today, who visited, or who someone is."


def _ago(iso, now_utc):
    then = datetime.fromisoformat(iso)
    mins = int((now_utc - then).total_seconds() // 60)
    if mins < 2:
        return "just now"
    if mins < 60:
        return f"{mins} minutes ago"
    if mins < 24 * 60:
        h = mins // 60
        return f"{h} hour{'s' if h > 1 else ''} ago"
    d = mins // (24 * 60)
    return "yesterday" if d == 1 else f"{d} days ago"


def _clock(dt):
    return dt.strftime("%I:%M %p").lstrip("0")


def match_object(text, patient_id, conn=None):
    """Fuzzy-match a registered object name in the question ('my specs' ~ 'glasses' needs an alias)."""
    objs = db.query("SELECT id, name, text_query FROM objects WHERE patient_id = ?", (patient_id,), conn=conn)
    if not objs:
        return None, None
    choices = {}
    for o in objs:
        for alias in {o["name"].lower(), o["text_query"].lower()}:
            choices[alias] = o
            if alias.endswith("s"):
                choices[alias[:-1]] = o
    best = process.extractOne(text.lower(), list(choices), scorer=fuzz.partial_ratio)
    if best and best[1] >= 80:
        return choices[best[0]], best[1]
    m = re.search(r"\bmy ([a-z ]{2,20}?)(\?|$|\s(is|are|gone|went))", text.lower())
    return None, (m.group(1).strip() if m else None)


def match_person(text, patient_id, conn=None):
    people = db.query("SELECT * FROM people WHERE patient_id = ?", (patient_id,), conn=conn)
    t = text.lower()
    for p in people:
        if p["relationship"] and re.search(rf"\bmy {re.escape(p['relationship'].lower())}\b", t):
            return p
    names = {p["name"].lower(): p for p in people}
    names.update({p["name"].split()[0].lower(): p for p in people})
    if not names:
        return None
    best = process.extractOne(t, list(names), scorer=fuzz.partial_ratio)
    return names[best[0]] if best and best[1] >= 85 else None


def answer(patient, text, tz, now_utc, conn=None, forced_intent=None):
    """Return (intent result, answer dict). The answer has 'text' plus optional images/audio.
    forced_intent skips the classifier (the patient tapped a button such as "When is my next medicine?")."""
    pid = patient["id"]
    cls = {"intent": forced_intent, "confidence": 1.0, "model": "button"} if forced_intent else intent.classify(text)
    name = cls["intent"]
    now_local = now_utc.astimezone(tz)
    out = {"text": "", "images": [], "audio_url": None, "sos": False}

    if name == "where_is_object":
        obj, guess = match_object(text, pid, conn)
        if obj is None:
            what = f"your {guess}" if isinstance(guess, str) else "that"
            out["text"] = f"I'm not keeping track of {what} yet. You can add it in 'Find my things'."
        else:
            s = db.query("SELECT * FROM sightings WHERE object_id = ? ORDER BY id DESC LIMIT 1", (obj["id"],), one=True, conn=conn)
            if not s:
                out["text"] = f"I haven't seen your {obj['name']} yet. Open 'Find my things' and let the camera look around."
            else:
                where = f" {s['description'].rstrip('.')}" if s["description"] else ""
                out["text"] = f"I last saw your {obj['name']} {_ago(s['created_at'], now_utc)}.{(' It was' + where + '.') if where else ''} Here is the photo."
                out["images"] = [storage.url(pid, s["frame_file"]), storage.url(pid, s["crop_file"])]
    elif name == "medicine":
        rows = db.query("SELECT * FROM reminders WHERE patient_id = ? AND type = 'medication' AND active = 1", (pid,), conn=conn)
        today = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now_local.weekday()]
        todays = sorted((r for r in rows if today in json.loads(r["days"])), key=lambda r: r["time"])
        done = {(l["reminder_id"], l["due_at"][-5:]) for l in db.query(
            "SELECT reminder_id, due_at FROM reminder_log WHERE patient_id = ? AND status = 'done' AND due_at LIKE ?",
            (pid, now_local.strftime("%Y-%m-%d") + "%"), conn=conn)}
        if not todays:
            out["text"] = "You have no medicines scheduled today."
        else:
            parts = []
            taken = [r for r in todays if (r["id"], r["time"]) in done]
            pending = [r for r in todays if (r["id"], r["time"]) not in done]
            if taken:
                parts.append("Today you have already taken: " + ", ".join(f"{r['title']} ({r['time']})" for r in taken) + ".")
            due_now = [r for r in pending if r["time"] <= now_local.strftime("%H:%M")]
            later = [r for r in pending if r["time"] > now_local.strftime("%H:%M")]
            if due_now:
                parts.append("Not marked as taken yet: " + ", ".join(f"{r['title']} at {r['time']}" + (f" ({r['notes']})" if r["notes"] else "") for r in due_now) + ".")
            if later:
                r = later[0]
                parts.append(f"Your next medicine is {r['title']} at {r['time']}" + (f", {r['notes']}" if r["notes"] else "") + ".")
            out["text"] = " ".join(parts)
    elif name == "today_schedule":
        rows = db.query("SELECT * FROM reminders WHERE patient_id = ? AND active = 1 ORDER BY time", (pid,), conn=conn)
        today = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now_local.weekday()]
        later = [r for r in rows if today in json.loads(r["days"]) and r["time"] > now_local.strftime("%H:%M")]
        out["text"] = ("Still to come today: " + "; ".join(f"{r['time']} {r['title']}" for r in later) + ".") if later \
            else "There's nothing else planned for today."
    elif name == "who_visited":
        since = (now_local.replace(hour=0, minute=0, second=0, microsecond=0)).astimezone(now_utc.tzinfo).isoformat(timespec="seconds")
        rows = db.query("SELECT p.name, p.relationship, MAX(e.created_at) AS last FROM recognition_events e JOIN people p ON p.id = e.person_id "
                        "WHERE e.patient_id = ? AND e.decision = 'match' AND e.created_at >= ? GROUP BY p.id ORDER BY last DESC",
                        (pid, since), conn=conn)
        if not rows:
            out["text"] = "I haven't recognised any visitors today."
        else:
            out["text"] = "Today you saw " + ", ".join(
                f"{r['name']}" + (f" (your {r['relationship']})" if r["relationship"] else "") +
                f" at {_clock(datetime.fromisoformat(r['last']).astimezone(tz))}" for r in rows) + "."
    elif name == "who_is_person":
        p = match_person(text, pid, conn)
        if not p:
            out["text"] = "I'm not sure who you mean. You can point the camera at them and I'll tell you."
        else:
            out["text"] = f"{p['name']} is your {p['relationship']}." if p["relationship"] else f"{p['name']} is someone you know well."
            out["audio_url"] = storage.url(pid, p["voice_file"])
    elif name == "orientation_time":
        part = "morning" if now_local.hour < 12 else "afternoon" if now_local.hour < 17 else "evening" if now_local.hour < 21 else "night"
        out["text"] = f"Today is {now_local.strftime('%A, %d %B %Y').replace(' 0', ' ')}. It's {_clock(now_local)} in the {part}."
    elif name == "orientation_place":
        home = patient["home_description"] or "at home"
        loc = db.query("SELECT inside, created_at FROM location_reports WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (pid,), one=True, conn=conn)
        safe = " You are inside your safe area." if loc and loc["inside"] == 1 and loc["created_at"] >= db.ago(30) else ""
        out["text"] = f"You are {home}. You are safe, and your family knows where you are.{safe}"
    elif name == "need_help":
        out["sos"] = True
        out["text"] = "I've told your caregiver that you need help. Stay where you are, help is on the way."
    elif name == "feeling":
        out["text"] = "Thank you for telling me. Your family cares about you, and I've let them know how you're feeling."
    else:
        out["text"] = "I'm not sure about that. " + HELP_TEXT
    return cls, out


def signals(patient_id, text, cls, now_utc, conn=None, spoken=True):
    """Confusion, repeated questions and distress flags for the caregiver. Returns the flags raised.
    spoken=False (a button press) skips the distress model: there are no words of the patient's to read."""
    raised = []
    name = cls["intent"]
    if name in intent.ORIENTATION_INTENTS:
        if raise_flag(patient_id, "confusion", "info", f"Asked: \"{text}\"", {"intent": name}, conn=conn, dedupe_minutes=60):
            raised.append("confusion")
    if name not in ("out_of_scope", "feeling", "need_help"):
        since = (now_utc - timedelta(minutes=config.REPEAT_QUESTION_WINDOW_MIN)).isoformat(timespec="seconds")
        n = db.query("SELECT COUNT(*) AS n FROM assistant_log WHERE patient_id = ? AND intent = ? AND created_at >= ?",
                     (patient_id, name, since), one=True, conn=conn)["n"]
        if n >= config.REPEAT_QUESTION_COUNT:
            if raise_flag(patient_id, "repeat_question", "warning",
                          f"Asked about '{name.replace('_', ' ')}' {n} times in {config.REPEAT_QUESTION_WINDOW_MIN} minutes.",
                          {"intent": name, "count": n}, conn=conn, dedupe_minutes=60):
                raised.append("repeat_question")
    if name == "need_help":
        raise_flag(patient_id, "sos", "urgent", f"Asked Memora for help: \"{text}\"", {"source": "assistant"}, conn=conn)
        raised.append("sos")
    if not spoken:
        return raised, {"emotion": None}
    d = distress.analyze(text)
    db.execute("INSERT INTO checkins (patient_id, source, text, emotion, scores, distress, flagged, created_at) VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
               (patient_id, text, d["emotion"], json.dumps(d["scores"]), d["distress"], int(d["flag"]), db.now()), conn=conn)
    if d["flag"]:
        if raise_flag(patient_id, "distress", "warning", f"Possible {d['emotion']}: \"{text}\"", {"source": "assistant", "scores": d["scores"]},
                      conn=conn, dedupe_minutes=30):
            raised.append("distress")
    return raised, d
