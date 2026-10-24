"""Flask application factory."""
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from flask import Flask, g, jsonify, request
from markupsafe import Markup, escape
from werkzeug.exceptions import HTTPException

from memora import auth, config, db

log = logging.getLogger("memora")
RATE_LIMITED = {"auth.caregiver_login": (10, 600), "auth.patient_login": (10, 600), "auth.caregiver_signup": (5, 600),
                "auth.patient_signup": (5, 600), "auth.link_patient": (10, 600)}
_hits = defaultdict(deque)


def _rate_limit():
    rule = RATE_LIMITED.get(request.endpoint)
    if not rule or request.method != "POST":
        return None
    limit, window = rule
    # The hosting proxy appends the real client IP last; earlier entries are client-controlled.
    key = (request.endpoint, request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[-1].strip())
    hits, now = _hits[key], time.monotonic()
    while hits and hits[0] < now - window:
        hits.popleft()
    if len(hits) >= limit:
        return jsonify({"status": "fail", "message": "Too many attempts. Wait a few minutes and try again."}), 429
    hits.append(now)
    return None


def create_app(test_config=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()

    app = Flask(__name__, template_folder=str(config.ROOT / "templates"), static_folder=str(config.ROOT / "static"))
    app.config.update(
        SECRET_KEY=config.secret_key(),
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
        SESSION_COOKIE_HTTPONLY=True,
    )
    if config.COOKIE_MODE == "iframe":
        # The Hugging Face Space runs inside an iframe on huggingface.co: third-party cookie rules
        # require SameSite=None; Secure, and Partitioned (CHIPS) keeps it working when browsers block
        # unpartitioned third-party cookies.
        app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True, SESSION_COOKIE_PARTITIONED=True)
    else:
        app.config.update(SESSION_COOKIE_SAMESITE="Lax")
    if test_config:
        app.config.update(test_config)

    from memora import alerts, pages, storage
    from memora.routes import assistant, checkins, dashboard, faces, memories, objects, reminders, safety
    for bp in (auth.bp, pages.bp, storage.bp, alerts.bp, faces.bp, reminders.bp, checkins.bp,
               objects.bp, assistant.bp, safety.bp, dashboard.bp, memories.bp):
        app.register_blueprint(bp)

    @app.before_request
    def _before():
        limited = _rate_limit()
        if limited:
            return limited
        auth.load_user()
        if not app.config.get("TESTING_SKIP_CSRF"):
            auth.check_csrf()
        return None

    app.teardown_appcontext(db.close)

    def icon(name, cls=""):
        """A Lucide line icon from the local sprite (static/icons.svg, built by scripts/build_icons.py)."""
        return Markup(f'<svg class="i {escape(cls)}" aria-hidden="true"><use href="/static/icons.svg?v=1#{escape(name)}"></use></svg>')
    app.jinja_env.globals["icon"] = icon

    @app.context_processor
    def _inject():
        return {"csrf_token": auth.csrf_token(), "role": g.get("role"), "caregiver": g.get("caregiver"),
                "patient": g.get("patient"), "demo_notice": config.DEMO_NOTICE}

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Content-Security-Policy", "frame-ancestors 'self' https://huggingface.co")
        resp.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self), geolocation=(self)")
        return resp

    @app.errorhandler(HTTPException)
    def _http_error(e):
        if request.path.startswith("/api/") or request.path.startswith("/files/"):
            message = e.description if e.code != 413 else f"File too large (max {config.MAX_UPLOAD_MB} MB)."
            return jsonify({"status": "fail", "message": message}), e.code
        return e

    @app.errorhandler(Exception)
    def _server_error(e):
        log.exception("unhandled error on %s", request.path)
        if request.path.startswith("/api/"):
            return jsonify({"status": "fail", "message": "Something went wrong on the server."}), 500
        return "Something went wrong on the server.", 500

    if not app.config.get("TESTING"):
        from memora import scheduler
        scheduler.start()
        if config.GEOFENCE_PROVIDER == "aws":
            threading.Thread(target=_prune_stale_geofences, name="memora-geofence-prune", daemon=True).start()
        if config.FACE_PROVIDER in ("cascade", "rekognition"):
            threading.Thread(target=_prune_stale_collections, name="memora-rekognition-prune", daemon=True).start()
    return app


def _prune_stale_geofences():
    """A fresh database (e.g. after a Space restart) can't own geofences left in AWS by the previous one."""
    from memora import geo
    aws, why = geo.aws_or_none()
    if aws is None:
        return
    conn = db.connect()
    try:
        n = aws.prune_stale(conn)
        log.info("pruned %d stale AWS geofences for deployment %s", n, config.DEPLOYMENT_ID)
    except Exception as e:
        log.warning("geofence prune skipped: %s", e)
    finally:
        conn.close()


def _prune_stale_collections():
    """Same for Rekognition: face data of patients this database no longer has must not stay in AWS."""
    from memora.ml import face_providers
    rek, why = face_providers.rekognition_or_none()
    if rek is None:
        return
    conn = db.connect()
    try:
        n = rek.prune_stale(conn)
        log.info("pruned %d stale Rekognition collections for deployment %s", n, config.DEPLOYMENT_ID)
    except Exception as e:
        log.warning("Rekognition prune skipped: %s", e)
    finally:
        conn.close()
