"""HTML pages. Data is loaded by each page's JavaScript from the /api routes."""
from flask import Blueprint, g, redirect, render_template, request, send_from_directory, url_for

from memora import config
from memora.auth import caregiver_required, patient_only, patient_required

bp = Blueprint("pages", __name__)


@bp.route("/")
def home():
    """The entry: two doors, patient or caregiver. Signed-in users go straight to their own home."""
    if g.role == "patient":
        return redirect(url_for("pages.patient_home"))
    if g.role == "caregiver":
        return redirect(url_for("pages.dashboard") if g.patient else url_for("auth.patients_page"))
    return render_template("home.html")


@bp.route("/register")
@caregiver_required
@patient_required
def register():
    return render_template("register.html", face_provider=config.FACE_PROVIDER)


@bp.route("/recognize")
@patient_required
def recognize():
    return render_template("recognize.html")


@bp.route("/reminders")
@caregiver_required
@patient_required
def reminders():
    return render_template("reminders.html")


@bp.route("/checkins")
@caregiver_required
@patient_required
def checkins():
    return render_template("mood.html")


@bp.route("/mood")
def mood_redirect():
    return redirect(url_for("pages.checkins"))


@bp.route("/safety")
@caregiver_required
@patient_required
def safety():
    return render_template("sos.html", geofence_provider=config.GEOFENCE_PROVIDER,
                           location_interval_min=max(1, round(config.LOCATION_REPORT_INTERVAL_S / 60)))


@bp.route("/sos")
def sos_redirect():
    return redirect(url_for("pages.safety"))


@bp.route("/things")
@patient_required
def things():
    """Find my things: the patient's device is also the camera that keeps track of them.
    ?kind=medicine shows only medicines ("Find my medicines")."""
    kind = "medicine" if request.args.get("kind") == "medicine" else None
    return render_template("things.html", interval_s=config.CAMERA_INTERVAL_S, kind=kind)


@bp.route("/memories")
@caregiver_required
@patient_required
def memories():
    return render_template("memories.html")


@bp.route("/home-guide")
@patient_only
def home_guide():
    return render_template("home_guide.html", location_interval_s=config.LOCATION_REPORT_INTERVAL_S)


@bp.route("/patient/account")
@patient_only
def patient_account():
    return render_template("patient_account.html", link_minutes=config.LINK_CODE_MINUTES)


@bp.route("/objects")
def objects_redirect():
    return redirect(url_for("pages.things"))


@bp.route("/dashboard")
@caregiver_required
@patient_required
def dashboard():
    return render_template("dashboard.html")


@bp.route("/patient")
@patient_only
def patient_home():
    return render_template("patient.html", stt_provider=config.STT_PROVIDER,
                           location_interval_s=config.LOCATION_REPORT_INTERVAL_S)


@bp.route("/sw.js")
def service_worker():
    resp = send_from_directory(config.ROOT / "static", "sw.js", mimetype="application/javascript", max_age=0)
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@bp.route("/manifest.webmanifest")
def manifest():
    return send_from_directory(config.ROOT / "static", "manifest.webmanifest", mimetype="application/manifest+json")
