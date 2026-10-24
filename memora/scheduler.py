"""Background sweep (one daemon thread per process): flags missed reminders every minute so the caregiver
is notified even when nobody has the app open."""
import logging
import threading

from memora import db

log = logging.getLogger(__name__)
_started = False


def _loop(stop, interval):
    from memora.routes.reminders import sweep_all
    while not stop.wait(interval):
        conn = db.connect()
        try:
            sweep_all(conn)
        except Exception:
            log.exception("reminder sweep failed")
        finally:
            conn.close()


def start(interval=60):
    global _started
    if _started:
        return None
    _started = True
    stop = threading.Event()
    threading.Thread(target=_loop, args=(stop, interval), name="memora-sweep", daemon=True).start()
    return stop
