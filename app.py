"""Memora entry point.

Local development:   python app.py          -> http://127.0.0.1:8000
WSGI servers:        gunicorn app:app
"""
from memora.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
