# server.py
import os
import time
from flask import Flask, Response, jsonify
from gunicorn.app.base import BaseApplication

app = Flask(__name__)
START_TIME = time.time()

@app.route("/")
def root():
    uptime = time.time() - START_TIME
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(uptime, 2),
        "uptime_human": f"{uptime/3600:.2f} hours",
    }), 200, headers

@app.route("/healthz")
def healthz():
    uptime = time.time() - START_TIME
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return Response(f"ok\nuptime={uptime:.0f}s", status=200, headers=headers, mimetype="text/plain")

class _GunicornApp(BaseApplication):
    def __init__(self, wsgi_app, options=None):
        self.options = options or {}
        self.application = wsgi_app
        super().__init__()

    def load_config(self):
        cfg = {k: v for k, v in self.options.items() if k in self.cfg.settings and v is not None}
        for k, v in cfg.items():
            self.cfg.set(k.lower(), v)

    def load(self):
        return self.application

def run_server():
    port = int(os.environ.get("PORT", "10000"))  # <-- bind to Render's port
    options = {
        "bind": f"0.0.0.0:{port}",
        "workers": 1,               # keep 1; your main thread is the Telegram bot
        "threads": 2,
        "timeout": 120,             # OCR/IO can be slow; safer than 30s
        "keepalive": 25,
        "loglevel": "info",
        # optional hardening:
        "max_requests": 1000,
        "max_requests_jitter": 100,
    }
    _GunicornApp(app, options).run()
