import time

from flask import Flask, Response, jsonify
from gunicorn.app.base import BaseApplication

app = Flask(__name__)

START_TIME = time.time()


@app.route("/")
def root():
    uptime = time.time() - START_TIME
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return (
        jsonify(
            {
                "status": "ok",
                "uptime_seconds": round(uptime, 2),
                "uptime_human": f"{uptime/3600:.2f} hours",
            }
        ),
        200,
        headers,
    )


@app.route("/healthz")
def healthz():
    uptime = time.time() - START_TIME
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(
        f"ok\nuptime={uptime:.0f}s",
        status=200,
        headers=headers,
        mimetype="text/plain",
    )

class _GunicornApp(BaseApplication):
    def __init__(self, wsgi_app, options=None):
        self.options = options or {}
        self.application = wsgi_app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run_server():
    options = {
        "bind": "0.0.0.0:10000",
        "workers": 1,
        "threads": 2,
        "timeout": 30,
        "loglevel": "info",
    }
    _GunicornApp(app, options).run()
