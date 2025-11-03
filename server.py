import os
import time
from flask import Flask, Response, jsonify

app = Flask(__name__)
START_TIME = time.time()

@app.route("/")
def root():
    uptime = time.time() - START_TIME
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(uptime, 2),
        "uptime_human": f"{uptime/3600:.2f} hours"
    }), 200, headers

@app.route("/healthz")
def healthz():
    uptime = time.time() - START_TIME
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return Response(f"ok\nuptime={uptime:.0f}s", status=200, headers=headers, mimetype="text/plain")

# No __main__ server needed when using Gunicorn on Render
