from flask import Flask
import threading

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

def run_server():
    app.run(host="0.0.0.0", port=10000)  # expose port 10000
