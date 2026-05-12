import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gabeai_model import GabeAIEngine


APP_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


class GabeAIRequestHandler(SimpleHTTPRequestHandler):
    server_version = "gai/50"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def log_message(self, fmt, *args):
        sys.stdout.write("[gai] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        if self.path == "/api/health":
            self.send_json(self.server.engine.status())
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "Unknown endpoint")
            return

        try:
            payload = self.read_json()
            message = str(payload.get("message", "")).strip()
            use_search = bool(payload.get("useSearch", True))
            provider = str(payload.get("provider", "auto")).strip().lower() or "auto"
            top_k = int(payload.get("topK", 5))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.send_json({"error": "Bad request", "detail": str(exc)}, status=400)
            return

        if not message:
            self.send_json({"error": "Message is empty"}, status=400)
            return

        try:
            response = self.server.engine.answer(
                message,
                use_search=use_search,
                provider=provider,
                top_k=top_k,
            )
        except Exception as exc:
            response = {
                "answer": "GabeAI hit an error while answering. The server is still running.",
                "error": str(exc),
                "sources": [],
                "provider": provider,
            }
        self.send_json(response)

    def read_json(self):
        content_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_len)
        return json.loads(body.decode("utf-8"))

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "text/javascript"
        if path.endswith(".css"):
            return "text/css"
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            return "image/jpeg"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def run(host=DEFAULT_HOST, port=DEFAULT_PORT):
    os.chdir(APP_DIR)
    server = ThreadingHTTPServer((host, port), GabeAIRequestHandler)
    server.engine = GabeAIEngine(APP_DIR)
    print(f"gai is running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GabeAI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: python app.py [port]")
            raise SystemExit(2)
    run(port=port)
