"""Local browser interface to the streaming analyzer. Python 3.10+, standard library."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import reliability


PAGE = Path(__file__).with_name("trace_dashboard.html")
MAX_UPLOAD = 256 * 1024 * 1024
MAX_LINE = 1024 * 1024


def records(stream, size):
    """Read exactly one request body, without buffering the whole upload."""
    remaining = size
    line_number = 0
    any_trace = False
    while remaining:
        line = stream.readline(min(remaining, MAX_LINE + 1))
        if not line:
            raise ValueError("Upload ended before its declared length.")
        remaining -= len(line)
        line_number += 1
        if len(line) > MAX_LINE:
            raise ValueError(f"Line {line_number} exceeds 1 MiB.")
        if not line.strip():
            continue
        try:
            trace = json.loads(line.decode("utf-8"))
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Line {line_number} must be a UTF-8 JSON object.") from exc
        if not isinstance(trace, dict) or type(trace.get("trace_id")) not in (str, int):
            raise ValueError(f"Line {line_number} needs a trace_id.")
        if not str(trace["trace_id"]).strip():
            raise ValueError(f"Line {line_number} has an empty trace_id.")
        if "user_goal" not in trace and "final_answer" not in trace:
            raise ValueError(f"Line {line_number} needs user_goal or final_answer. Use the downloadable sample as a template.")
        if not isinstance(trace.get("user_goal", ""), str) or not isinstance(trace.get("final_answer", ""), (str, type(None))):
            raise ValueError(f"Line {line_number} needs text for user_goal and text or null for final_answer.")
        spans = trace.get("spans", [])
        if not isinstance(spans, list) or any(not isinstance(span, dict) for span in spans):
            raise ValueError(f"Line {line_number} needs an array of span objects.")
        if not isinstance(trace.get("attrs", {}), dict):
            raise ValueError(f"Line {line_number} needs an object for attrs.")
        any_trace = True
        yield trace
    if not any_trace:
        raise ValueError("The upload contains no traces.")


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, Handler)
        self.analysis_slot = threading.BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, mime, data, attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, value):
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(value).encode("utf-8"))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif path == "/api/sample":
            self.send_json(200, reliability.analyze(reliability.demo(1000)))
        elif path == "/sample.jsonl":
            sample = "".join(json.dumps(t) + "\n" for t in reliability.demo(20))
            self.send_bytes(200, "application/x-ndjson", sample.encode("utf-8"), "sample.jsonl")
        else:
            self.send_json(404, {"error": "Page unavailable."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/analyze":
            return self.send_json(404, {"error": "Endpoint unavailable."})
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            return self.send_json(403, {"error": "Open the dashboard on the same origin."})
        if self.headers.get_content_type() not in ("application/x-ndjson", "application/octet-stream"):
            return self.send_json(415, {"error": "Send a JSONL file using application/x-ndjson."})
        try:
            size = int(self.headers["Content-Length"])
        except (ValueError, TypeError):
            return self.send_json(411, {"error": "A valid content length is required."})
        if size <= 0:
            return self.send_json(400, {"error": "Choose a nonempty JSONL file."})
        if size > MAX_UPLOAD:
            return self.send_json(413, {"error": "Upload limit is 256 MiB. Use reliability.py --input for larger files."})
        if not self.server.analysis_slot.acquire(blocking=False):
            return self.send_json(429, {"error": "Another analysis is running. Retry after it finishes."})
        try:
            self.connection.settimeout(60)
            # Uploaded data always uses the local rule engine, independently of API keys.
            report = reliability.analyze(records(self.rfile, size))
        except (ValueError, RecursionError) as exc:
            return self.send_json(400, {"error": str(exc) if isinstance(exc, ValueError) else "A record is too deeply nested."})
        except TimeoutError:
            return self.send_json(408, {"error": "The upload timed out."})
        finally:
            self.server.analysis_slot.release()
        self.send_json(200, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    with DashboardServer(("127.0.0.1", args.port)) as server:
        print(f"Open http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
