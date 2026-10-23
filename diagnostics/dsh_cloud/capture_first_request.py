#!/usr/bin/env python3
"""Capture one real DSH OpenAI request locally; always fail generation with 503.

No upstream client, tokenizer, model, GPU, or tool runner is imported or called.
The operator starts DSH separately using the shared single-trajectory overlay.
"""
import argparse
import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory; existing paths are refused.")
    parser.add_argument("--expected-task-file", type=Path, required=True)
    parser.add_argument("--model", default="kvmem-gc-qwen38")
    parser.add_argument("--port", type=int, default=18388)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-body-bytes", type=int, default=32 * 1024 * 1024)
    return parser.parse_args()


def write_exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def user_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    return ""


def main():
    args = parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0 or args.max_body_bytes <= 0:
        raise SystemExit("Invalid server bounds")
    expected = args.expected_task_file.read_text().strip()
    if not expected:
        raise SystemExit("Expected task must be nonempty")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    captured = False
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *unused):
            pass  # Never log headers, request bodies, or credentials.

        def reply(self, status, body):
            data = json_bytes(body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            self.close_connection = True

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"capture_only": True, "model_execution": False})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [
                    {"id": args.model, "object": "model", "owned_by": "capture-only"}
                ]})
            else:
                self.reply(404, {"error": {"message": "Unknown capture endpoint"}})

        def do_POST(self):
            nonlocal captured, request_count
            request_count += 1
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": {"message": "Expected chat completions"}})
                return
            # OpenAI's JSON fetch body has an explicit Content-Length. Refuse
            # unknown framing rather than silently capturing a partial request.
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= args.max_body_bytes:
                    raise ValueError("Missing or out-of-bounds Content-Length")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw)
                if not isinstance(body, dict) or body.get("model") != args.model:
                    raise ValueError("Unexpected model")
                messages = body.get("messages")
                if not isinstance(messages, list) or not all(
                        isinstance(message, dict) for message in messages):
                    raise ValueError("Expected complete OpenAI messages array")
                users = [message for message in messages if message.get("role") == "user"]
                if not any(user_text(message).strip() == expected for message in users):
                    raise ValueError("First request is not the expected main-agent task")
                tools = body.get("tools", [])
                if not isinstance(tools, list):
                    raise ValueError("Invalid tools array")
            except (ValueError, TypeError, TimeoutError) as error:
                self.reply(400, {"error": {"message": str(error),
                                         "type": "capture_validation_error"}})
                return
            write_exclusive(args.output / "first-request.body.json", raw)
            tool_names = [
                tool.get("function", {}).get("name") for tool in tools
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            ]
            metadata = {
                "capture_only": True,
                "model_execution": False,
                "model_tools_executed": False,
                "request_number": request_count,
                "path": self.path,
                "model": args.model,
                "body_bytes": len(raw),
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "message_count": len(messages),
                "roles": [message.get("role") for message in messages],
                "tool_count": len(tools),
                "tool_names": tool_names,
                "stream": body.get("stream", False),
                "max_tokens": body.get("max_tokens"),
                "max_completion_tokens": body.get("max_completion_tokens"),
                "response_status": 503,
                "tokenization_performed": False,
                "captured_at_unix": time.time(),
            }
            write_exclusive(args.output / "capture.json", json_bytes(metadata))
            captured = True
            self.reply(503, {"error": {
                "message": "Captured complete request; model execution intentionally unavailable.",
                "type": "capture_only",
                "code": "capture_complete",
            }})

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.timeout = 1
        write_exclusive(args.output / "ready.json", json_bytes({
            "pid": os.getpid(), "base_url": f"http://127.0.0.1:{args.port}/v1",
            "model": args.model, "capture_only": True,
        }))
        deadline = time.monotonic() + args.timeout
        while not captured and time.monotonic() < deadline:
            server.handle_request()
    if not captured:
        raise SystemExit("No complete matching request captured before timeout")
    print(json.dumps({"captured": True, "output": str(args.output),
                      "body_file": "first-request.body.json",
                      "request_count": request_count}))


if __name__ == "__main__":
    main()
