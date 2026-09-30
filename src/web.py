"""Small local web chat for the Personal Movie Vault."""

from __future__ import annotations

import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402


CONTEXTUAL_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|they|them|there|those|these|he|she|his|her|their)\b"
    r"|\b(what about (the )?(director|cast|ending|scene|rating|soundtrack|cinema|location|format|story|character|visuals|acting)"
    r"|and what|why is that|how about the|tell me more|say more|"
    r"which one|what else|why did it|how did it)\b",
    re.IGNORECASE,
)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 32_000:
                self.send_error(413)
                return
            payload = json.loads(self.rfile.read(length))
            question = str(payload.get("message", "")).strip()
            if not question:
                self.send_error(400, "Message cannot be empty")
                return
            history = payload.get("history", [])
            # Standalone questions begin a fresh retrieval topic. Carry only the
            # immediately preceding exchange when the wording depends on it.
            follow_up = bool(CONTEXTUAL_FOLLOW_UP.search(question))
            recent_history = [item for item in history[-2:] if isinstance(item, dict)] if follow_up else []
            history_text = "\n".join(
                f"{item.get('role', 'user')}: {str(item.get('content', ''))[:1000]}"
                for item in recent_history
            )
            previous_user_question = next(
                (str(item.get("content", ""))[:500] for item in reversed(recent_history)
                 if item.get("role") == "user"),
                "",
            )
            retrieval_query = (
                f"Previous question: {previous_user_question}\nFollow-up: {question}"
                if previous_user_question else question
            )
            results = rag.search(retrieval_query, 5, "semantic", 0.5, rag.INDEX)
            answer = rag.answer_from_results(question, results, "qwen2.5:3b", history_text)
            body = json.dumps({
                "answer": answer,
                "sources": [{"title": item["title"], "section": item["section"], "source": item["source"]}
                            for item in results],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Movie Vault chat: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
