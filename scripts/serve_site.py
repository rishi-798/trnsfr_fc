"""Serve site/ locally, telling the browser to recheck files on every load.

Plain `python -m http.server` lets browsers reuse old copies of app.js, style.css and
players.json after they change, so updates only appear after a hard refresh. Here every
response carries Cache-Control: no-cache, so the browser asks first and gets a cheap
"not modified" reply when nothing changed.

Usage: python scripts/serve_site.py [port]   (default 8100)
"""
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    handler = functools.partial(NoCacheHandler, directory=str(SITE))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"serving {SITE} at http://localhost:{port} (ctrl+c to stop)")
        server.serve_forever()


if __name__ == "__main__":
    main()
