"""Start the demo: python run.py  (Python 3.8+, nothing to install)."""
import http.server, os, socketserver, threading, webbrowser

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
for port in range(8000, 8011):
    try:
        server = socketserver.TCPServer(("127.0.0.1", port), Quiet)
        break
    except OSError:
        continue
else:
    raise SystemExit("No free port between 8000 and 8010.")

url = f"http://127.0.0.1:{port}/index.html"
print(f"Headway Recommender running at {url}  (Ctrl+C to stop)")
threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopped.")
