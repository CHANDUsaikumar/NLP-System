"""Web Server & API Backend for Dynamic NLP Model Router."""

import sys
import json
import http.server
import socketserver
from pathlib import Path

# Ensure workspace root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.router.router import DynamicRouter
from src.evaluation.model_evaluator import ModelEvaluator
from src.utils.logger import logger

PORT = 8000
UI_DIR = ROOT_DIR / "ui"

router = DynamicRouter()
evaluator = ModelEvaluator()


class RouterHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving static UI files and processing router API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/benchmark":
            try:
                results = evaluator.evaluate_all()
                serializable = [
                    {
                        "task": r.task,
                        "status": r.status,
                        "model": r.model,
                        "quality_metric_name": r.quality_metric_name,
                        "quality_score": r.quality_score,
                        "latency_ms": r.latency_ms
                    }
                    for r in results
                ]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(serializable).encode("utf-8"))
            except Exception as e:
                logger.error(f"Error executing benchmark API: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path in ["/api/chat", "/api/route"]:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
                prompt = payload.get("prompt", "")
                
                # Execute Dynamic Router
                res = router.process(prompt)
                
                response_data = {
                    "success": True,
                    "prompt": res.prompt,
                    "intent_detected": res.intent_detected,
                    "detected_task": res.detected_task,
                    "selected_model": res.selected_model,
                    "model_type": res.model_type,
                    "fallback_reason": res.fallback_reason,
                    "latency_ms": res.latency_ms,
                    "response_text": res.response_text
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                
            except Exception as e:
                logger.error(f"Error handling API chat request: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server():
    """Launches the Python HTTP Web Server."""
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RouterHTTPHandler) as httpd:
        print(f"\n🚀 Dynamic NLP Model Router Server running at http://localhost:{PORT}")
        print(f"📁 Serving UI from: {UI_DIR}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    start_server()
