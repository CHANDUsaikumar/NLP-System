"""Web Server & API Backend for Dynamic Model-Routing Chatbot Application."""

import sys
import json
import http.server
import socketserver
from pathlib import Path

# Ensure workspace root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.routing_framework import RuleBasedRouter
from src.models.model_manager import ModelManager
from src.utils.logger import logger

PORT = 8000
WEB_DIR = ROOT_DIR / "web"

router = RuleBasedRouter()
model_manager = ModelManager()


class ChatbotHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web assets and routing API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_POST(self):
        if self.path in ["/api/route", "/api/chat"]:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
                prompt = payload.get("prompt", "")
                
                # Execute rule-based routing framework
                decision = router.route(prompt)
                
                # Run dynamic model inference if in full chat mode
                model_output = ""
                if self.path == "/api/chat":
                    try:
                        pipeline = model_manager.get_pipeline(decision.task_key)
                        out_text, lat_ms, tps = pipeline.run(prompt)
                        model_output = out_text
                    except Exception as err:
                        logger.warning(f"Live model execution fallback for '{decision.selected_model}': {err}")
                        model_output = f"[Simulated Output for {decision.selected_model}]: Processed request for '{decision.task_name}' successfully."

                response_data = {
                    "success": True,
                    "prompt": prompt,
                    "task_key": decision.task_key,
                    "task_name": decision.task_name,
                    "selected_model": decision.selected_model,
                    "fallback_model": decision.fallback_model,
                    "architecture_family": decision.architecture_family,
                    "confidence_score": decision.confidence_score,
                    "matched_rule_id": decision.matched_rule_id,
                    "matched_rule_description": decision.matched_rule_description,
                    "rationale": decision.rationale,
                    "metrics": decision.benchmark_metrics,
                    "model_output": model_output
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                
            except Exception as e:
                logger.error(f"Error handling API request: {e}")
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
    with socketserver.TCPServer(("", PORT), ChatbotHandler) as httpd:
        print(f"\n🚀 Dynamic Model-Routing Chatbot Server running at http://localhost:{PORT}")
        print(f"📁 Serving Web UI from: {WEB_DIR}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    start_server()
