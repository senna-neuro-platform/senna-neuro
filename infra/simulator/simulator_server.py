from http.server import BaseHTTPRequestHandler, HTTPServer
import math
import time

HOST = "0.0.0.0"
PORT = 8000


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            now = time.time()
            active_ratio = 0.03 + 0.01 * math.sin(now / 5.0)
            spikes_per_tick = 120.0 + 20.0 * math.sin(now / 7.0)
            synapse_count = 300000
            payload = "\n".join(
                [
                    "# HELP senna_active_neurons_ratio Ratio of active neurons.",
                    "# TYPE senna_active_neurons_ratio gauge",
                    f"senna_active_neurons_ratio {active_ratio:.6f}",
                    "# HELP senna_spikes_per_tick Average spikes per simulation tick.",
                    "# TYPE senna_spikes_per_tick gauge",
                    f"senna_spikes_per_tick {spikes_per_tick:.6f}",
                    "# HELP senna_synapse_count Number of active synapses.",
                    "# TYPE senna_synapse_count gauge",
                    f"senna_synapse_count {synapse_count}",
                    "",
                ]
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/health":
            payload = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), MetricsHandler)
    print(f"Simulator metrics stub running on http://{HOST}:{PORT}")
    server.serve_forever()
