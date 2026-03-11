const statusEl = document.getElementById("status");
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${protocol}://${window.location.hostname}:8080/ws`;

statusEl.textContent = `connecting to ${wsUrl}`;

try {
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    statusEl.textContent = "connected (stub)";
  };
  ws.onclose = () => {
    statusEl.textContent = "disconnected";
  };
  ws.onerror = () => {
    statusEl.textContent = "connection error (expected in stub)";
  };
} catch (err) {
  statusEl.textContent = `WebSocket failed: ${err}`;
}
