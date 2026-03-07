const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const path = require("path");

const host = "0.0.0.0";
const port = Number(process.env.PORT || 8080);
const tickIntervalMs = Number(process.env.VIS_TICK_INTERVAL_MS || 50);
const tickStride = Math.max(1, Number(process.env.VIS_TICK_STRIDE || 2));
const tracePollMs = Math.max(250, Number(process.env.VIS_TRACE_POLL_MS || 1000));
const tracePath = process.env.VISUALIZER_TRACE_PATH
  ? path.resolve(process.env.VISUALIZER_TRACE_PATH)
  : path.resolve(__dirname, "..", "data", "artifacts", "visualizer", "latest.json");

const contentTypeByExt = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

const websocketGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const clients = new Set();

let currentTrace = null;
let currentTraceVersion = "";
let frameCursor = 0;

function encodeTextFrame(text) {
  const payload = Buffer.from(text);
  const length = payload.length;

  if (length < 126) {
    const header = Buffer.alloc(2);
    header[0] = 0x81;
    header[1] = length;
    return Buffer.concat([header, payload]);
  }

  if (length < 65536) {
    const header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(length, 2);
    return Buffer.concat([header, payload]);
  }

  const header = Buffer.alloc(10);
  header[0] = 0x81;
  header[1] = 127;
  header.writeBigUInt64BE(BigInt(length), 2);
  return Buffer.concat([header, payload]);
}

function sendJson(socket, payload) {
  if (!socket || socket.destroyed) {
    return;
  }

  const frame = encodeTextFrame(JSON.stringify(payload));
  socket.write(frame);
}

function broadcastJson(payload) {
  const frame = encodeTextFrame(JSON.stringify(payload));

  for (const socket of clients) {
    if (socket.destroyed) {
      clients.delete(socket);
      continue;
    }

    try {
      socket.write(frame);
    } catch (error) {
      clients.delete(socket);
      socket.destroy();
    }
  }
}

function writeJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store, no-cache, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
  });
  res.end(body);
}

function normalizeNeuron(entry, outputZ) {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const id = Number(entry.id);
  const x = Number(entry.x);
  const y = Number(entry.y);
  const z = Number(entry.z);
  if (!Number.isInteger(id) || !Number.isInteger(x) || !Number.isInteger(y) || !Number.isInteger(z)) {
    return null;
  }

  let type = typeof entry.type === "string" ? entry.type : "excitatory";
  if (type === "excitatory" && z === outputZ) {
    type = "output";
  }
  if (!["excitatory", "inhibitory", "output"].includes(type)) {
    return null;
  }

  return { id, x, y, z, type };
}

function normalizeFrame(entry, totalNeurons) {
  if (!entry || typeof entry !== "object" || !Array.isArray(entry.neurons)) {
    return null;
  }

  const neurons = [];
  const seen = new Set();
  for (const neuron of entry.neurons) {
    if (!neuron || typeof neuron !== "object") {
      continue;
    }

    const id = Number(neuron.id);
    if (!Number.isInteger(id) || seen.has(id)) {
      continue;
    }

    seen.add(id);
    neurons.push({
      id,
      x: Number(neuron.x) || 0,
      y: Number(neuron.y) || 0,
      z: Number(neuron.z) || 0,
      type: typeof neuron.type === "string" ? neuron.type : "excitatory",
      fired: true,
    });
  }

  const tick = Number(entry.tick);
  return {
    tick: Number.isFinite(tick) ? tick : 0,
    neurons,
    activeCount: neurons.length,
    totalNeurons,
  };
}

function normalizeTrace(raw, stats) {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  if (!raw.lattice || typeof raw.lattice !== "object" || !Array.isArray(raw.frames)) {
    return null;
  }

  const width = Number(raw.lattice.width);
  const height = Number(raw.lattice.height);
  const depth = Number(raw.lattice.depth);
  if (!Number.isInteger(width) || !Number.isInteger(height) || !Number.isInteger(depth)) {
    return null;
  }

  const outputZ = Math.max(0, depth - 1);
  const neurons = [];
  for (const entry of raw.lattice.neurons || []) {
    const neuron = normalizeNeuron(entry, outputZ);
    if (neuron !== null) {
      neurons.push(neuron);
    }
  }
  if (neurons.length === 0) {
    return null;
  }

  const totalNeurons = neurons.length;
  const frames = [];
  for (const entry of raw.frames) {
    const frame = normalizeFrame(entry, totalNeurons);
    if (frame !== null) {
      frames.push(frame);
    }
  }
  if (frames.length === 0) {
    return null;
  }

  return {
    version: `${stats.mtimeMs}:${stats.size}`,
    traceSource: typeof raw.trace_source === "string" ? raw.trace_source : "unknown",
    datasetMode: typeof raw.dataset_mode === "string" ? raw.dataset_mode : "unknown",
    epoch: Number.isInteger(Number(raw.epoch)) ? Number(raw.epoch) : null,
    label: Number.isInteger(Number(raw.label)) ? Number(raw.label) : null,
    prediction: Number.isInteger(Number(raw.prediction)) ? Number(raw.prediction) : null,
    checkpointPath: typeof raw.checkpoint_path === "string" ? raw.checkpoint_path : null,
    lattice: {
      width,
      height,
      depth,
      neuronCount: totalNeurons,
      neurons,
    },
    frames,
  };
}

function loadTrace() {
  let stats;
  try {
    stats = fs.statSync(tracePath);
  } catch (error) {
    return null;
  }

  if (!stats.isFile()) {
    return null;
  }

  try {
    const payload = JSON.parse(fs.readFileSync(tracePath, "utf-8"));
    return normalizeTrace(payload, stats);
  } catch (error) {
    return null;
  }
}

function traceMetaPayload(type = "hello") {
  return {
    type,
    tickStride,
    tickIntervalMs,
    traceReady: currentTrace !== null,
    tracePath,
    neuronCount: currentTrace ? currentTrace.lattice.neuronCount : 0,
    frameCount: currentTrace ? currentTrace.frames.length : 0,
    epoch: currentTrace ? currentTrace.epoch : null,
    label: currentTrace ? currentTrace.label : null,
    prediction: currentTrace ? currentTrace.prediction : null,
    datasetMode: currentTrace ? currentTrace.datasetMode : null,
    traceSource: currentTrace ? currentTrace.traceSource : null,
  };
}

function refreshTrace() {
  const nextTrace = loadTrace();
  const nextVersion = nextTrace ? nextTrace.version : "";
  if (nextVersion === currentTraceVersion) {
    return;
  }

  currentTrace = nextTrace;
  currentTraceVersion = nextVersion;
  frameCursor = 0;

  if (clients.size > 0) {
    broadcastJson(traceMetaPayload("trace_meta"));
  }
}

function handleClientFrame(socket, buffer) {
  if (!buffer || buffer.length < 2) {
    return;
  }

  const opcode = buffer[0] & 0x0f;
  if (opcode === 0x8) {
    clients.delete(socket);
    socket.end();
    return;
  }

  if (opcode === 0x9) {
    socket.write(Buffer.from([0x8a, 0x00]));
  }
}

function serveStatic(req, res) {
  const reqUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const requestPath = reqUrl.pathname === "/" ? "/index.html" : reqUrl.pathname;

  if (requestPath === "/lattice") {
    refreshTrace();
    if (!currentTrace) {
      writeJson(res, 503, {
        error: "trace_not_ready",
        traceReady: false,
        tracePath,
      });
      return;
    }
    writeJson(res, 200, currentTrace.lattice);
    return;
  }

  if (requestPath === "/health") {
    refreshTrace();
    writeJson(res, 200, {
      status: currentTrace ? "ok" : "waiting_for_trace",
      traceReady: currentTrace !== null,
      tracePath,
      frameCount: currentTrace ? currentTrace.frames.length : 0,
      neuronCount: currentTrace ? currentTrace.lattice.neuronCount : 0,
    });
    return;
  }

  if (requestPath === "/trace") {
    refreshTrace();
    if (!currentTrace) {
      writeJson(res, 503, {
        error: "trace_not_ready",
        traceReady: false,
        tracePath,
      });
      return;
    }

    writeJson(res, 200, {
      type: "trace_snapshot",
      traceReady: true,
      ...currentTrace,
    });
    return;
  }

  if (requestPath === "/ws") {
    writeJson(res, 426, { error: "Upgrade Required" });
    return;
  }

  const safePath = path.normalize(decodeURIComponent(requestPath)).replace(/^\/+/, "");
  const filePath = path.join(__dirname, safePath);
  if (!filePath.startsWith(__dirname)) {
    writeJson(res, 403, { error: "Forbidden" });
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      writeJson(res, 404, { error: "Not found" });
      return;
    }

    const extension = path.extname(filePath);
    const contentType = contentTypeByExt[extension] || "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": contentType,
      "Content-Length": data.length,
      "Cache-Control": "no-store, no-cache, must-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    });
    res.end(data);
  });
}

const server = http.createServer(serveStatic);

server.on("upgrade", (req, socket) => {
  const reqUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  if (reqUrl.pathname !== "/ws") {
    socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
    socket.destroy();
    return;
  }

  const key = req.headers["sec-websocket-key"];
  if (!key) {
    socket.write("HTTP/1.1 400 Bad Request\r\n\r\n");
    socket.destroy();
    return;
  }

  const accept = crypto
    .createHash("sha1")
    .update(`${key}${websocketGuid}`, "binary")
    .digest("base64");

  const headers = [
    "HTTP/1.1 101 Switching Protocols",
    "Upgrade: websocket",
    "Connection: Upgrade",
    `Sec-WebSocket-Accept: ${accept}`,
  ];

  socket.write(`${headers.join("\r\n")}\r\n\r\n`);
  socket.setNoDelay(true);
  clients.add(socket);
  refreshTrace();
  sendJson(socket, traceMetaPayload("hello"));

  socket.on("data", (buffer) => {
    handleClientFrame(socket, buffer);
  });

  socket.on("end", () => {
    clients.delete(socket);
  });

  socket.on("close", () => {
    clients.delete(socket);
  });

  socket.on("error", () => {
    clients.delete(socket);
    socket.destroy();
  });
});

refreshTrace();
setInterval(refreshTrace, tracePollMs);

setInterval(() => {
  if (!currentTrace || currentTrace.frames.length === 0) {
    return;
  }

  const frame = currentTrace.frames[frameCursor];
  broadcastJson(frame);
  frameCursor = (frameCursor + tickStride) % currentTrace.frames.length;
}, tickIntervalMs);

server.listen(port, host, () => {
  console.log(`Visualizer running at http://${host}:${port} (tracePath=${tracePath})`);
});
