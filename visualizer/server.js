const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const path = require("path");

const host = "0.0.0.0";
const port = Number(process.env.PORT || 8080);
const tickIntervalMs = Number(process.env.VIS_TICK_INTERVAL_MS || 50);
const tickStride = Math.max(1, Number(process.env.VIS_TICK_STRIDE || 2));

const latticeConfig = {
    width: Number(process.env.VIS_WIDTH || 28),
    height: Number(process.env.VIS_HEIGHT || 28),
    depth: Number(process.env.VIS_DEPTH || 20),
    processingDensity: Number(process.env.VIS_PROCESSING_DENSITY || 0.7),
    excitatoryRatio: Number(process.env.VIS_EXCITATORY_RATIO || 0.8),
    outputNeurons: Number(process.env.VIS_OUTPUT_NEURONS || 10),
    maxActiveRatio: Number(process.env.VIS_MAX_ACTIVE_RATIO || 0.05),
    seed: Number(process.env.VIS_SEED || 42),
};

const contentTypeByExt = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
};

const websocketGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const clients = new Set();

function createRng(seed) {
    let state = seed >>> 0;
    return () => {
        state = ((state * 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function evenlyDistributedOutputPositions(width, height, count) {
    const layerSize = width * height;
    const positions = [];
    for (let i = 0; i < count; i += 1) {
        const linear = Math.floor(((2 * i + 1) * layerSize) / (2 * count));
        const x = linear % width;
        const y = Math.floor(linear / width);
        positions.push({ x, y });
    }
    return positions;
}

function buildLattice(config) {
    const rng = createRng(config.seed);
    const neurons = [];

    const pushNeuron = (x, y, z, type) => {
        neurons.push({
            id: neurons.length,
            x,
            y,
            z,
            type,
        });
    };

    for (let y = 0; y < config.height; y += 1) {
        for (let x = 0; x < config.width; x += 1) {
            pushNeuron(x, y, 0, "excitatory");
        }
    }

    for (let z = 1; z < config.depth - 1; z += 1) {
        for (let y = 0; y < config.height; y += 1) {
            for (let x = 0; x < config.width; x += 1) {
                if (rng() > config.processingDensity) {
                    continue;
                }
                const type = rng() <= config.excitatoryRatio ? "excitatory" : "inhibitory";
                pushNeuron(x, y, z, type);
            }
        }
    }

    const outputPositions = evenlyDistributedOutputPositions(
        config.width,
        config.height,
        config.outputNeurons,
    );
    for (const pos of outputPositions) {
        pushNeuron(pos.x, pos.y, config.depth - 1, "output");
    }

    return {
        width: config.width,
        height: config.height,
        depth: config.depth,
        neurons,
    };
}

const lattice = buildLattice(latticeConfig);
const maxActiveNeurons = Math.max(1, Math.floor(lattice.neurons.length * latticeConfig.maxActiveRatio));

const stimulusPath = [
    { x: Math.floor(lattice.width * 0.25), y: Math.floor(lattice.height * 0.25), z: 0 },
    { x: Math.floor(lattice.width * 0.75), y: Math.floor(lattice.height * 0.35), z: 0 },
    { x: Math.floor(lattice.width * 0.35), y: Math.floor(lattice.height * 0.75), z: 0 },
    { x: Math.floor(lattice.width * 0.7), y: Math.floor(lattice.height * 0.7), z: 0 },
];

function dist3(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

function computeActiveNeurons(tick) {
    const centerA = stimulusPath[Math.floor(tick / 40) % stimulusPath.length];
    const centerB = stimulusPath[(Math.floor(tick / 40) + 1) % stimulusPath.length];

    const waveSpan = Math.sqrt(
        (lattice.width * lattice.width) +
        (lattice.height * lattice.height) +
        (lattice.depth * lattice.depth),
    );

    const frontA = (tick * 0.28) % (waveSpan + 2.0);
    const frontB = ((tick * 0.23) + 3.5) % (waveSpan + 2.0);

    const active = [];

    for (const neuron of lattice.neurons) {
        const dA = dist3(neuron, centerA);
        const dB = dist3(neuron, centerB);

        const ridgeA = Math.abs(dA - frontA) < 0.85;
        const ridgeB = Math.abs(dB - frontB) < 0.65;

        if (!ridgeA && !(ridgeB && (neuron.id % 3 === 0))) {
            continue;
        }

        const noise = (((neuron.id * 2654435761) + (tick * 1013904223)) >>> 0) / 4294967296;
        if (noise < 0.58) {
            continue;
        }

        active.push({
            id: neuron.id,
            x: neuron.x,
            y: neuron.y,
            z: neuron.z,
            type: neuron.type,
            fired: true,
        });
    }

    if (active.length <= maxActiveNeurons) {
        return active;
    }

    const stride = Math.ceil(active.length / maxActiveNeurons);
    const trimmed = [];
    for (let i = 0; i < active.length; i += stride) {
        trimmed.push(active[i]);
        if (trimmed.length >= maxActiveNeurons) {
            break;
        }
    }
    return trimmed;
}

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

function writeJson(res, statusCode, payload) {
    const body = JSON.stringify(payload);
    res.writeHead(statusCode, {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": Buffer.byteLength(body),
    });
    res.end(body);
}

function serveStatic(req, res) {
    const reqUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const requestPath = reqUrl.pathname === "/" ? "/index.html" : reqUrl.pathname;

    if (requestPath === "/lattice") {
        writeJson(res, 200, {
            width: lattice.width,
            height: lattice.height,
            depth: lattice.depth,
            neuronCount: lattice.neurons.length,
            neurons: lattice.neurons,
        });
        return;
    }

    if (requestPath === "/health") {
        writeJson(res, 200, { status: "ok" });
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

    sendJson(socket, {
        type: "hello",
        tickStride,
        tickIntervalMs,
        neuronCount: lattice.neurons.length,
    });
});

let tick = 0;
setInterval(() => {
    tick += 1;
    if (tick % tickStride !== 0) {
        return;
    }

    const activeNeurons = computeActiveNeurons(tick);

    broadcastJson({
        tick,
        neurons: activeNeurons,
        activeCount: activeNeurons.length,
        totalNeurons: lattice.neurons.length,
    });
}, tickIntervalMs);

server.listen(port, host, () => {
    console.log(
        `Visualizer running at http://${host}:${port} (neurons=${lattice.neurons.length}, stride=${tickStride})`,
    );
});
