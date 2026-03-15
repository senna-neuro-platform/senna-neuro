import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { decode as cborDecode } from "https://cdn.jsdelivr.net/npm/cbor-x@1.5.4/+esm";

const statusEl = document.getElementById("status");
const timeEl = document.getElementById("time");
const firedEl = document.getElementById("fired");
const toastEl = document.getElementById("toast");
const container = document.getElementById("canvas-container");

let scene, camera, renderer, controls, mesh;
let baseColors, flashStrength;
let neuronCount = 0;
let synLines = null;
let activityCounts = [];
let perClassMaps = Array.from({ length: 10 }, () => null);
let neuronPositions = null;
let waveStartMs = performance.now();
const waveSpeed = 12; // units per second along +Z
const waveSigma = 1.2;
let waveOriginZ = 0;

const showSyn = document.getElementById("showSyn");
const filterE = document.getElementById("filterE");
const filterI = document.getElementById("filterI");
const filterO = document.getElementById("filterO");
const heatmapToggle = document.getElementById("heatmap");
const alphaSlider = document.getElementById("alpha");
const btnPause = document.getElementById("btnPause");
const btnResume = document.getElementById("btnResume");
const tickInterval = document.getElementById("tickInterval");
const classSelect = document.getElementById("classSelect");
const btnCapture = document.getElementById("btnCapture");
const btnShowInterf = document.getElementById("btnShowInterf");
const autoInterf = document.getElementById("autoInterf");
const lblCurrent = document.getElementById("lblCurrent");

const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${protocol}://${window.location.hostname}:8080`;
statusEl.textContent = `connecting ${wsUrl}`;

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 2000);
}

function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#0b1021");

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    200
  );
  camera.position.set(40, 40, 60);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(14, 14, 10);
  controls.update();

  const ambient = new THREE.AmbientLight(0x7dd3fc, 0.4);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(30, 40, 20);
  scene.add(ambient, dir);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const grid = new THREE.GridHelper(60, 30, 0x23304f, 0x1b2742);
  grid.position.set(13.5, -2, 10);
  scene.add(grid);
}

function buildNeurons(neurons) {
  neuronCount = neurons.length;
  const geometry = new THREE.SphereGeometry(0.35, 12, 12);
  const material = new THREE.MeshPhongMaterial({
    vertexColors: true,
    shininess: 30,
  });

  mesh = new THREE.InstancedMesh(geometry, material, neuronCount);
  baseColors = [];
  flashStrength = new Float32Array(neuronCount);
  activityCounts = new Float32Array(neuronCount);
  perClassMaps = Array.from({ length: 10 }, () => new Float32Array(neuronCount));
  neuronPositions = neurons;
  waveOriginZ = Math.min(...neurons.map((n) => n.z));
  waveStartMs = performance.now();

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  neurons.forEach((n, idx) => {
    dummy.position.set(n.x, n.y, n.z);
    dummy.updateMatrix();
    mesh.setMatrixAt(idx, dummy.matrix);

    const c =
      n.type === "E" ? color.set("#60a5fa") : n.type === "I" ? color.set("#f97316") : color.set("#22c55e");
    mesh.setColorAt(idx, c);
    baseColors.push(c.clone());
  });
  mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
}

function animate() {
  requestAnimationFrame(animate);
  if (mesh) {
    const color = new THREE.Color();
    const showE = filterE.checked;
    const showI = filterI.checked;
    const showO = filterO.checked;
    const now = performance.now();
    const waveAge = (now - waveStartMs) / 1000; // seconds
    const waveFront = waveAge * waveSpeed;
    for (let i = 0; i < neuronCount; ++i) {
      const base = baseColors[i].clone();
      const type = window.__neuronPositions?.[i]?.type || "E";
      if ((type === "E" && !showE) || (type === "I" && !showI) || (type === "O" && !showO)) {
        mesh.setColorAt(i, base.setScalar(0.02));
        continue;
      }
      // wave from sensory panel (lowest Z)
      if (neuronPositions) {
        const z = neuronPositions[i].z;
        const dist = Math.abs(z - waveOriginZ);
        const band = dist - waveFront;
        const wave = Math.exp(-0.5 * (band * band) / (waveSigma * waveSigma));
        if (wave > 0.05) {
          base.lerp(new THREE.Color("#7dd3fc"), Math.min(0.7, wave));
        }
      }
      if (flashStrength[i] > 0) {
        const c = base.clone().lerp(new THREE.Color("#e2e8f0"), flashStrength[i]);
        mesh.setColorAt(i, c);
        flashStrength[i] = Math.max(0, flashStrength[i] - 0.02);
      } else if (heatmapToggle.checked && activityCounts[i] > 0) {
        const norm = Math.min(1, activityCounts[i] / 50);
        const heat = new THREE.Color().setHSL(0.33 - norm * 0.33, 1, 0.5);
        mesh.setColorAt(i, base.clone().lerp(heat, 0.7));
      } else if (btnShowInterf.dataset.active === "1") {
        const cls = parseInt(classSelect.value, 10);
        const map = perClassMaps[cls];
        if (map) {
          const norm = Math.min(1, map[i] / 50);
          const heat = new THREE.Color().setHSL(0.66 - norm * 0.66, 1, 0.45);
          mesh.setColorAt(i, base.clone().lerp(heat, 0.8));
        }
      } else {
        const alpha = parseFloat(alphaSlider.value);
        base.multiplyScalar(alpha + 0.4);
        mesh.setColorAt(i, base);
      }
    }
    mesh.instanceColor.needsUpdate = true;
  }
  renderer.render(scene, camera);
}

function handleMessage(evt) {
  if (!(evt.data instanceof ArrayBuffer)) {
    return;
  }
  const view = new DataView(evt.data);
  const type = view.getUint8(0);
  const len =
    (view.getUint8(1) << 24) |
    (view.getUint8(2) << 16) |
    (view.getUint8(3) << 8) |
    view.getUint8(4);
  const compressed = new Uint8Array(evt.data, 5, len);
  const msg = cborDecode(pako.inflate(compressed));
  if (msg.type === "NetworkState") {
    statusEl.textContent = `network: ${msg.neuron_count} neurons`;
    buildNeurons(msg.neurons);
    showToast("Network loaded");
    if (synLines) {
      scene.remove(synLines);
      synLines.geometry.dispose();
      synLines = null;
    }
    // cache neuron positions for synapses
    window.__neuronPositions = msg.neurons;
  } else if (msg.type === "SynapseChunk") {
    if (!showSyn.checked) return;
    const positions = window.__neuronPositions;
    if (!positions) return;
    if (!synLines) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(msg.total * 2 * 3), 3)
      );
      const mat = new THREE.LineBasicMaterial({
        color: 0x1f2937,
        transparent: true,
        opacity: 0.15,
      });
      synLines = new THREE.LineSegments(geo, mat);
      scene.add(synLines);
    }
    const posArr = synLines.geometry.attributes.position.array;
    msg.synapses.forEach((s, idx) => {
      const i = (msg.start + idx) * 6;
      const pre = positions[s.pre];
      const post = positions[s.post];
      if (!pre || !post) return;
      posArr[i] = pre.x;
      posArr[i + 1] = pre.y;
      posArr[i + 2] = pre.z;
      posArr[i + 3] = post.x;
      posArr[i + 4] = post.y;
      posArr[i + 5] = post.z;
    });
    synLines.geometry.attributes.position.needsUpdate = true;
  } else if (msg.type === "TickUpdate") {
    waveStartMs = performance.now(); // restart wavefront each update
    timeEl.textContent = `${msg.t_ms.toFixed(1)} ms`;
    if (typeof msg.label === "number") {
      lblCurrent.textContent = msg.label;
    }
    firedEl.textContent = msg.fired.length;
    msg.fired.forEach((n) => {
      const id = typeof n === "object" ? n.id : n;
      if (id < flashStrength.length) {
        flashStrength[id] = 1.0;
        activityCounts[id] += 1;
      }
    });
    if (autoInterf.checked && typeof msg.label === "number" && msg.label >= 0 && msg.label < 10) {
      const cls = msg.label;
      const map = perClassMaps[cls];
      for (let i = 0; i < neuronCount; ++i) {
        map[i] += activityCounts[i];
      }
      // reset for next accumulation window
      activityCounts.fill(0);
    }
  }
}

function connect() {
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    statusEl.textContent = "connected";
    showToast("Connected");
    ws.send(JSON.stringify({ type: "SetUpdateInterval", ticks: parseInt(tickInterval.value, 10) }));
  };
  ws.onclose = () => {
    statusEl.textContent = "disconnected";
    showToast("Disconnected");
    setTimeout(connect, 2000);
  };
  ws.onerror = () => {
    statusEl.textContent = "error";
  };
  ws.onmessage = handleMessage;
  btnPause.onclick = () => ws.send(JSON.stringify({ type: "Pause" }));
  btnResume.onclick = () => ws.send(JSON.stringify({ type: "Resume" }));
  tickInterval.oninput = () =>
    ws.send(JSON.stringify({ type: "SetUpdateInterval", ticks: parseInt(tickInterval.value, 10) }));
  autoInterf.onchange = () => {
    if (autoInterf.checked) {
      ws.send(JSON.stringify({ type: "SetLabel", label: parseInt(classSelect.value, 10) }));
    }
  };
  classSelect.onchange = () => {
    if (autoInterf.checked) {
      ws.send(JSON.stringify({ type: "SetLabel", label: parseInt(classSelect.value, 10) }));
    }
  };

  btnCapture.onclick = () => {
    const cls = parseInt(classSelect.value, 10);
    perClassMaps[cls].set(activityCounts);
    showToast(`Captured interference map for class ${cls}`);
  };
  btnShowInterf.onclick = () => {
    const active = btnShowInterf.dataset.active === "1";
    btnShowInterf.dataset.active = active ? "0" : "1";
    btnShowInterf.textContent = active ? "Show Interf" : "Hide Interf";
  };
}

initThree();
animate();
connect();
