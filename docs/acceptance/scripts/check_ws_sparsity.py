#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class WsAddress:
    secure: bool
    host: str
    port: int
    path: str


def parse_ws_url(url: str) -> WsAddress:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(f"unsupported websocket scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("websocket URL is missing host")

    secure = parsed.scheme == "wss"
    port = parsed.port or (443 if secure else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    return WsAddress(
        secure=secure,
        host=parsed.hostname,
        port=port,
        path=path,
    )


def fetch_total_neurons(lattice_url: str, timeout_sec: float) -> int | None:
    try:
        with urlopen(lattice_url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    neuron_count = payload.get("neuronCount")
    if isinstance(neuron_count, int) and neuron_count > 0:
        return neuron_count

    neurons = payload.get("neurons")
    if isinstance(neurons, list):
        return len(neurons)
    return None


def read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("websocket closed while reading frame")
        data.extend(chunk)
    return bytes(data)


def encode_client_frame(opcode: int, payload: bytes = b"") -> bytes:
    fin_opcode = 0x80 | (opcode & 0x0F)
    payload_len = len(payload)
    mask_key = os.urandom(4)

    if payload_len < 126:
        header = bytearray([fin_opcode, 0x80 | payload_len])
    elif payload_len <= 0xFFFF:
        header = bytearray([fin_opcode, 0x80 | 126])
        header.extend(payload_len.to_bytes(2, "big"))
    else:
        header = bytearray([fin_opcode, 0x80 | 127])
        header.extend(payload_len.to_bytes(8, "big"))

    masked_payload = bytes(
        payload[index] ^ mask_key[index % 4] for index in range(payload_len)
    )
    return bytes(header) + mask_key + masked_payload


class WsClient:
    def __init__(self, url: str, timeout_sec: float) -> None:
        self.address = parse_ws_url(url)
        self.timeout_sec = timeout_sec
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        raw = socket.create_connection(
            (self.address.host, self.address.port),
            timeout=self.timeout_sec,
        )
        raw.settimeout(self.timeout_sec)

        if self.address.secure:
            context = ssl.create_default_context()
            sock = context.wrap_socket(raw, server_hostname=self.address.host)
        else:
            sock = raw

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.address.path} HTTP/1.1\r\n"
            f"Host: {self.address.host}:{self.address.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        sock.sendall(request)

        response = bytearray()
        while b"\r\n\r\n" not in response and len(response) < 16384:
            chunk = sock.recv(1024)
            if not chunk:
                break
            response.extend(chunk)

        header = response.decode("iso-8859-1", errors="replace")
        lines = header.split("\r\n")
        if not lines or "101" not in lines[0]:
            raise RuntimeError(
                f"websocket upgrade failed: {lines[0] if lines else header}"
            )

        accept_header = ""
        for line in lines[1:]:
            if line.lower().startswith("sec-websocket-accept:"):
                accept_header = line.split(":", 1)[1].strip()
                break

        expected_accept = base64.b64encode(
            hashlib.sha1(
                f"{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11".encode("ascii")
            ).digest()
        ).decode("ascii")
        if accept_header and accept_header != expected_accept:
            raise RuntimeError("websocket accept header mismatch")

        self.sock = sock

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.sendall(encode_client_frame(0x8))
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None

    def read_frame(self) -> tuple[int, bytes]:
        if self.sock is None:
            raise RuntimeError("websocket is not connected")

        header = read_exact(self.sock, 2)
        first, second = header[0], header[1]
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F

        if length == 126:
            length = int.from_bytes(read_exact(self.sock, 2), "big")
        elif length == 127:
            length = int.from_bytes(read_exact(self.sock, 8), "big")

        mask_key = read_exact(self.sock, 4) if masked else b""
        payload = read_exact(self.sock, length)

        if masked:
            payload = bytes(
                payload[index] ^ mask_key[index % 4] for index in range(len(payload))
            )

        return opcode, payload

    def send_pong(self, payload: bytes = b"") -> None:
        if self.sock is None:
            return
        self.sock.sendall(encode_client_frame(0xA, payload))


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate visualizer websocket active ratio (Step 16 DoD sparsity gate)."
    )
    parser.add_argument(
        "--ws-url",
        default="ws://localhost:8080/ws",
        help="Visualizer websocket URL",
    )
    parser.add_argument(
        "--lattice-url",
        default="http://localhost:8080/lattice",
        help="Visualizer lattice endpoint (fallback total neuron count)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=40,
        help="How many websocket frames to check",
    )
    parser.add_argument(
        "--max-ratio",
        "--max-active-ratio",
        dest="max_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed active ratio for each frame",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="Socket timeout in seconds",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        return fail("--frames must be > 0")
    if args.max_ratio <= 0.0:
        return fail("--max-ratio must be > 0")

    total_neurons_fallback = fetch_total_neurons(args.lattice_url, args.timeout_sec)
    if total_neurons_fallback is None:
        print(
            "[WARN] could not fetch total neurons from /lattice, will rely on WS payload"
        )

    client = WsClient(args.ws_url, args.timeout_sec)
    try:
        client.connect()
    except Exception as exc:
        return fail(f"cannot connect websocket: {exc}")

    checked = 0
    max_seen_ratio = 0.0
    max_seen_tick: int | None = None

    try:
        while checked < args.frames:
            try:
                opcode, payload = client.read_frame()
            except Exception as exc:
                return fail(f"cannot read websocket frame #{checked + 1}: {exc}")

            if opcode == 0x8:
                return fail("websocket closed before collecting required frame count")
            if opcode == 0x9:
                client.send_pong(payload)
                continue
            if opcode != 0x1:
                continue

            try:
                message: Any = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                return fail(f"invalid JSON frame #{checked + 1}: {exc}")

            if not isinstance(message, dict):
                return fail(f"frame #{checked + 1} payload is not JSON object")

            active_raw = message.get("activeCount")
            total_raw = message.get("totalNeurons")
            neurons_raw = message.get("neurons")

            if not isinstance(active_raw, int):
                if isinstance(neurons_raw, list):
                    active_raw = len(neurons_raw)
                else:
                    return fail(f"frame #{checked + 1} missing activeCount")

            if not isinstance(total_raw, int) or total_raw <= 0:
                total_raw = total_neurons_fallback
            if total_raw is None or total_raw <= 0:
                return fail(
                    f"frame #{checked + 1} has no totalNeurons and lattice fallback unavailable"
                )

            ratio = float(active_raw) / float(total_raw)
            tick_raw = message.get("tick")
            tick = int(tick_raw) if isinstance(tick_raw, int) else None

            if ratio > max_seen_ratio:
                max_seen_ratio = ratio
                max_seen_tick = tick

            checked += 1
            if ratio > args.max_ratio:
                tick_str = "unknown" if tick is None else str(tick)
                return fail(
                    f"frame #{checked} tick={tick_str}: active_ratio {ratio:.4f} > {args.max_ratio:.4f}"
                )
    finally:
        client.close()

    tick_str = "unknown" if max_seen_tick is None else str(max_seen_tick)
    print(
        "summary "
        f"frames={checked} "
        f"max_ratio={max_seen_ratio:.4f} "
        f"threshold={args.max_ratio:.4f} "
        f"max_ratio_tick={tick_str}"
    )
    print("[PASS] WebSocket sparsity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
