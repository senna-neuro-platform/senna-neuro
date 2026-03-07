from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


@dataclass(frozen=True)
class Config:
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_prefix: str
    outbox_dir: Path
    state_file: Path
    batch_epochs: int
    max_batch_files: int
    flush_interval_sec: int
    poll_interval_sec: int
    min_file_age_sec: int


@dataclass(frozen=True)
class PendingFile:
    path: Path
    relative_key: str
    signature: str
    epoch: int
    mtime: float


def load_config() -> Config:
    return Config(
        s3_endpoint=os.getenv("S3_ENDPOINT", "http://minio:9000"),
        s3_access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        s3_secret_key=os.getenv("S3_SECRET_KEY", "minioadmin123"),
        s3_bucket=os.getenv("S3_BUCKET", "senna-artifacts"),
        s3_prefix=os.getenv("S3_PREFIX", "senna-neuro").strip("/"),
        outbox_dir=Path(os.getenv("OUTBOX_DIR", "/artifacts/outbox")),
        state_file=Path(os.getenv("STATE_FILE", "/artifacts/uploader_state.json")),
        batch_epochs=int(os.getenv("UPLOAD_BATCH_EPOCHS", "5")),
        max_batch_files=int(os.getenv("UPLOAD_MAX_BATCH_FILES", "20")),
        flush_interval_sec=int(os.getenv("UPLOAD_FLUSH_INTERVAL_SEC", "45")),
        poll_interval_sec=int(os.getenv("UPLOAD_POLL_INTERVAL_SEC", "5")),
        min_file_age_sec=int(os.getenv("UPLOAD_MIN_FILE_AGE_SEC", "10")),
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_uploaded_signatures(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()

    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()

    signatures = payload.get("uploaded_signatures", [])
    if not isinstance(signatures, list):
        return set()

    return {str(item) for item in signatures}


def save_uploaded_signatures(state_file: Path, signatures: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = state_file.with_suffix(state_file.suffix + ".tmp")

    payload = {
        "uploaded_signatures": sorted(signatures),
        "updated_at": int(time.time()),
    }
    tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_file.replace(state_file)


def extract_epoch(filename: str) -> int:
    match = re.search(r"epoch[_-]?(\d+)", filename)
    if match is None:
        return 2**31 - 1
    return int(match.group(1))


def signature_for(path: Path, relative_key: str) -> str:
    stat = path.stat()
    return f"{relative_key}|{stat.st_size}|{int(stat.st_mtime_ns)}"


def collect_pending_files(
    config: Config, uploaded_signatures: set[str]
) -> list[PendingFile]:
    if not config.outbox_dir.exists():
        return []

    pending: list[PendingFile] = []
    for path in config.outbox_dir.rglob("*.h5"):
        if not path.is_file():
            continue

        relative_key = path.relative_to(config.outbox_dir).as_posix()
        signature = signature_for(path, relative_key)
        if signature in uploaded_signatures:
            continue

        pending.append(
            PendingFile(
                path=path,
                relative_key=relative_key,
                signature=signature,
                epoch=extract_epoch(path.name),
                mtime=path.stat().st_mtime,
            )
        )

    pending.sort(key=lambda item: (item.epoch, item.mtime, item.relative_key))
    return pending


def select_batch(config: Config, pending: list[PendingFile]) -> list[PendingFile]:
    if not pending:
        return []

    now = time.time()
    ready = [item for item in pending if (now - item.mtime) >= config.min_file_age_sec]
    if not ready:
        return []

    oldest_age = now - ready[0].mtime
    if len(ready) < config.batch_epochs and oldest_age < config.flush_interval_sec:
        return []

    return ready[: config.max_batch_files]


def object_key(config: Config, relative_key: str) -> str:
    if config.s3_prefix:
        return f"{config.s3_prefix}/{relative_key}"
    return relative_key


def ensure_bucket(client: object, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
        return
    except ClientError:
        pass

    client.create_bucket(Bucket=bucket)


def make_s3_client(config: Config) -> object:
    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint,
        aws_access_key_id=config.s3_access_key,
        aws_secret_access_key=config.s3_secret_key,
    )


def upload_batch(config: Config, batch: list[PendingFile]) -> set[str]:
    client = make_s3_client(config)
    ensure_bucket(client, config.s3_bucket)

    uploaded_signatures: set[str] = set()
    for item in batch:
        key = object_key(config, item.relative_key)
        client.upload_file(str(item.path), config.s3_bucket, key)
        uploaded_signatures.add(item.signature)
        logging.info("Uploaded %s -> s3://%s/%s", item.path, config.s3_bucket, key)

    return uploaded_signatures


def main() -> None:
    setup_logging()
    config = load_config()

    config.outbox_dir.mkdir(parents=True, exist_ok=True)
    uploaded_signatures = load_uploaded_signatures(config.state_file)

    logging.info(
        "Artifact uploader started: outbox=%s batch_epochs=%d max_batch=%d flush_interval=%ds",
        config.outbox_dir,
        config.batch_epochs,
        config.max_batch_files,
        config.flush_interval_sec,
    )

    while True:
        pending = collect_pending_files(config, uploaded_signatures)
        batch = select_batch(config, pending)

        if not batch:
            time.sleep(config.poll_interval_sec)
            continue

        try:
            uploaded_now = upload_batch(config, batch)
        except (BotoCoreError, ClientError, OSError) as error:
            logging.warning("Upload batch failed: %s", error)
            time.sleep(config.poll_interval_sec)
            continue

        uploaded_signatures.update(uploaded_now)
        save_uploaded_signatures(config.state_file, uploaded_signatures)


def run() -> None:
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Artifact uploader stopped")


if __name__ == "__main__":
    run()
