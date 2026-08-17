"""Minimal path-style S3 client for the project's `superfly` container.

Why this exists: the AirLab object store has no DNS for virtual-host-style
bucket subdomains, so the default `osmo data` CLI and any vhost-style S3
client fail. boto3 with addressing_style="path" against the base endpoint
works. Results/checkpoints must be mirrored here because OSMO container
scratch vanishes when a job exits.

Replaces the previous dependence on `gs_drone_sim.remote_store` (a different
repo) for superfly uploads.

Credential resolution order:
  1. env SUPERFLY_S3_KEY_ID / SUPERFLY_S3_KEY / SUPERFLY_S3_URL (inside OSMO
     jobs, injected via --set-env),
  2. env GSDS_S3_KEY_ID / GSDS_S3_KEY -- the same Keystone EC2 credential
     pair covers every bucket in the account, so until dedicated superfly
     keys are minted (~/.s3env) the gs-drone-sim ones work.

One-time bucket setup (needs the account's EC2 keys; see
INFRASTRUCTURE.md): `python -m superfly.remote_store create-bucket`.

Prefix convention inside s3://superfly: tmp_data/ (staging tarballs),
deps/ (per-project deps cache), runs/<tag>/ (results + checkpoints).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BUCKET = os.environ.get("SUPERFLY_S3_BUCKET", "superfly")
DEFAULT_URL = "https://airlab-cloud.andrew.cmu.edu:8080"


def _creds() -> dict:
    for prefix in ("SUPERFLY", "GSDS"):
        kid = os.environ.get(f"{prefix}_S3_KEY_ID")
        if kid:
            return {
                "access_key_id": kid,
                "access_key": os.environ[f"{prefix}_S3_KEY"],
                "override_url": os.environ.get(f"{prefix}_S3_URL", DEFAULT_URL),
                "region": os.environ.get(f"{prefix}_S3_REGION", "us-east-1"),
            }
    raise SystemExit(
        "no S3 credentials: set SUPERFLY_S3_KEY_ID/SUPERFLY_S3_KEY "
        "(or GSDS_* -- same account) e.g. via `source ~/.s3env`.")


def _client():
    import boto3
    from botocore.config import Config
    c = _creds()
    return boto3.client(
        "s3",
        aws_access_key_id=c["access_key_id"],
        aws_secret_access_key=c["access_key"],
        endpoint_url=c["override_url"],
        region_name=c["region"],
        config=Config(s3={"addressing_style": "path"}),
    )


def upload(local: str, key: str):
    s3 = _client()
    size = Path(local).stat().st_size
    print(f"upload {local} ({size / 1e6:.1f} MB) -> s3://{BUCKET}/{key}")
    s3.upload_file(str(local), BUCKET, key)


def download(key: str, local: str):
    s3 = _client()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    print(f"download s3://{BUCKET}/{key} -> {local}")
    s3.download_file(BUCKET, key, str(local))


def listing(prefix: str = ""):
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(f"{obj['Size']:>14}  {obj['Key']}")


def create_bucket():
    s3 = _client()
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if BUCKET in existing:
        print(f"bucket {BUCKET} already exists")
        return
    s3.create_bucket(Bucket=BUCKET)
    print(f"created bucket {BUCKET}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: python -m superfly.remote_store "
             "upload <local> <key> | download <key> <local> | "
             "list [prefix] | create-bucket")
    if not argv:
        raise SystemExit(usage)
    cmd, args = argv[0], argv[1:]
    if cmd == "upload" and len(args) == 2:
        upload(*args)
    elif cmd == "download" and len(args) == 2:
        download(*args)
    elif cmd == "list":
        listing(args[0] if args else "")
    elif cmd == "create-bucket":
        create_bucket()
    else:
        raise SystemExit(usage)


if __name__ == "__main__":
    main()
