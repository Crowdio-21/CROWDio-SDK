"""Publish an exported model (shards + manifest) to the Foreman catalogue.

REST protocol (Foreman HTTP port; see ``docs/client-protocol.md``):

* ``GET  /api/catalogue/{model_id}/versions`` -> ``{"versions": [n, ...]}`` (404: none)
* ``PUT  /api/catalogue/{model_id}/versions/{n}/files/{path}`` raw bytes with
  ``X-Content-SHA256``; returns ``{"path", "bytes", "sha256"}`` computed by the Foreman
* ``PUT  /api/catalogue/{model_id}/versions/{n}/manifest`` JSON; the Foreman
  checks every listed file and commits the version (409 if already committed)
* ``GET  /api/catalogue/{model_id}/manifest?version=n`` read back after commit

A version is immutable: re-publishing identical content is a no-op, and
changed content needs a new version number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .catalogue.manifest import ManifestError, ModelManifest, sha256_file, validate_manifest


class PublishError(RuntimeError):
    pass


class VersionConflict(PublishError):
    """The version already exists in the catalogue with different content."""


@dataclass
class PublishResult:
    model_id: str
    version: int
    status: str  # "published" or "unchanged"
    uploaded: List[Dict[str, Any]] = field(default_factory=list)


def content_digest(manifest: Union[ModelManifest, Dict[str, Any]]) -> str:
    """Digest of what a version *contains* (ignores ``created_at``)."""
    if not isinstance(manifest, ModelManifest):
        manifest = ModelManifest.from_dict(manifest)
    data = {k: v for k, v in manifest.to_dict().items() if k != "created_at"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Http:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any = None, headers: Optional[Dict[str, str]] = None,
                query: Optional[Dict[str, Any]] = None):
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
                return response.status, (json.loads(text) if text else None)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(text)
            except ValueError:
                detail = text
            return exc.code, detail


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _files_to_upload(manifest: ModelManifest, export_dir: Path) -> List[Dict[str, Any]]:
    files = []
    for shard in manifest.shards:
        files.append({"path": shard.file, "bytes": shard.bytes, "sha256": shard.sha256})
        for ext in shard.external_data_files:
            files.append({"path": ext.path, "bytes": ext.bytes, "sha256": ext.sha256})
    tokenizer_dir = manifest.source.get("tokenizer_dir")
    if tokenizer_dir and (export_dir / tokenizer_dir).is_dir():
        for path in sorted((export_dir / tokenizer_dir).rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(export_dir).as_posix(), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    return files


def publish_model(
    export_dir: Union[str, Path],
    foreman_url: str,
    version: Optional[int] = None,
    dry_run: bool = False,
    timeout: float = 600.0,
    log: Callable[[str], None] = print,
) -> PublishResult:
    """Upload a verified export and commit it as one catalogue version.

    ``version`` overrides the manifest's version (the export directory is not
    modified). Raises :class:`VersionConflict` if that version already exists
    with different content.
    """
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PublishError(f"{manifest_path} not found; run crowdio-export first (it writes the "
                           f"manifest only after verification passes)")
    report_path = export_dir / "verification_report.json"
    if not report_path.is_file() or not json.loads(report_path.read_text(encoding="utf-8")).get("passed"):
        raise PublishError(f"{report_path} is missing or did not pass; refusing to publish")

    manifest = ModelManifest.load(manifest_path)
    if version is not None:
        manifest.version = int(version)
    try:
        validate_manifest(manifest, export_dir, check_files=True)
    except ManifestError as exc:
        raise PublishError(f"local export does not match its manifest: {exc}") from None

    http = _Http(foreman_url, timeout)
    model = _quote(manifest.model_id)
    status, body = http.request("GET", f"/api/catalogue/{model}/versions")
    if status == 200:
        existing = [int(v) for v in (body or {}).get("versions", [])]
    elif status == 404:
        existing = []
    else:
        raise PublishError(f"catalogue unavailable at {foreman_url} (HTTP {status}): {body}")

    if manifest.version in existing:
        status, remote = http.request("GET", f"/api/catalogue/{model}/manifest", query={"version": manifest.version})
        if status != 200:
            raise PublishError(f"cannot read published {manifest.model_id} v{manifest.version} (HTTP {status})")
        if content_digest(remote) == content_digest(manifest):
            log(f"{manifest.model_id} v{manifest.version} is already published with identical content")
            return PublishResult(manifest.model_id, manifest.version, "unchanged")
        latest = max(existing)
        raise VersionConflict(
            f"{manifest.model_id} v{manifest.version} is already published with different content; "
            f"publish as a new version (e.g. --version {latest + 1}). Versions are never overwritten."
        )

    files = _files_to_upload(manifest, export_dir)
    total = sum(f["bytes"] for f in files)
    log(f"publishing {manifest.model_id} v{manifest.version}: {len(files)} files, {total / (1 << 20):.1f} MiB"
        + (" (dry run)" if dry_run else ""))
    if dry_run:
        return PublishResult(manifest.model_id, manifest.version, "dry_run", files)

    base = f"/api/catalogue/{model}/versions/{manifest.version}"
    uploaded = []
    for item in files:
        path = export_dir / item["path"]
        with open(path, "rb") as handle:
            status, body = http.request(
                "PUT", f"{base}/files/{urllib.parse.quote(item['path'])}", body=handle,
                headers={"Content-Type": "application/octet-stream",
                         "Content-Length": str(item["bytes"]), "X-Content-SHA256": item["sha256"]},
            )
        if status == 409:
            raise VersionConflict(f"{manifest.model_id} v{manifest.version} was committed by someone else "
                                  f"during upload: {body}")
        if status in (404, 405) and not uploaded:
            raise PublishError(f"{foreman_url} has no catalogue upload endpoint (HTTP {status}); "
                               f"the Foreman predates the shard catalogue")
        if status not in (200, 201):
            raise PublishError(f"upload of {item['path']} failed (HTTP {status}): {body}")
        if not isinstance(body, dict) or body.get("sha256") != item["sha256"] or body.get("bytes") != item["bytes"]:
            raise PublishError(f"checksum mismatch after uploading {item['path']}: local "
                               f"{item['sha256']} ({item['bytes']} B), Foreman reported {body}")
        uploaded.append(item)
        log(f"  uploaded {item['path']} ({item['bytes'] / (1 << 20):.1f} MiB, sha256 verified)")

    status, body = http.request("PUT", f"{base}/manifest", body=manifest.to_json().encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    if status == 409:
        raise VersionConflict(f"{manifest.model_id} v{manifest.version} already committed: {body}")
    if status not in (200, 201):
        raise PublishError(f"committing the manifest failed (HTTP {status}): {body}")

    status, remote = http.request("GET", f"/api/catalogue/{model}/manifest", query={"version": manifest.version})
    if status != 200 or content_digest(remote) != content_digest(manifest):
        raise PublishError(f"read-back of {manifest.model_id} v{manifest.version} does not match what was published")
    log(f"published {manifest.model_id} v{manifest.version}")
    return PublishResult(manifest.model_id, manifest.version, "published", uploaded)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="crowdio-publish", description="Publish an export to the Foreman catalogue")
    parser.add_argument("export_dir", help="Directory written by crowdio-export (<out>/<model-id>)")
    parser.add_argument("--foreman", required=True, help="Foreman HTTP base URL, e.g. http://192.168.1.7:8000")
    parser.add_argument("--version", type=int, help="Publish as this catalogue version")
    parser.add_argument("--dry-run", action="store_true", help="Check and list files without uploading")
    args = parser.parse_args(argv)
    try:
        publish_model(args.export_dir, args.foreman, version=args.version, dry_run=args.dry_run)
    except VersionConflict as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
