"""Fetch candidate source assets; never infer or certify numeric ground truth."""

import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def validate(data: bytes, filename: str) -> None:
    if filename.endswith(".xlsx"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "xl/workbook.xml" not in archive.namelist():
                raise ValueError("Not an XLSX workbook")
            if archive.testzip() is not None:
                raise ValueError("Corrupt workbook ZIP member")
    elif filename.endswith(".pdf") and not data.startswith(b"%PDF-"):
        raise ValueError("Not a PDF (possibly an HTML challenge page)")
    if not data:
        raise ValueError("Empty response")


def main() -> int:
    manifest = json.loads((ROOT / "research/candidates.json").read_text())
    records = []
    for candidate in manifest["candidates"]:
        for asset in candidate["assets"]:
            destination = ROOT / "sources" / candidate["id"] / asset["filename"]
            record = {"candidate": candidate["id"], **asset}
            try:
                cached = destination.exists()
                if cached:
                    data = destination.read_bytes()
                else:
                    request = urllib.request.Request(
                        asset["url"], headers={"User-Agent": "heatmap-decoder-source-review/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        data = response.read()
                        record["resolved_url"] = response.url
                validate(data, asset["filename"])
                if not cached:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                record.update(
                    status="local_file_validated" if cached else "downloaded",
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                )
            except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
                record.update(status="failed", error=str(error))
            records.append(record)
            print(f"{candidate['id']}/{asset['filename']}: {record['status']}", flush=True)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "File integrity only; table contents and figure order remain unverified.",
        "assets": records,
    }
    (ROOT / "research/download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    sys.exit(main())
