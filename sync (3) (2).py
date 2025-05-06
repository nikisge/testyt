#!/usr/bin/env python3
"""
Sync Airtable → OpenAI embeddings → Qdrant
Run:  python sync_embeddings.py

Changes from original version:
• Caches Airtable rows in a CSV file (path configurable via env `AIRTABLE_CACHE_CSV`).
• If the CSV already exists, loads records from it instead of querying the Airtable API.
• When fresh data are fetched, the CSV cache is (re)written.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

import requests                    # noqa: E402  (after load_dotenv)
from openai import OpenAI          # noqa: E402
from pyairtable import Table       # noqa: E402

# ───────────────────────────────────────────
# Logging
# ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ───────────────────────────────────────────
# Environment – abort early if anything is missing
# ───────────────────────────────────────────
REQUIRED_VARS = (
    "AIRTABLE_API_KEY",
    "AIRTABLE_BASE_ID",
    "AIRTABLE_TABLE_ID",      # table *name* or "tbl…"
    "OPENAI_API_KEY",
    "QDRANT_URL",             # e.g. https://xxxxxxxx.aws.cloud.qdrant.io:6333
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
)
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    log.critical("Missing environment variables: %s", ", ".join(missing))
    sys.exit(1)

# read once → shorter names
AIRTABLE_API_KEY   = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID   = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID  = os.getenv("AIRTABLE_TABLE_ID")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
QDRANT_URL         = os.getenv("QDRANT_URL").rstrip("/")           # safety
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION")
# optional
CSV_CACHE_PATH     = os.getenv("AIRTABLE_CACHE_CSV", "airtable_records_cache.csv")

# ───────────────────────────────────────────
# Clients
# ───────────────────────────────────────────
table  = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)
openai = OpenAI(api_key=OPENAI_API_KEY)

# ───────────────────────────────────────────
# CSV helpers
# ───────────────────────────────────────────

def _save_records_to_csv(records: List[Dict], path: str) -> None:
    """Persist Airtable records in a flat CSV (id + all field keys)."""
    if not records:
        return

    # Collect headers (id + union of all field names)
    headers: set[str] = {"id"}
    for rec in records:
        headers.update(rec["fields"].keys())

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers))
        writer.writeheader()
        for rec in records:
            row = {"id": rec["id"]}
            row.update(rec["fields"])
            writer.writerow(row)
    log.info("Wrote %s records → %s", len(records), path)


def _load_records_from_csv(path: str) -> List[Dict]:
    """Rehydrate records from the CSV cache into the Airtable record format."""
    if not os.path.isfile(path):
        return []

    records: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid = row.pop("id", None)
            # Remove empty strings so that payload stays clean
            fields = {k: v for k, v in row.items() if v}
            records.append({"id": rid, "fields": fields})
    log.info("Loaded %s records from cache %s", len(records), path)
    return records

# ───────────────────────────────────────────
# Airtable sync
# ───────────────────────────────────────────

def fetch_all_records(force_refresh: bool = False) -> List[Dict]:
    """Return Airtable records – pulls from cache unless refresh requested."""
    if not force_refresh and os.path.isfile(CSV_CACHE_PATH):
        return _load_records_from_csv(CSV_CACHE_PATH)

    # Otherwise hit Airtable
    all_records: List[Dict] = []
    for page in table.iterate(page_size=100, view="Grid view"):
        all_records.extend(page)
        time.sleep(0.22)               # ≤ 5 req/s per Airtable policy

    log.info("Fetched %s Airtable rows via API", len(all_records))

    # Persist fresh copy for next run
    try:
        _save_records_to_csv(all_records, CSV_CACHE_PATH)
    except Exception as exc:
        log.warning("Failed to write CSV cache (%s): %s", CSV_CACHE_PATH, exc)

    return all_records

# ───────────────────────────────────────────
# Embedding + Qdrant
# ───────────────────────────────────────────

def embed_and_upsert(records: List[Dict], batch_size: int = 50) -> None:
    """Build embeddings with OpenAI, then push to Qdrant in chunks."""
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]

        # 1) build input strings
        texts = [
            (
                f"{rec['fields'].get('Img_Name', '')} "
                f"{rec['fields'].get('Description', '')}"
            )[:4000]                     # OpenAI limit ≈ 8k tokens → safe side
            for rec in batch
        ]

        # 2) OpenAI embeddings
        try:
            embed_resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            vectors = [e.embedding for e in embed_resp.data]
        except Exception as exc:
            log.error("OpenAI error on batch %s: %s", start // batch_size, exc)
            continue

        # 3) build Qdrant payload
        now_iso = datetime.utcnow().isoformat()

        points = [
            {
                "id": start + idx + 1,      #  <<< hier: fortlaufende ID
                "vector": vectors[idx],
                "payload": {
                    "bildtitel":   rec["fields"].get("Img_Name"),
                    "beschreibung": rec["fields"].get("Description"),
                    "url":         rec["fields"].get("Direct_Link"),
                    "pfad":        rec["fields"].get("Full_Path"),
                    "sync_time":   now_iso,
                },
            }
            for idx, rec in enumerate(batch)
        ]

        # 4) upsert
        try:
            r = requests.put(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
                headers={
                    "Content-Type": "application/json",
                    "api-key": QDRANT_API_KEY,
                },
                json={"points": points},
                timeout=30,
            )
            r.raise_for_status()
            log.info(
                "Uploaded batch %s → Qdrant (%s vectors)",
                start // batch_size + 1,
                len(points),
            )
        except Exception as exc:
            log.error("Qdrant error on batch %s: %s", start // batch_size, exc)
            continue

        time.sleep(0.5)                 # gentle on Qdrant

# ───────────────────────────────────────────
# Main
# ───────────────────────────────────────────

def main() -> None:
    # Pass --refresh to force download even when cache exists
    force_refresh = "--refresh" in sys.argv[1:]

    records = fetch_all_records(force_refresh=force_refresh)
    if not records:
        log.warning("Nothing to process – exiting.")
        return

    embed_and_upsert(records)
    log.info("Done.")


if __name__ == "__main__":
    main()