"""Look inside the DeepSecure database from the terminal.

Usage (from the repo root):
    uv run python scripts/show_db.py                      # row counts + latest results + audit
    uv run python scripts/show_db.py --limit 50 --level MALICIOUS
    uv run python scripts/show_db.py --export reports/    # results.csv + audit_log.csv
    uv run python scripts/show_db.py --purge-days 30      # delete analyses older than 30 days

For a graphical view, open data/deepsecure.db in "DB Browser for SQLite"
(DOCUMENTATION §8.12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from deepsecure.config.settings import get_settings
from deepsecure.storage.repository import Repository


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--level", choices=["SAFE", "SUSPICIOUS", "MALICIOUS"], default=None)
    parser.add_argument("--export", type=Path, default=None, help="folder for CSV exports")
    parser.add_argument("--purge-days", type=int, default=None)
    args = parser.parse_args(argv)

    repo = Repository(s)
    print(f"[info] database: {repo.url}")
    if args.purge_days is not None:
        n = repo.purge_older_than(args.purge_days)
        repo.log_event("retention_purge", f"deleted {n} analyses older than {args.purge_days} days")
        print(f"[ok  ] deleted {n} analyses older than {args.purge_days} days")

    print("\nRows per table:")
    for table, n in repo.counts().items():
        print(f"  {table:20} {n}")
    results = repo.results_frame(limit=args.limit, risk_level=args.level)
    audit = repo.audit_frame(limit=args.limit)
    with pd.option_context("display.width", 160, "display.max_colwidth", 50):
        cols = ["created_at", "filename", "media_type", "risk_level", "risk_score", "repeat_count"]
        print("\nLatest results:" if len(results) else "\nNo results yet.")
        if len(results):
            print(results[cols].round(1).to_string(index=False))
        print("\nLatest audit events:" if len(audit) else "\nNo audit events yet.")
        if len(audit):
            print(audit[["timestamp", "event_type", "level", "message"]].to_string(index=False))

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        repo.results_frame(limit=1_000_000).to_csv(args.export / "results.csv", index=False)
        repo.audit_frame(limit=1_000_000).to_csv(args.export / "audit_log.csv", index=False)
        print(f"\n[ok  ] exported results.csv and audit_log.csv to {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
