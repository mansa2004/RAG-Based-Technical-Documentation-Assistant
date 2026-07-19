"""
Standalone ingestion script.

Usage:
    python scripts/ingest.py                     # ingest the bundled corpus/ directory
    python scripts/ingest.py --dir path/to/docs   # ingest a different directory
    python scripts/ingest.py --urls https://a.com/doc.md https://b.com/doc.html
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest_corpus_dir, ingest_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--dir", type=str, default=None, help="Directory of .md/.txt/.html files")
    parser.add_argument("--urls", nargs="*", default=None, help="One or more URLs to fetch and ingest")
    args = parser.parse_args()

    if args.urls:
        n = ingest_paths(urls=args.urls)
        print(f"Ingested {n} chunks from {len(args.urls)} URL(s).")
    else:
        n = ingest_corpus_dir(args.dir)
        print(f"Ingested {n} chunks from directory: {args.dir or '(default corpus/)'}")


if __name__ == "__main__":
    main()
