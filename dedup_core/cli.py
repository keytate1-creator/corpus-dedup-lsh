"""
Command-line interface for CorpusDedup engine.
"""

from __future__ import annotations

import sys
import json
import argparse
from typing import Sequence, Optional, List, Dict, Any

from dedup_core.pipeline import DeduplicationPipeline
from dedup_core.shingling import ShingleType


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus-dedup",
        description="High-Performance MinHash LSH Deduplication Engine for LLM Pretraining & SFT Corpora.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Deduplicate a JSON or JSONL document file.")
    run_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSON or JSONL file.",
    )
    run_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save deduplicated output file.",
    )
    run_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.75,
        help="Jaccard similarity threshold for deduplication (default: 0.75)",
    )
    run_parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128)",
    )
    run_parser.add_argument(
        "--bands",
        type=int,
        default=16,
        help="Number of LSH bands (default: 16)",
    )
    run_parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="Size k of shingles (default: 5)",
    )
    run_parser.add_argument(
        "--shingle-type",
        choices=["word", "char"],
        default="word",
        help="Shingle type (default: word)",
    )
    run_parser.add_argument(
        "--text-key",
        default="text",
        help="Key name containing document text (default: 'text')",
    )
    run_parser.add_argument(
        "--id-key",
        default="id",
        help="Key name containing document ID (default: 'id')",
    )

    return parser


def load_dataset(filepath: str, id_key: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if id_key not in item:
                    item[id_key] = f"doc_{idx}"
                docs.append(item)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if id_key not in item:
                        item[id_key] = f"doc_{idx}"
                    docs.append(item)
            else:
                raise ValueError("JSON file must contain an array of objects.")
    return docs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            documents = load_dataset(args.input, args.id_key)
        except Exception as e:
            sys.stderr.write(f"Error loading input dataset: {e}\n")
            return 2

        shingle_type = ShingleType.WORD if args.shingle_type == "word" else ShingleType.CHAR

        pipeline = DeduplicationPipeline(
            num_perm=args.num_perm,
            num_bands=args.bands,
            similarity_threshold=args.threshold,
            shingle_size=args.shingle_size,
            shingle_type=shingle_type,
        )

        result = pipeline.deduplicate(
            documents,
            text_key=args.text_key,
            id_key=args.id_key,
        )

        sys.stdout.write("==================================================\n")
        sys.stdout.write(" DEDUPLICATION METRICS\n")
        sys.stdout.write("==================================================\n")
        sys.stdout.write(f" Total Input Docs : {result.total_documents:,}\n")
        sys.stdout.write(f" Unique Retained  : {len(result.unique_documents):,} ({result.retention_rate:.2%})\n")
        sys.stdout.write(f" Duplicates Dropped: {result.duplicates_removed:,}\n")
        sys.stdout.write(f" Runtime Elapsed  : {result.runtime_sec:.3f} s\n")
        sys.stdout.write("==================================================\n")

        with open(args.output, "w", encoding="utf-8") as out_file:
            if args.output.endswith(".jsonl"):
                for doc in result.unique_documents:
                    out_file.write(json.dumps(doc) + "\n")
            else:
                json.dump(result.unique_documents, out_file, indent=2)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
