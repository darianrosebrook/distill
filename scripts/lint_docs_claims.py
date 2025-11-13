#!/usr/bin/env python3
# scripts/lint_docs_claims.py
# Linter for documentation claims requiring evidence anchors
# @author: @darianrosebrook

import sys
import re
import pathlib
import argparse


BANNED = re.compile(
    r"(?i)\b(state[-\s]?of[-\s]?the[-\s]?art|revolutionary|breakthrough|best|leading|next[-\s]?generation)\b"
)
STATUS = re.compile(
    r"(?i)\b(production[-\s]?ready|enterprise[-\s]?grade|battle[-\s]?tested|deployed|released)\b"
)
BENCH = re.compile(r"(?i)\b(p50|p95|ttft|tps|throughput|latency|tokens\s?per\s?second)\b")
NUM = re.compile(r"(?i)-?\d+(?:\.\d+)?\s?(%|ms|s|tok/s|tokens/s|items)?")

# Evidence anchor syntax: [evidence: path.json#dotted.key]
ANCHOR = re.compile(r"\[evidence:\s*([^\]#\s]+)\s*#\s*([A-Za-z0-9_.]+)\s*\]")


def main():
    ap = argparse.ArgumentParser(
        description="Lint documentation for banned terms and missing evidence anchors"
    )
    ap.add_argument("--root", default="docs", help="Docs root to scan")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"Root directory {root} does not exist", file=sys.stderr)
        sys.exit(1)

    bad = []
    for p in root.rglob("*.*"):
        if p.suffix.lower() not in {".md", ".rst", ".txt", ".adoc"}:
            continue
        # Skip archive files, internal documentation, and external resources
        if any(skip in p.parts for skip in ["archive", "internal", "external"]):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}", file=sys.stderr)
            continue

        # Hard bans (skip code blocks and URLs)
        lines_list = text.splitlines()
        in_code_block_for_ban = False
        for line_idx, line in enumerate(lines_list):
            if line.strip().startswith("```"):
                in_code_block_for_ban = not in_code_block_for_ban
                continue
            if in_code_block_for_ban:
                continue
            # Skip URLs and external links (markdown link format)
            if "http" in line or "www." in line or ("](" in line and "http" in line):
                continue
            # Skip lines that are just link titles (markdown format)
            if line.strip().startswith("[") and "]" in line and "(" in line:
                continue
            # Skip image tags
            if line.strip().startswith("!["):
                continue
            # Skip lines that appear to be external link titles (standalone lines between image tags and URLs)
            # Check if this line is between an image tag and a URL (within 3 lines)
            nearby_start = max(0, line_idx - 3)
            nearby_end = min(len(lines_list), line_idx + 4)
            nearby_lines = lines_list[nearby_start:nearby_end]
            has_image_tag = any(
                nearby_line.strip().startswith("![") for nearby_line in nearby_lines
            )
            has_url = any(
                "http" in nearby_line or "www." in nearby_line for nearby_line in nearby_lines
            )
            # If line is between image tag and URL, it's likely an external link title
            if (
                has_image_tag
                and has_url
                and line.strip()
                and not line.strip().startswith(("!", "[", "]", "("))
            ):
                # Check if it's a short line (likely a title) between image and URL
                if len(line.strip()) < 100:  # Titles are typically shorter
                    continue
            for m in BANNED.finditer(line):
                bad.append((str(p), m.group(0), "banned_superlative"))
                break  # Only report once per line

        # Check if document has any evidence anchors (to skip overly strict checking)
        ANCHOR.search(text) is not None

        # Status/bench/quant must carry an evidence anchor on the same line
        # Skip code blocks and example/documentation lines
        in_code_block = False
        lines_with_anchors = set()

        # First pass: identify lines with anchors (including adjacent lines)
        for i, line in enumerate(text.splitlines(), start=1):
            if ANCHOR.search(line):
                # Mark this line and adjacent lines as having anchors
                lines_with_anchors.add(i)
                if i > 1:
                    lines_with_anchors.add(i - 1)
                if i < len(text.splitlines()):
                    lines_with_anchors.add(i + 1)

        # Second pass: check for violations
        for i, line in enumerate(text.splitlines(), start=1):
            # Skip lines that already have anchors nearby
            if i in lines_with_anchors:
                continue

            # Track code block boundaries
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            # Skip code blocks, comments, and example/documentation patterns
            if in_code_block:
                continue

            # Skip URLs and external links
            if "http" in line or "www." in line or ("](" in line and "http" in line):
                continue
            # Skip image tags
            if line.strip().startswith("!["):
                continue
            # Skip markdown link titles
            if line.strip().startswith("[") and "]" in line and "(" in line:
                continue

            # Skip lines that are clearly examples or documentation
            if any(
                skip in line.lower()
                for skip in [
                    "example:",
                    "for example",
                    "e.g.,",
                    "i.e.,",
                    "see:",
                    "note:",
                    "hint:",
                    "todo:",
                    "fixme:",
                    "```",
                    "<!--",
                    "date:",
                    "author:",
                    "created",
                    "for future reference",
                    "helps reason",
                    "record",
                    "log",
                    "script",
                    "implementation:",
                    "features:",
                ]
            ):
                continue

            # Skip lines that are just markdown formatting or lists
            stripped = line.strip()
            if stripped.startswith(("#", "-", "*", "|", "```", ">", "<!--")):
                continue

            # Skip numbered lists (e.g., "1. **Title**" or "1) Title")
            if re.match(r"^\d+[\.\)]\s+", stripped):
                continue

            # Skip technical implementation details (code snippets, file paths, etc.)
            if any(
                skip in line.lower()
                for skip in [
                    "created",
                    "for future reference",
                    "helps reason",
                    "record",
                    "log",
                    "script",
                    "implementation:",
                    "features:",
                    "wrapper",
                    "fingerprint",
                    "reproducibility",
                    "path update",
                    "diagnostics",
                    "memory pressure",
                ]
            ):
                continue

            # Check for status/bench/quant claims that need anchors
            # Only flag if it's a substantive claim, not just a number in a list
            has_status = STATUS.search(line)
            has_bench = BENCH.search(line)
            # For NUM, only flag if it's part of a claim (has units or is with status/bench terms)
            has_num_claim = False
            if NUM.search(line):
                # Only flag numbers that appear to be claims (with units, percentages, or alongside status/bench)
                num_match = NUM.search(line)
                if num_match:
                    # Check if number has units or is near status/benchmark terms
                    num_context = line[
                        max(0, num_match.start() - 20) : min(len(line), num_match.end() + 20)
                    ]
                    if any(
                        unit in num_context.lower()
                        for unit in ["%", "ms", "s", "tok/s", "tokens/s", "items"]
                    ):
                        # Only flag if it's clearly a performance claim, not just a technical detail
                        if STATUS.search(num_context) or BENCH.search(num_context):
                            has_num_claim = True

            if has_status or has_bench or has_num_claim:
                bad.append((f"{p}:{i}", line.strip()[:140], "missing_evidence_anchor"))

    if bad:
        print("Documentation claims linter failed. Offenses:")
        for loc, snippet, kind in bad:
            print(f" - {loc}: [{kind}] {snippet}")
        print(
            "\nHint: attach an anchor like: [evidence: eval/reports/latest.json#summary.gates_ok]"
        )
        sys.exit(1)

    print("Documentation claims linter passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
