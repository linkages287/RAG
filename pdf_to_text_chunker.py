#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from PyPDF2 import PdfReader


# Extract raw text per page from a PDF file.
def iter_pdf_pages_text(pdf_path: str) -> Iterable[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        yield page_index, page_text


# Tokenize text into whitespace-separated tokens.
def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenization to avoid external models.
    return re.findall(r"\S+", text)


# Detect lines that are likely page numbers (arabic or roman numerals).
def is_page_number_line(line: str) -> bool:
    stripped = re.sub(r"[^\w]", "", line).lower()
    if not stripped:
        return False
    if stripped.isdigit():
        return True
    return stripped in {
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
        "xix",
        "xx",
    }


# Find repeated first/last lines across pages as header/footer candidates.
def extract_header_footer_candidates(pages: List[str]) -> Tuple[set, set]:
    header_counts = {}
    footer_counts = {}
    for page_text in pages:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        footer = lines[-1]
        header_counts[header] = header_counts.get(header, 0) + 1
        footer_counts[footer] = footer_counts.get(footer, 0) + 1

    headers = {line for line, count in header_counts.items() if count >= 2}
    footers = {line for line, count in footer_counts.items() if count >= 2}
    return headers, footers


SPLIT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}

SPLIT_SUFFIXES = {
    "al",
    "ed",
    "e",
    "er",
    "ers",
    "es",
    "ing",
    "ins",
    "ion",
    "ive",
    "ly",
    "nal",
    "ness",
    "ss",
}


# Merge common OCR split tokens while avoiding stopword joins.
def merge_ocr_splits(text: str) -> str:
    def repl(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in SPLIT_STOPWORDS or right.lower() in SPLIT_STOPWORDS:
            return f"{left} {right}"
        return f"{left}{right}"

    def repl_suffix(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in SPLIT_STOPWORDS:
            return f"{left} {right}"
        if right.lower() not in SPLIT_SUFFIXES:
            return f"{left} {right}"
        return f"{left}{right}"

    text = re.sub(r"\b([A-Za-z]{1,2})\s+([a-z]{2,})\b", repl, text)
    text = re.sub(r"\b([A-Za-z]{3,})\s+([a-z]{1,3})\b", repl_suffix, text)
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)
    text = re.sub(r"\b([A-Z]{1,3})\s+([A-Z]{2,})\b", repl, text)
    return text


# Regex patterns for coordinate detection (same as coords_finder.py)
DECIMAL_PAIR_RE = re.compile(
    r"(?P<lat>[+-]?\d{1,2}\.\d+)\s*[,;/]\s*(?P<lon>[+-]?\d{1,3}\.\d+)"
)

DMS_RE = re.compile(
    r"(?P<deg>\d{1,3})\s*[°]\s*"
    r"(?P<min>\d{1,2})\s*[′'\u2019]\s*"
    r"(?P<sec>\d{1,2}(?:\.\d+)?)?\s*[″\"\u201d]?\s*"
    r"(?P<hem>[NSEW])",
    re.IGNORECASE,
)


def normalize_decimal_pair(lat: float, lon: float) -> tuple[float, float] | None:
    """Validate and normalize lat/lon pair (same logic as coords_finder)."""
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    if -90.0 <= lon <= 90.0 and -180.0 <= lat <= 180.0:
        return lon, lat
    return None


def truncate_decimal(num: float) -> str:
    """Truncate a decimal number to 3 decimal places, removing trailing zeros."""
    truncated = f"{num:.3f}"
    if "." in truncated:
        truncated = truncated.rstrip("0").rstrip(".")
    return truncated


# Truncate coordinate-like decimal numbers to 3 decimal places using coords_finder methodology.
def truncate_coordinates(text: str) -> str:
    """
    Find and truncate coordinates using the same detection methodology as coords_finder.py.
    Handles decimal pairs (e.g., "12.345678, -1.234567") and validates lat/lon ranges.
    """
    # Process decimal pairs
    def truncate_decimal_pair(match: re.Match) -> str:
        lat_str = match.group("lat")
        lon_str = match.group("lon")
        full_match = match.group(0)
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            normalized = normalize_decimal_pair(lat, lon)
            if normalized:
                # Valid coordinate pair - truncate both
                lat_trunc = truncate_decimal(normalized[0])
                lon_trunc = truncate_decimal(normalized[1])
                # Extract separator from original match (comma, semicolon, or slash)
                sep_match = re.search(r"\s*([,;/])\s*", full_match)
                sep = sep_match.group(1) if sep_match else ","
                return f"{lat_trunc}{sep} {lon_trunc}"
        except (ValueError, AttributeError):
            pass
        return full_match
    
    # Process DMS coordinates - truncate seconds to 3 decimal places if present
    def truncate_dms_sec(match: re.Match) -> str:
        sec_str = match.group("sec")
        if sec_str and "." in sec_str:
            try:
                sec = float(sec_str)
                sec_trunc = truncate_decimal(sec)
                # Reconstruct the DMS string with truncated seconds, preserving original format
                deg = match.group("deg")
                min_val = match.group("min")
                hem = match.group("hem")
                # Preserve original quote style if present
                original = match.group(0)
                quote_char = "\"" if "\"" in original else "″" if "″" in original else "\""
                sec_part = f"{sec_trunc}{quote_char}"
                return f"{deg}°{min_val}'{sec_part}{hem}"
            except (ValueError, AttributeError):
                pass
        return match.group(0)
    
    # Apply truncation to decimal pairs
    text = DECIMAL_PAIR_RE.sub(truncate_decimal_pair, text)
    
    # Apply truncation to DMS seconds
    text = DMS_RE.sub(truncate_dms_sec, text)
    
    return text


# Clean page text for RAG: remove headers/footers and normalize spacing.
def clean_page_text(text: str, headers: set, footers: set) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    for line in lines:
        if not line:
            continue
        if line in headers or line in footers or is_page_number_line(line):
            continue
        if re.search(r"\.{2,}\s*\d+\b", line):
            continue
        if re.search(r"\bTABLE OF CONTENTS?\b", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    cleaned = re.sub(r"\bNATO\s+UNCLASSIFIED\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", cleaned)
    cleaned = re.sub(r"[•]", " ", cleaned)
    cleaned = re.sub(r"\s+([,;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=\w)", r"\1 ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\.\s*(\.)+\s*", ". ", cleaned)
    cleaned = re.sub(r"\b([Aa])\s+nd\b", "and", cleaned)
    cleaned = re.sub(r"\b([Aa])\s+nnually\b", "annually", cleaned)
    cleaned = merge_ocr_splits(cleaned)
    cleaned = re.sub(r"\bt\s+o([a-z]{2,})\b", r"to \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([A-Z]{2,})([a-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b([A-Z]{2,})\s+s\b", r"\1s", cleaned)
    cleaned = re.sub(r"([a-z])(?=etc\.)", r"\1 ", cleaned, flags=re.IGNORECASE)
    cleaned = truncate_coordinates(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Chunk token stream with overlap and track page ranges.
def chunk_tokens(
    tokens_with_pages: List[Tuple[str, int]], max_tokens: int, overlap_ratio: float = 0.17
) -> List[dict]:
    chunks = []
    overlap_tokens = max(1, int(max_tokens * overlap_ratio))
    step = max(1, max_tokens - overlap_tokens)
    for i in range(0, len(tokens_with_pages), step):
        chunk_items = tokens_with_pages[i : i + max_tokens]
        if not chunk_items:
            continue
        chunk_tokens_list = [token for token, _page in chunk_items]
        page_numbers = [page for _token, page in chunk_items]
        chunk_text = " ".join(chunk_tokens_list)
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "text": chunk_text,
                "token_count": len(chunk_tokens_list),
                "page_start": min(page_numbers),
                "page_end": max(page_numbers),
            }
        )
    return chunks


# Build cleaned, tokenized, overlapping chunks for a single PDF.
def build_chunks_from_pdf(pdf_path: str, max_tokens: int) -> List[dict]:
    tokens_with_pages: List[Tuple[str, int]] = []
    pages = [text for _page_index, text in iter_pdf_pages_text(pdf_path)]
    headers, footers = extract_header_footer_candidates(pages)
    total_pages = len(pages)
    print(f"\n--- Processing file: {pdf_path} ({total_pages} pages) ---")
    for page_index, page_text in enumerate(pages, start=1):
        percent = int((page_index / max(total_pages, 1)) * 100)
        print(f"\n[Status] {pdf_path} page {page_index}/{total_pages} ({percent}%)")
        cleaned_text = clean_page_text(page_text, headers, footers)
        page_tokens = tokenize(cleaned_text)
        tokens_with_pages.extend((token, page_index) for token in page_tokens)
    return chunk_tokens(tokens_with_pages, max_tokens=max_tokens, overlap_ratio=0.15)


# Load existing collective JSON or initialize a new payload.
def load_or_init_json(json_path: Path) -> dict:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"sources": [], "chunks": []}


# CLI entry point: process one or more PDFs and append into a single JSON.
def collect_pdf_paths(inputs: List[str]) -> List[Path]:
    pdfs: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        else:
            pdfs.append(path)
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs, visualize, chunk into tokens, save JSON."
    )
    parser.add_argument("output_json", help="Path to output JSON file.")
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="One or more input PDF files and/or directories.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="Maximum tokens per chunk (default: 250).",
    )
    args = parser.parse_args()

    json_path = Path(args.output_json)
    payload = load_or_init_json(json_path)
    payload.setdefault("sources", [])
    payload.setdefault("chunks", [])

    next_chunk_id = (
        max((c.get("chunk_id", 0) for c in payload["chunks"]), default=0) + 1
    )
    total_new = 0
    pdf_files = collect_pdf_paths(args.pdf_paths)
    if not pdf_files:
        raise SystemExit("No PDF files found in the provided inputs.")
    for pdf_file in pdf_files:
        chunks = build_chunks_from_pdf(str(pdf_file), max_tokens=args.max_tokens)
        for chunk in chunks:
            chunk["chunk_id"] = next_chunk_id
            chunk["source_pdf"] = pdf_file.name
            next_chunk_id += 1
            payload["chunks"].append(chunk)
        payload["sources"].append(pdf_file.name)
        total_new += len(chunks)

    payload["max_tokens"] = args.max_tokens
    payload["chunk_count"] = len(payload["chunks"])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nAdded {total_new} chunks. Saved {payload['chunk_count']} total to {json_path}")


if __name__ == "__main__":
    main()
