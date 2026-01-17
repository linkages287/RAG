#!/usr/bin/env python3
import argparse
import json
import re
from typing import Iterable, List, Tuple

from PyPDF2 import PdfReader


def iter_pdf_pages_text(pdf_path: str) -> Iterable[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        yield page_index, page_text


def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenization to avoid external models.
    return re.findall(r"\S+", text)


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


def merge_ocr_splits(text: str) -> str:
    def repl(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in SPLIT_STOPWORDS:
            return f"{left} {right}"
        return f"{left}{right}"

    text = re.sub(r"\b([A-Za-z]{1,2})\s+([a-z]{2,})\b", repl, text)
    text = re.sub(r"\b([A-Za-z]{3,})\s+([a-z]{2,3})\b", repl, text)
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)
    text = re.sub(r"\b([A-Z]{1,3})\s+([A-Z]{2,})\b", r"\1\2", text)
    return text


def clean_page_text(text: str, headers: set, footers: set) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    for line in lines:
        if not line:
            continue
        if line in headers or line in footers or is_page_number_line(line):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    cleaned = re.sub(r"\bNATO\s+UNCLASSIFIED\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", cleaned)
    cleaned = re.sub(r"[•]", " ", cleaned)
    cleaned = re.sub(r"\s*\.{2,}\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\s+([,;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=\w)", r"\1 ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\.\s*(\.)+\s*", ". ", cleaned)
    cleaned = re.sub(r"\b([Aa])\s+nd\b", "and", cleaned)
    cleaned = re.sub(r"\b([Aa])\s+nnually\b", "annually", cleaned)
    cleaned = merge_ocr_splits(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def chunk_tokens(
    tokens_with_pages: List[Tuple[str, int]], max_tokens: int, overlap_ratio: float = 0.15
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


def build_chunks_from_pdf(pdf_path: str, max_tokens: int) -> List[dict]:
    tokens_with_pages: List[Tuple[str, int]] = []
    pages = [text for _page_index, text in iter_pdf_pages_text(pdf_path)]
    headers, footers = extract_header_footer_candidates(pages)
    for page_index, page_text in enumerate(pages, start=1):
        cleaned_text = clean_page_text(page_text, headers, footers)
        print(f"\n--- Page {page_index} cleaned text ---")
        print(cleaned_text)
        page_tokens = tokenize(cleaned_text)
        tokens_with_pages.extend((token, page_index) for token in page_tokens)
    return chunk_tokens(tokens_with_pages, max_tokens=max_tokens, overlap_ratio=0.15)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF, visualize it, chunk into tokens, save JSON."
    )
    parser.add_argument("pdf_path", help="Path to input PDF file.")
    parser.add_argument("output_json", help="Path to output JSON file.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum tokens per chunk (default: 300).",
    )
    args = parser.parse_args()

    chunks = build_chunks_from_pdf(args.pdf_path, max_tokens=args.max_tokens)

    payload = {
        "source_pdf": args.pdf_path,
        "max_tokens": args.max_tokens,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chunks)} chunks to {args.output_json}")


if __name__ == "__main__":
    main()
