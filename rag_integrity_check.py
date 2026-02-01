#!/usr/bin/env python3
"""
Standalone program to check integrity and logical connection of data extracted
from the RAG vector DB (Weaviate).

Checks:
  - Integrity: schema, required fields, vector presence/dimensions, value ranges
  - Logical connection: chunk ordering, page continuity, metadata consistency,
    duplicate detection, country/source alignment

Usage:
    python rag_integrity_check.py
    python rag_integrity_check.py --url http://localhost:8080
    python rag_integrity_check.py --collection Cms --collection DocumentChunk
    python rag_integrity_check.py --sample 500
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import weaviate
except ImportError:
    print("Error: weaviate package is not installed.")
    print("Install it with: pip install weaviate-client")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Connection (same logic as connect_weaviate.py)
# ---------------------------------------------------------------------------

def connect_to_weaviate(weaviate_url: str = "http://localhost:8080"):
    """Connect to Weaviate. Returns client or None."""
    try:
        if weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
            client = weaviate.connect_to_local()
        else:
            url_clean = weaviate_url.replace("http://", "").replace("https://", "")
            host, port = (url_clean.split(":", 1)[0], int(url_clean.split(":", 1)[1])) if ":" in url_clean else (url_clean, 8080)
            http_secure = weaviate_url.startswith("https://")
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=http_secure,
            )
        if client.is_ready():
            return client
        return None
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class CollectionReport:
    collection_name: str
    object_count: int
    integrity_results: List[CheckResult] = field(default_factory=list)
    logical_results: List[CheckResult] = field(default_factory=list)
    sample_size: int = 0


# ---------------------------------------------------------------------------
# Data fetching (paginated, with optional vector sampling)
# ---------------------------------------------------------------------------

def fetch_collection_sample(
    client: weaviate.WeaviateClient,
    collection_name: str,
    max_objects: int = 2000,
    include_vectors: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Optional[List[float]]], Optional[Dict]]:
    """
    Fetch a sample of objects from a collection.
    Returns (list of property dicts, list of vectors or None, schema info).
    """
    if not client.collections.exists(collection_name):
        return [], [], None

    collection = client.collections.get(collection_name)
    config = collection.config.get()
    schema_info = {
        "properties": [p.name for p in config.properties],
        "vectorizer": getattr(config.vectorizer_config, "vectorizer", None) if hasattr(config, "vectorizer_config") else None,
    }

    objects_data: List[Dict[str, Any]] = []
    vectors: List[Optional[List[float]]] = []
    limit = min(500, max_objects)
    offset = 0
    fetched = 0

    while fetched < max_objects:
        result = collection.query.fetch_objects(limit=limit, offset=offset)
        if not result.objects:
            break
        for obj in result.objects:
            if fetched >= max_objects:
                break
            props = dict(obj.properties) if obj.properties else {}
            props["_uuid"] = str(obj.uuid)
            objects_data.append(props)

            vec = None
            if include_vectors:
                try:
                    obj_with_vec = collection.data.get_by_id(obj.uuid, include_vector=True)
                    if obj_with_vec:
                        v = getattr(obj_with_vec, "vector", None) or (getattr(obj_with_vec, "vectors", None) or {})
                        if isinstance(v, dict):
                            vec = v.get("default") or (list(v.values())[0] if v else None)
                        else:
                            vec = v
                except Exception:
                    pass
            vectors.append(vec)
            fetched += 1
        if len(result.objects) < limit:
            break
        offset += limit

    return objects_data, vectors, schema_info


def get_collection_count(client: weaviate.WeaviateClient, collection_name: str) -> int:
    """Return total object count for a collection."""
    try:
        collection = client.collections.get(collection_name)
        result = collection.query.fetch_objects(limit=1, return_metadata=weaviate.classes.query.MetadataQuery(count=True))
        return getattr(result, "total_count", None) or getattr(result, "total", 0) or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def check_schema(schema_info: Optional[Dict], expected_props: Optional[List[str]] = None) -> CheckResult:
    """Check that collection has expected properties (if specified)."""
    if not schema_info or "properties" not in schema_info:
        return CheckResult("schema", False, "Could not retrieve schema", schema_info)
    props = set(schema_info["properties"])
    if expected_props:
        missing = set(expected_props) - props
        if missing:
            return CheckResult("schema", False, f"Missing properties: {sorted(missing)}", {"expected": expected_props, "actual": list(props)})
    return CheckResult("schema", True, f"Properties: {sorted(props)}", {"properties": list(props)})


def check_object_count(objects: List[Dict], min_expected: int = 1) -> CheckResult:
    """Check that collection has at least min_expected objects."""
    n = len(objects)
    passed = n >= min_expected
    return CheckResult("object_count", passed, f"Object count: {n}" + (f" (min {min_expected})" if min_expected > 1 else ""), {"count": n})


def check_required_fields(objects: List[Dict], required: List[str], allow_empty: Optional[Set[str]] = None) -> CheckResult:
    """Check that required fields exist and (unless in allow_empty) are non-empty."""
    allow_empty = allow_empty or set()
    missing_field_count = 0
    empty_count: Dict[str, int] = defaultdict(int)
    for obj in objects:
        for key in required:
            if key not in obj:
                missing_field_count += 1
                break
            val = obj.get(key)
            if key not in allow_empty and (val is None or (isinstance(val, str) and not val.strip())):
                empty_count[key] += 1
    issues = []
    if missing_field_count:
        issues.append(f"{missing_field_count} objects missing required keys")
    for k, c in empty_count.items():
        issues.append(f"{c} objects with empty '{k}'")
    passed = missing_field_count == 0 and not empty_count
    return CheckResult(
        "required_fields",
        passed,
        "All required fields present and non-empty" if passed else "; ".join(issues),
        {"missing_objects": missing_field_count, "empty_by_field": dict(empty_count)},
    )


def check_vector_presence(vectors: List[Optional[List[float]]]) -> CheckResult:
    """Check that vectors exist and have consistent dimension."""
    present = sum(1 for v in vectors if v is not None and len(v) > 0)
    total = len(vectors)
    dims: Set[int] = {len(v) for v in vectors if v is not None and len(v) > 0}
    if total == 0:
        return CheckResult("vectors", False, "No objects to check", {})
    if present == 0:
        return CheckResult("vectors", False, "No vectors found", {"total": total})
    if len(dims) > 1:
        return CheckResult("vectors", False, f"Inconsistent vector dimensions: {dims}", {"total": total, "with_vector": present, "dimensions": list(dims)})
    dim = dims.pop() if dims else 0
    passed = present == total and dim > 0
    return CheckResult(
        "vectors",
        passed,
        f"Vectors: {present}/{total} present, dimension {dim}" + (" (all present)" if passed else " (some missing)"),
        {"total": total, "with_vector": present, "dimension": dim},
    )


def check_page_ranges(objects: List[Dict], page_start_key: str = "page_start", page_end_key: str = "page_end") -> CheckResult:
    """Check that page_start <= page_end and both are non-negative."""
    invalid = 0
    for obj in objects:
        ps = obj.get(page_start_key)
        pe = obj.get(page_end_key)
        if ps is None or pe is None:
            continue
        try:
            ps, pe = int(ps), int(pe)
            if ps < 0 or pe < 0 or ps > pe:
                invalid += 1
        except (TypeError, ValueError):
            invalid += 1
    total = len(objects)
    passed = invalid == 0
    return CheckResult(
        "page_ranges",
        passed,
        f"Page range validity: {total - invalid}/{total} valid" + (f", {invalid} invalid" if invalid else ""),
        {"invalid_count": invalid, "total": total},
    )


def check_token_count_non_negative(objects: List[Dict], token_key: str = "token_count") -> CheckResult:
    """Check that token_count (if present) is non-negative."""
    negative = 0
    for obj in objects:
        t = obj.get(token_key)
        if t is None:
            continue
        try:
            if int(t) < 0:
                negative += 1
        except (TypeError, ValueError):
            negative += 1
    total = len([o for o in objects if token_key in o])
    passed = negative == 0
    return CheckResult(
        "token_count",
        passed,
        f"Token count non-negative: {total - negative}/{total} valid" + (f", {negative} invalid" if negative else ""),
        {"negative_count": negative, "checked": total},
    )


# ---------------------------------------------------------------------------
# Logical connection checks
# ---------------------------------------------------------------------------

def extract_country_from_pdf(source_pdf: str) -> str:
    """Extract country code from PDF filename (e.g. NU_JWC_ETI_DUSHMAN -> DUSHMAN)."""
    if not source_pdf:
        return ""
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", str(source_pdf).upper())
    return match.group(1) if match else ""


def check_chunk_id_uniqueness_per_source(objects: List[Dict], source_key: str = "source_pdf", chunk_id_key: str = "chunk_id") -> CheckResult:
    """Check that (source_pdf, chunk_id) are unique."""
    seen: Set[Tuple[str, Any]] = set()
    duplicates: List[Tuple[str, Any]] = []
    for obj in objects:
        src = obj.get(source_key) or "unknown"
        cid = obj.get(chunk_id_key)
        key = (src, cid)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    total = len(objects)
    dup_count = len(duplicates)
    passed = dup_count == 0
    return CheckResult(
        "chunk_id_uniqueness",
        passed,
        f"Unique (source, chunk_id): {total - dup_count}/{total}" + (f", {dup_count} duplicates" if dup_count else ""),
        {"duplicates": duplicates[:20], "duplicate_count": dup_count},
    )


def check_page_continuity_per_source(objects: List[Dict], source_key: str = "source_pdf", page_start_key: str = "page_start", page_end_key: str = "page_end") -> CheckResult:
    """
    Per source, check that page ranges do not overlap in an invalid way.
    We allow adjacent or overlapping (e.g. overlap by design); flag only if end < start.
    """
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for obj in objects:
        by_source[obj.get(source_key) or "unknown"].append(obj)

    gaps_or_overlaps = 0
    sources_checked = 0
    for source, objs in by_source.items():
        objs_sorted = sorted(objs, key=lambda x: (x.get(page_start_key) or 0, x.get(page_end_key) or 0))
        prev_end = -1
        for o in objs_sorted:
            ps = o.get(page_start_key)
            pe = o.get(page_end_key)
            if ps is None or pe is None:
                continue
            try:
                ps, pe = int(ps), int(pe)
                if prev_end >= 0 and ps > prev_end + 1:
                    gaps_or_overlaps += 1  # gap
                elif prev_end >= 0 and pe < prev_end:
                    gaps_or_overlaps += 1  # backward
                prev_end = max(prev_end, pe)
            except (TypeError, ValueError):
                pass
        sources_checked += 1

    total_objects = len(objects)
    passed = gaps_or_overlaps == 0
    return CheckResult(
        "page_continuity",
        passed,
        f"Page continuity: {sources_checked} sources, {gaps_or_overlaps} gaps/ordering issues" if gaps_or_overlaps else "Page ordering consistent per source",
        {"gaps_or_issues": gaps_or_overlaps, "sources": sources_checked},
    )


def check_country_source_alignment(objects: List[Dict], country_key: str = "country", source_key: str = "source_pdf") -> CheckResult:
    """Check that country field aligns with country extracted from source_pdf."""
    mismatches = 0
    for obj in objects:
        country = (obj.get(country_key) or "").strip().upper()
        source = obj.get(source_key) or ""
        extracted = extract_country_from_pdf(source)
        if not extracted:
            continue
        if country != extracted:
            mismatches += 1
    total = len(objects)
    passed = mismatches == 0
    return CheckResult(
        "country_source_alignment",
        passed,
        f"Country/source alignment: {total - mismatches}/{total} match" + (f", {mismatches} mismatches" if mismatches else ""),
        {"mismatch_count": mismatches},
    )


def check_duplicate_text(objects: List[Dict], text_key: str = "text") -> CheckResult:
    """Flag duplicate text content (same text in multiple chunks)."""
    by_text: Dict[str, int] = defaultdict(int)
    for obj in objects:
        t = (obj.get(text_key) or "").strip()
        if t:
            by_text[t] += 1
    duplicates = {k: v for k, v in by_text.items() if v > 1}
    dup_count = sum(v - 1 for v in duplicates.values())
    total = len(objects)
    passed = len(duplicates) == 0
    return CheckResult(
        "duplicate_text",
        passed,
        f"Duplicate text: {len(duplicates)} unique texts repeated, {dup_count} extra copies" if duplicates else "No duplicate text found",
        {"duplicate_texts_count": len(duplicates), "extra_copies": dup_count},
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_checks(
    client: weaviate.WeaviateClient,
    collection_name: str,
    sample_size: int = 2000,
) -> CollectionReport:
    """Run all integrity and logical checks on a collection."""
    report = CollectionReport(collection_name=collection_name, object_count=0, sample_size=sample_size)

    total = get_collection_count(client, collection_name)
    report.object_count = total

    objects, vectors, schema_info = fetch_collection_sample(client, collection_name, max_objects=sample_size, include_vectors=True)
    report.sample_size = len(objects)

    if not objects:
        report.integrity_results.append(CheckResult("fetch", False, "Could not fetch any objects", {}))
        if total == 0:
            report.integrity_results.append(CheckResult("object_count", False, "Collection is empty or count unavailable", {}))
        return report
    if total == 0:
        report.object_count = len(objects)
        report.integrity_results.append(CheckResult("object_count", True, f"Sample only: {len(objects)} objects (total count API returned 0)", {"count": len(objects)}))
    else:
        report.integrity_results.append(CheckResult("object_count", True, f"Object count: {total}", {"count": total}))

    # Expected properties for RAG document chunks (flexible: only check if present)
    expected_props = ["text", "source_pdf", "chunk_id", "country", "page_start", "page_end", "token_count"]
    report.integrity_results.append(check_schema(schema_info, expected_props))
    report.integrity_results.append(check_required_fields(objects, required=["text"], allow_empty=set()))
    report.integrity_results.append(check_vector_presence(vectors))
    report.integrity_results.append(check_page_ranges(objects))
    report.integrity_results.append(check_token_count_non_negative(objects))

    report.logical_results.append(check_chunk_id_uniqueness_per_source(objects))
    report.logical_results.append(check_page_continuity_per_source(objects))
    report.logical_results.append(check_country_source_alignment(objects))
    report.logical_results.append(check_duplicate_text(objects))

    return report


def print_report(report: CollectionReport, verbose: bool = False) -> None:
    """Print a single collection report."""
    print(f"\n{'='*60}")
    print(f"Collection: {report.collection_name}")
    print(f"Total objects: {report.object_count}  (sample size: {report.sample_size})")
    print("=" * 60)

    def _print_results(results: List[CheckResult], title: str):
        print(f"\n  {title}")
        for r in results:
            icon = "✓" if r.passed else "✗"
            print(f"    {icon} {r.name}: {r.message}")
            if verbose and r.details:
                for k, v in r.details.items():
                    print(f"       {k}: {v}")

    _print_results(report.integrity_results, "Integrity checks")
    _print_results(report.logical_results, "Logical connection checks")

    passed_i = sum(1 for r in report.integrity_results if r.passed)
    passed_l = sum(1 for r in report.logical_results if r.passed)
    total_i = len(report.integrity_results)
    total_l = len(report.logical_results)
    print(f"\n  Summary: Integrity {passed_i}/{total_i}, Logical {passed_l}/{total_l}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check integrity and logical connection of RAG vector DB data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--collection", action="append", dest="collections", help="Collection name (can be repeated); default: all")
    parser.add_argument("--sample", type=int, default=2000, help="Max objects to sample per collection (default: 2000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed check results")
    args = parser.parse_args()

    print("RAG Vector DB Integrity & Logical Connection Check")
    print("Connecting to Weaviate...")
    client = connect_to_weaviate(args.url)
    if not client:
        sys.exit(1)
    print("✓ Connected\n")

    try:
        if args.collections:
            collection_names = args.collections
        else:
            collection_names = list(client.collections.list_all())
        if not collection_names:
            print("No collections found.")
            return

        for name in collection_names:
            if not client.collections.exists(name):
                print(f"\n✗ Collection '{name}' does not exist.")
                continue
            report = run_checks(client, name, sample_size=args.sample)
            print_report(report, verbose=args.verbose)
    finally:
        if hasattr(client, "__exit__"):
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()
