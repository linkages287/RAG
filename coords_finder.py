#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def dms_to_decimal(deg: float, minute: float, sec: float, hem: str) -> float:
    value = deg + (minute / 60.0) + (sec / 3600.0)
    hem = hem.upper()
    if hem in {"S", "W"}:
        return -value
    return value


def normalize_decimal_pair(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    if -90.0 <= lon <= 90.0 and -180.0 <= lat <= 180.0:
        return lon, lat
    return None


def extract_decimals(text: str) -> List[Dict]:
    results = []
    for match in DECIMAL_PAIR_RE.finditer(text):
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
        normalized = normalize_decimal_pair(lat, lon)
        if not normalized:
            continue
        results.append(
            {
                "format": "decimal",
                "lat": normalized[0],
                "lon": normalized[1],
                "raw": match.group(0),
            }
        )
    return results


def extract_dms(text: str) -> List[Dict]:
    matches = list(DMS_RE.finditer(text))
    results = []
    for i in range(0, len(matches) - 1, 2):
        first = matches[i]
        second = matches[i + 1]
        lat_hem = first.group("hem").upper()
        lon_hem = second.group("hem").upper()
        if {lat_hem, lon_hem} != {"N", "S"} and {lat_hem, lon_hem} != {"E", "W"}:
            pass
        lat_val = dms_to_decimal(
            float(first.group("deg")),
            float(first.group("min")),
            float(first.group("sec")),
            lat_hem,
        )
        lon_val = dms_to_decimal(
            float(second.group("deg")),
            float(second.group("min")),
            float(second.group("sec")),
            lon_hem,
        )
        normalized = normalize_decimal_pair(lat_val, lon_val)
        if not normalized:
            continue
        raw = f"{first.group(0)} {second.group(0)}"
        results.append(
            {
                "format": "dms",
                "lat": normalized[0],
                "lon": normalized[1],
                "raw": raw,
            }
        )
    return results


def load_json_text(json_path: Path) -> str:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False)


def find_in_lines(lines: List[str]) -> List[Dict]:
    results: List[Dict] = []
    for line_no, line in enumerate(lines, start=1):
        for match in DECIMAL_PAIR_RE.finditer(line):
            lat = float(match.group("lat"))
            lon = float(match.group("lon"))
            normalized = normalize_decimal_pair(lat, lon)
            if not normalized:
                continue
            results.append(
                {
                    "format": "decimal",
                    "lat": normalized[0],
                    "lon": normalized[1],
                    "raw": match.group(0),
                    "line": line_no,
                }
            )
        dms_matches = list(DMS_RE.finditer(line))
        pending_lat = None
        pending_lon = None
        for match in dms_matches:
            hem = match.group("hem").upper()
            if hem in {"N", "S"}:
                if pending_lon is not None:
                    lat_match = match
                    lon_match = pending_lon
                    pending_lon = None
                else:
                    pending_lat = match
                    continue
            else:
                if pending_lat is not None:
                    lat_match = pending_lat
                    lon_match = match
                    pending_lat = None
                else:
                    pending_lon = match
                    continue

            lat_hem = lat_match.group("hem").upper()
            lon_hem = lon_match.group("hem").upper()
            lat_sec = float(lat_match.group("sec") or 0)
            lon_sec = float(lon_match.group("sec") or 0)
            lat_val = dms_to_decimal(
                float(lat_match.group("deg")),
                float(lat_match.group("min")),
                lat_sec,
                lat_hem,
            )
            lon_val = dms_to_decimal(
                float(lon_match.group("deg")),
                float(lon_match.group("min")),
                lon_sec,
                lon_hem,
            )
            normalized = normalize_decimal_pair(lat_val, lon_val)
            if not normalized:
                continue
            raw = f"{lat_match.group(0)} {lon_match.group(0)}"
            results.append(
                {
                    "format": "dms",
                    "lat": normalized[0],
                    "lon": normalized[1],
                    "raw": raw,
                    "line": line_no,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find coordinates in a JSON file (decimal or DMS formats)."
    )
    parser.add_argument("input_json", help="Path to input JSON file.")
    parser.add_argument("output_json", help="Path to output JSON with matches.")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    results = find_in_lines(lines)

    for item in results:
        print(f"Line {item['line']}: {item['raw']}")

    out = {
        "input": args.input_json,
        "count": len(results),
        "coordinates": results,
    }
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Found {len(results)} coordinate matches in {args.input_json}")


if __name__ == "__main__":
    main()
