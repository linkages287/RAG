#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from flask import Flask, abort, render_template, request


def create_app(default_json: Path) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        json_path = request.args.get("path") or str(default_json)
        file_path = Path(json_path)
        if not file_path.exists() or not file_path.is_file():
            abort(404, description=f"JSON file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        chunks = payload.get("chunks", [])
        return render_template(
            "index.html",
            source_pdf=payload.get("source_pdf"),
            max_tokens=payload.get("max_tokens"),
            chunk_count=payload.get("chunk_count", len(chunks)),
            chunks=chunks,
            json_path=str(file_path),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flask UI to view PDF text chunks from JSON."
    )
    parser.add_argument(
        "--json",
        default="out.json",
        help="Default JSON file to load (default: out.json).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    args = parser.parse_args()

    app = create_app(Path(args.json))
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
