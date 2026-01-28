#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def compress_folder(source_dir: Path, output_zip: Path) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Source folder not found: {source_dir}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(source_dir):
            for filename in files:
                file_path = Path(root) / filename
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)


def decompress_folder(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.is_file():
        raise SystemExit(f"Zip file not found: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zipf:
        zipf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress or decompress a folder using ZIP."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress a folder.")
    compress_parser.add_argument("source_dir", help="Folder to compress.")
    compress_parser.add_argument("output_zip", help="Output .zip file.")

    decompress_parser = subparsers.add_parser("decompress", help="Decompress a zip.")
    decompress_parser.add_argument("zip_path", help="Input .zip file.")
    decompress_parser.add_argument("output_dir", help="Output folder.")

    args = parser.parse_args()

    if args.command == "compress":
        compress_folder(Path(args.source_dir), Path(args.output_zip))
    elif args.command == "decompress":
        decompress_folder(Path(args.zip_path), Path(args.output_dir))


if __name__ == "__main__":
    main()
