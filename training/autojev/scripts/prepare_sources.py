"""Download the pinned public inputs and produce the raw source cache."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from autojev.data import ARCHIVES, ARCHIVE_SHA256, checkout, download

ROOT = Path(__file__).resolve().parents[1]


def normalize(paths, target):
    temporary = target.with_suffix(".jsonl.part")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for path in sorted(paths):
            if path.suffix == ".parquet":
                batches = (batch.to_pylist() for batch in pq.ParquetFile(path).iter_batches(batch_size=1024))
            else:
                content = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(content, list):
                    raise ValueError(f"Expected a JSON array: {path}")
                batches = [content]
            for batch in batches:
                for row in batch:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
    temporary.replace(target)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/full-public/raw")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/hf/hub")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "configs/public-sources.json").read_text())
    results = {}
    for name, spec in records.items():
        splits = {}
        for filename in spec["files"]:
            path = Path(hf_hub_download(repo_id=spec["repository"], repo_type="dataset", revision=spec["revision"], filename=filename, cache_dir=str(args.cache_dir)))
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = spec["metadata"][filename]["lfs"]["sha256"]
            if actual != expected:
                raise ValueError(f"Source checksum mismatch: {name}/{filename}")
            split = Path(filename).name.split("-")[0].split(".")[0]
            splits.setdefault(split, []).append(path)
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        results[name] = {split: normalize(paths, destination / (split + ".jsonl")) for split, paths in splits.items()}
        print(json.dumps({"source": name, "rows": results[name]}), flush=True)
    for name, (url, filename) in ARCHIVES.items():
        path = args.output / filename
        download(url, path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != ARCHIVE_SHA256[name]:
            raise ValueError(f"Archive checksum mismatch: {name}")
    dependency = checkout(None, args.output)
    (args.output / "source-cache-manifest.json").write_text(json.dumps({"rows": results, "nimble_revision": (dependency / ".revision").read_text().strip()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
