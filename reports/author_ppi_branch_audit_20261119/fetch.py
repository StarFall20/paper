"""Download the nine public Python files from the pinned PPI++ branch."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def fetch(out):
    manifest = json.loads(Path(__file__).with_name("upstream.json").read_text(encoding="utf-8"))
    base = f"https://raw.githubusercontent.com/{manifest['repo']}/{manifest['commit']}/"
    for path, size, blob in manifest["files"]:
        with urlopen(base + path, timeout=20) as response:
            data = response.read(size + 1)
        if len(data) != size or hashlib.sha1(f"blob {size}\0".encode("ascii") + data).hexdigest() != blob:
            raise ValueError(f"Unexpected contents for {path}")
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    fetch(parser.parse_args().out)
