"""Inventory the xBD archives and extract only earthquake image pairs."""

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

ROOT = Path(__file__).resolve().parent
ARCHIVES = Path("D:/pc/earthquake data/xView2/data")
NAMES = {"hold": "hold_images_labels_targets.tar.gz", "tier3": "tier3.tar.gz",
         "test": "test_images_labels_targets.tar.gz"}
PATTERN = re.compile(r"([\w-]*earthquake[\w-]*)_\d+_(?:pre|post)_disaster(?:_target)?\.(?:png|json)")


def extract(archive, output):
    counts = Counter()
    start = time.monotonic()
    with tarfile.open(archive, mode="r|gz") as source:
        for index, member in enumerate(source):
            if index % 2500 == 0:
                print(f"{archive.name}: scanned {index:,} entries, {time.monotonic() - start:.1f}s",
                      flush=True)
            path = PurePosixPath(member.name)
            match = PATTERN.fullmatch(path.name)
            if not member.isfile() or not match or len(path.parts) < 2:
                continue
            parents = {"images", "labels", "targets"}.intersection(path.parts[:-1])
            if len(parents) != 1:
                continue
            bucket = parents.pop()
            dest = output / bucket / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = source.extractfile(member).read()
            if dest.exists() and dest.read_bytes() != data:
                raise ValueError(f"Duplicate file with different content: {dest}")
            if not dest.exists():
                temp = dest.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(dest)
            counts[(match.group(1), bucket)] += 1
    print({f"{event}/{bucket}": n for (event, bucket), n in counts.items()}, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "extraction_complete.txt").write_text(str(archive.resolve()), encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("split", choices=NAMES)
    args = cli.parse_args()
    extract(ARCHIVES / NAMES[args.split], ROOT / "data" / "original" / args.split)
