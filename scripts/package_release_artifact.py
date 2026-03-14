from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename built binary for release upload")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing the built binary")
    parser.add_argument("--artifact-name", required=True, help="Base output name without extension")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.exists():
        raise SystemExit(f"Dist directory does not exist: {dist_dir}")

    files = [path for path in dist_dir.iterdir() if path.is_file()]
    if len(files) != 1:
        raise SystemExit(f"Expected exactly one built file in {dist_dir}, found {len(files)}")

    source = files[0]
    suffix = source.suffix
    target = dist_dir / f"{args.artifact_name}{suffix}"
    if target.exists():
        target.unlink()
    source.rename(target)
    print(target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

