from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_ROOT = ROOT / "server_package"
SRC_ROOT = ROOT / "src" / "almacode_server"
ARCHIVE_NAME = "almacode-server-linux-x86_64.zip"


def main() -> int:
    DIST.mkdir(parents=True, exist_ok=True)
    archive_path = DIST / ARCHIVE_NAME
    if archive_path.exists():
        archive_path.unlink()

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in PACKAGE_ROOT.rglob("*"):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                zf.write(file_path, file_path.relative_to(PACKAGE_ROOT.parent))
        for file_path in SRC_ROOT.rglob("*"):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                target = Path("server_package") / "src" / file_path.relative_to(SRC_ROOT.parent)
                zf.write(file_path, target)

    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
