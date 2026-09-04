from __future__ import annotations

from pathlib import Path

import pytest

RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"
OAK_D_DIR = RECORDINGS / "oak-d"
OPTITRACK_DIR = RECORDINGS / "optitrack"


def _files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"))


@pytest.fixture(params=_files(OAK_D_DIR), ids=lambda p: p.stem)
def oak_d_path(request) -> Path:
    return request.param


@pytest.fixture(params=_files(OPTITRACK_DIR), ids=lambda p: p.stem)
def optitrack_path(request) -> Path:
    return request.param
