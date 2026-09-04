"""Application entry point."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from .ui.main_window import run

    return run(sys.argv if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
