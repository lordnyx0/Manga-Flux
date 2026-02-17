#!/usr/bin/env python3
"""Run local bootstrap API server."""

from __future__ import annotations

import argparse

from api.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Manga-Flux bootstrap API")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
