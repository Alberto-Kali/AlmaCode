from __future__ import annotations

import argparse
from pathlib import Path

from almacode_server.config import ServerConfig, apply_override, default_layout, save_server_config
from almacode_server.manager import ServerManager, ServerManagerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="almacode-server", description="Manage a local llama-server runtime")
    parser.add_argument("--home", default=None, help="Server home directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("install", help="Build llama-server from source")

    configure = subparsers.add_parser("configure", help="Write or update server config")
    configure.add_argument("--write-default", action="store_true", help="Write a default config file if missing")
    configure.add_argument("--set", action="append", default=[], help="Override config keys with section.key=value")

    subparsers.add_parser("start", help="Start llama-server")
    subparsers.add_parser("stop", help="Stop llama-server")
    subparsers.add_parser("status", help="Show server status")

    logs = subparsers.add_parser("logs", help="Show recent logs")
    logs.add_argument("--lines", type=int, default=50, help="How many log lines to show")

    subparsers.add_parser("doctor", help="Validate the environment")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = default_layout(Path(args.home).expanduser().resolve() if args.home else None)
    manager = ServerManager(layout)

    try:
        if args.command == "install":
            for line in manager.install_runtime():
                print(line)
            return 0
        if args.command == "configure":
            config = manager.load_config() if layout.config_path.exists() else ServerConfig()
            if args.write_default and not layout.config_path.exists():
                save_server_config(layout.config_path, config)
                print(f"Wrote default config to {layout.config_path}")
            for assignment in args.set:
                apply_override(config, assignment)
            save_server_config(layout.config_path, config)
            print(f"Saved config to {layout.config_path}")
            return 0
        if args.command == "start":
            print(manager.start())
            return 0
        if args.command == "stop":
            print(manager.stop())
            return 0
        if args.command == "status":
            print(manager.status())
            return 0
        if args.command == "logs":
            print(manager.logs(lines=args.lines))
            return 0
        if args.command == "doctor":
            report = manager.doctor()
            for line in report.messages:
                print(line)
            return 0 if report.ok else 1
    except ServerManagerError as exc:
        print(exc)
        return 2

    return 1
