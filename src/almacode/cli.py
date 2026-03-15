from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from almacode.agent import CodingAgent
from almacode.config import AgentConfig, ClientSettings, autodetect_server_url, deprecated_runtime_flags, resolve_client_settings, save_client_settings
from almacode.llm import LlamaBackend, ModelLoadError
from almacode.tools import WorkspaceTools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="almacode", description="CLI coding agent for llama-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server-url", default=None, help="llama-server base URL")
    common.add_argument("--api-key", default=None, help="Optional bearer token for llama-server")
    common.add_argument("--request-timeout", type=int, default=None, help="HTTP timeout in seconds")
    common.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path or URL. May be passed multiple times.",
    )
    common.add_argument("--workspace", default=".", help="Workspace root for file and shell access")
    common.add_argument("--max-steps", type=int, default=18, help="Maximum reasoning/tool steps")
    common.add_argument("--max-tokens", type=int, default=768, help="Max tokens per model response")
    common.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    common.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    common.add_argument("--command-timeout", type=int, default=60, help="Shell command timeout in seconds")
    common.add_argument("--system-note", default="", help="Extra instructions appended to the system prompt")
    common.add_argument("--verbose", action="store_true", help="Print raw model JSON responses")

    # Deprecated compatibility flags: parsed only to produce migration guidance.
    common.add_argument("--model", default=None, help=argparse.SUPPRESS)
    common.add_argument("--mmproj", default=None, help=argparse.SUPPRESS)
    common.add_argument("--mm-handler", default=None, help=argparse.SUPPRESS)
    common.add_argument("--backend", default="auto", help=argparse.SUPPRESS)
    common.add_argument("--llama-server-binary", default=None, help=argparse.SUPPRESS)
    common.add_argument("--chat-format", default=None, help=argparse.SUPPRESS)
    common.add_argument("--n-gpu-layers", type=int, default=0, help=argparse.SUPPRESS)

    run_parser = subparsers.add_parser("run", parents=[common], help="Run one autonomous task")
    run_parser.add_argument("task", help="Task for the agent")

    chat_parser = subparsers.add_parser("chat", parents=[common], help="Start an interactive session")
    chat_parser.add_argument("--opening-task", default="", help="Optional first task to execute")

    init_parser = subparsers.add_parser("init-client", help="Write ~/.config/almacode/client.json")
    init_parser.add_argument("--server-url", default=None, help="llama-server base URL")
    init_parser.add_argument("--api-key", default=None, help="Optional bearer token")
    init_parser.add_argument("--request-timeout", type=int, default=120, help="HTTP timeout in seconds")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing client config")

    return parser


def deprecated_flags_message(active_flags: dict[str, object]) -> str:
    names = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(active_flags))
    return (
        "Deprecated local runtime flags detected: "
        f"{names}\n"
        "AlmaCode now talks only to llama-server.\n"
        "Move model, mmproj, GPU and backend settings into the server config, and use --server-url on the client."
    )


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    active_flags = deprecated_runtime_flags(args)
    if active_flags:
        raise ValueError(deprecated_flags_message(active_flags))

    client = resolve_client_settings(args.server_url, args.api_key, args.request_timeout)
    return AgentConfig(
        workspace=Path(args.workspace).resolve(),
        server_url=client.server_url,
        api_key=client.api_key,
        request_timeout=client.request_timeout,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        command_timeout=args.command_timeout,
        verbose=args.verbose,
        system_note=args.system_note,
    )


def build_agent(args: argparse.Namespace, console: Console) -> CodingAgent:
    config = config_from_args(args)
    backend = LlamaBackend(config)
    tools = WorkspaceTools(root=config.workspace, default_timeout=config.command_timeout)
    return CodingAgent(config=config, backend=backend, tools=tools, console=console)


def run_once(
    agent: CodingAgent,
    task: str,
    history: list[dict[str, object]] | None = None,
    image_refs: list[str] | None = None,
) -> str:
    return agent.run(task, history=history, image_refs=image_refs)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()
    if args.command == "init-client":
        detected = (args.server_url or "").strip() or autodetect_server_url(timeout=args.request_timeout)
        if not detected:
            console.print(
                "[bold red]Startup failed[/bold red]\n"
                "Could not auto-detect a local llama-server.\n"
                "Pass --server-url explicitly."
            )
            return 2
        try:
            config_path = save_client_settings(
                ClientSettings(server_url=detected, api_key=args.api_key, request_timeout=args.request_timeout),
                overwrite=args.force,
            )
        except ValueError as exc:
            console.print(f"[bold red]Startup failed[/bold red]\n{exc}")
            return 2
        console.print(f"Wrote client config to {config_path}\nServer URL: {detected}")
        return 0

    try:
        agent = build_agent(args, console)
    except (ModelLoadError, ValueError) as exc:
        console.print(f"[bold red]Startup failed[/bold red]\n{exc}")
        return 2

    if args.command == "run":
        answer = run_once(agent, args.task, image_refs=args.image)
        console.print(answer)
        return 0

    history: list[dict[str, object]] = []
    active_images = list(args.image)
    if args.opening_task:
        answer = run_once(agent, args.opening_task, history=history, image_refs=active_images)
        console.print(answer)
        history.extend(
            [
                {"role": "user", "content": args.opening_task},
                {"role": "assistant", "content": answer},
            ]
        )

    console.print("Interactive mode. Type /exit to quit, /image <path...> to set images, /clear-images to unset.")
    while True:
        prompt = Prompt.ask("[bold green]you[/bold green]")
        if prompt.strip() in {"/exit", "/quit"}:
            return 0
        if prompt.startswith("/image "):
            active_images = prompt.split()[1:]
            console.print(f"Active images: {', '.join(active_images)}")
            continue
        if prompt.strip() == "/clear-images":
            active_images = []
            console.print("Active images cleared.")
            continue
        if not prompt.strip():
            continue
        answer = run_once(agent, prompt, history=history, image_refs=active_images)
        console.print(answer)
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )
