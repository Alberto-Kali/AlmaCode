from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from almacode.agent import AgentRuntimeError, CodingAgent
from almacode.config import (
    AgentConfig,
    ClientSettings,
    autodetect_server_url,
    build_server_url,
    deprecated_runtime_flags,
    resolve_client_settings,
    save_client_settings,
)
from almacode.llm import LlamaBackend, ModelLoadError
from almacode.session import AgentSession
from almacode.tools import WorkspaceTools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="almacode", description="CLI coding agent for llama-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server-url", default=None, help="llama-server base URL")
    common.add_argument("--server-host", default=None, help="llama-server host, used when --server-url is omitted")
    common.add_argument("--server-port", type=int, default=None, help="llama-server port, used with --server-host")
    common.add_argument("--api-key", default=None, help="Optional bearer token for llama-server")
    common.add_argument("--request-timeout", type=int, default=None, help="HTTP timeout in seconds")
    common.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path or URL. May be passed multiple times.",
    )
    common.add_argument("--workspace", default=".", help="Workspace root for file and shell access")
    common.add_argument("--max-steps", type=int, default=15, help="Maximum reasoning/tool substeps per plan step")
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
    chat_parser.add_argument("--plain", action="store_true", help="Use the legacy line-by-line chat mode")

    init_parser = subparsers.add_parser("init-client", help="Write ~/.config/almacode/client.json")
    init_parser.add_argument("--server-url", default=None, help="llama-server base URL")
    init_parser.add_argument("--server-host", default=None, help="llama-server host, used when --server-url is omitted")
    init_parser.add_argument("--server-port", type=int, default=None, help="llama-server port, used with --server-host")
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

    client = resolve_client_settings(
        args.server_url,
        getattr(args, "server_host", None),
        getattr(args, "server_port", None),
        args.api_key,
        args.request_timeout,
    )
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
        context_window=4096,
        context_soft_limit_ratio=0.82,
        summary_max_tokens=256,
        history_tail_messages=4,
        request_retries=2,
        max_plan_steps=10,
    )


def build_agent(args: argparse.Namespace, console: Console) -> CodingAgent:
    config = config_from_args(args)
    backend = LlamaBackend(config)
    tools = WorkspaceTools(root=config.workspace, default_timeout=config.command_timeout)
    return CodingAgent(config=config, backend=backend, tools=tools, console=console)


def run_once(agent: CodingAgent, task: str, session: AgentSession | None = None, image_refs: list[str] | None = None) -> str:
    return agent.run(task, session=session, image_refs=image_refs)


def _extract_tool_result_payload(message: str) -> tuple[str, str]:
    marker = "\nresult:\n"
    if marker not in message:
        return message, ""
    header, payload = message.split(marker, 1)
    return header, payload.strip()


def format_plain_tool_event(kind: str, message: str) -> list[str]:
    header, payload = _extract_tool_result_payload(message)
    lines = [line for line in header.splitlines() if line.strip()]
    field_map: dict[str, str] = {}
    for line in lines:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        field_map[key] = value

    tool = field_map.get("tool", "tool")
    if kind == "tool_start":
        if tool == "run_command":
            return [
                f"[bold yellow]Command[/bold yellow] {field_map.get('command', '')}",
                f"[dim]cwd: {field_map.get('cwd', '.')}[/dim]",
            ]
        if tool in {"read_file", "write_file", "replace_in_file", "make_dir", "list_dir"}:
            return [f"[bold yellow]{tool}[/bold yellow] {field_map.get('path', '.')}"]
        if tool == "web_search":
            return [f"[bold yellow]web_search[/bold yellow] {field_map.get('query', '')}"]
        if tool == "open_url":
            return [f"[bold yellow]open_url[/bold yellow] {field_map.get('url', '')}"]
        return [f"[bold yellow]{tool}[/bold yellow]"]

    rendered: list[str] = []
    if tool == "run_command":
        status_label = "Command done"
        status_style = "bold blue"
        rendered.append(
            f"[{status_style}]{status_label}[/{status_style}] {field_map.get('command', '')} "
            f"(cwd={field_map.get('cwd', '.')})"
        )
        try:
            payload_json = json.loads(payload)
        except json.JSONDecodeError:
            payload_json = {}
        if payload_json:
            exit_code = payload_json.get("exit_code", "?")
            if exit_code == 0:
                rendered.append("[green]status: success[/green]")
            else:
                rendered.append("[red]status: failure[/red]")
            rendered.append(f"[dim]exit_code: {exit_code}[/dim]")
            if payload_json.get("virtual_env"):
                rendered.append(f"[dim]virtual_env: {payload_json['virtual_env']}[/dim]")
            stdout = str(payload_json.get("stdout", "")).strip()
            stderr = str(payload_json.get("stderr", "")).strip()
            if stdout:
                rendered.append("[bold]stdout:[/bold]")
                rendered.append(stdout)
            if stderr:
                if exit_code == 0:
                    rendered.append("[yellow]stderr contained warnings/notices, but the command succeeded[/yellow]")
                rendered.append("[bold]stderr:[/bold]")
                rendered.append(stderr)
            if not stdout and not stderr:
                rendered.append("[dim]No stdout/stderr[/dim]")
            return rendered

    label = {
        "read_file": "Read file",
        "write_file": "Wrote file",
        "replace_in_file": "Changed file",
        "make_dir": "Created dir",
        "list_dir": "List dir",
        "web_search": "Web search",
        "open_url": "Open URL",
    }.get(tool, tool)
    rendered.append(f"[bold blue]{label}[/bold blue]")
    if payload:
        rendered.append(payload)
    return rendered


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()
    if args.command == "init-client":
        detected = build_server_url(args.server_url, args.server_host, args.server_port) or autodetect_server_url(timeout=args.request_timeout)
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
        try:
            answer = run_once(agent, args.task, image_refs=args.image)
        except AgentRuntimeError as exc:
            console.print(f"[bold red]Run failed[/bold red]\n{exc}")
            return 2
        console.print(answer)
        return 0

    session = AgentSession()
    active_images = list(args.image)
    if not args.plain:
        from almacode.tui import AlmaCodeApp

        tui_agent = CodingAgent(
            config=agent._config,
            backend=agent._backend,
            tools=agent._tools,
            console=console,
        )
        app = AlmaCodeApp(tui_agent, session, initial_images=active_images, opening_task=args.opening_task)
        tui_agent._event_handler = app.handle_agent_event
        app.run()
        return 0

    if args.opening_task:
        try:
            answer = run_once(agent, args.opening_task, session=session, image_refs=active_images)
        except AgentRuntimeError as exc:
            console.print(f"[bold red]Run failed[/bold red]\n{exc}")
            return 2
        console.print(answer)

    console.print("Interactive mode. Type /exit to quit, /image <path...> to set images, /clear-images to unset.")

    def plain_event_handler(kind: str, message: str) -> None:
        if kind in {"tool_start", "tool"}:
            for line in format_plain_tool_event(kind, message):
                console.print(line)
            return
        if kind == "plan":
            console.print("[bold cyan]Plan[/bold cyan]")
            console.print(message)

    agent._event_handler = plain_event_handler
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
        try:
            answer = run_once(agent, prompt, session=session, image_refs=active_images)
        except AgentRuntimeError as exc:
            console.print(f"[bold red]Run failed[/bold red]\n{exc}")
            continue
        console.print(answer)
