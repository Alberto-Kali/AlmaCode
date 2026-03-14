from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from almacode.agent import CodingAgent
from almacode.config import AgentConfig
from almacode.llm import LlamaBackend, ModelLoadError
from almacode.tools import WorkspaceTools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="almacode", description="Local GGUF coding agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="Path to a local GGUF model")
    common.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path or URL. May be passed multiple times for multimodal models.",
    )
    common.add_argument("--mmproj", default=None, help="Path to the multimodal projector file")
    common.add_argument(
        "--mm-handler",
        default=None,
        help="Optional multimodal handler, for example qwen2.5-vl or llava-1-5",
    )
    common.add_argument("--workspace", default=".", help="Workspace root for file and shell access")
    common.add_argument("--chat-format", default=None, help="Force a llama.cpp chat format")
    common.add_argument("--n-ctx", type=int, default=8192, help="Context window")
    common.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers, use -1 for all")
    common.add_argument("--n-threads", type=int, default=None, help="Override llama.cpp thread count")
    common.add_argument("--max-steps", type=int, default=18, help="Maximum reasoning/tool steps")
    common.add_argument("--max-tokens", type=int, default=768, help="Max tokens per model response")
    common.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    common.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    common.add_argument("--command-timeout", type=int, default=60, help="Shell command timeout in seconds")
    common.add_argument("--system-note", default="", help="Extra instructions appended to the system prompt")
    common.add_argument("--verbose", action="store_true", help="Print raw model JSON responses")

    run_parser = subparsers.add_parser("run", parents=[common], help="Run one autonomous task")
    run_parser.add_argument("task", help="Task for the agent")

    chat_parser = subparsers.add_parser("chat", parents=[common], help="Start an interactive session")
    chat_parser.add_argument(
        "--opening-task",
        default="",
        help="Optional first task to execute before entering the interactive prompt",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        model_path=Path(args.model),
        workspace=Path(args.workspace).resolve(),
        mmproj_path=Path(args.mmproj) if args.mmproj else None,
        mm_handler=args.mm_handler,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        command_timeout=args.command_timeout,
        chat_format=args.chat_format,
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
    try:
        agent = build_agent(args, console)
    except ModelLoadError as exc:
        console.print(f"[bold red]Model load failed[/bold red]\n{exc}")
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
