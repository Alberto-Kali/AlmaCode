from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Input, RichLog, Static, TabbedContent, TabPane

from almacode.agent import AgentRuntimeError, CodingAgent
from almacode.session import AgentSession


class AlmaCodeApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #status-bar { height: 3; padding: 0 1; }
    #composer { height: 3; }
    RichLog, #status-view, #context-view { border: solid #666666; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+shift+c", "copy_current_tab", "Copy"),
        ("f1", "show_chat", "Chat"),
        ("f2", "show_status", "Status"),
        ("f3", "show_logs", "Logs"),
        ("f4", "show_context", "Context"),
    ]

    def __init__(
        self,
        agent: CodingAgent,
        session: AgentSession,
        initial_images: list[str] | None = None,
        opening_task: str = "",
    ) -> None:
        super().__init__()
        self._agent = agent
        self._session = session
        self._active_images = list(initial_images or [])
        self._opening_task = opening_task
        self._chat_plain_lines: list[str] = []
        self._tool_plain_lines: list[str] = []
        self._last_chat_group = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status-bar")
        with TabbedContent(initial="chat"):
            with TabPane("Chat", id="chat"):
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
            with TabPane("Plan/Status", id="status"):
                yield Static("", id="status-view")
            with TabPane("Logs/Tools", id="logs"):
                yield RichLog(id="tool-log", wrap=True, highlight=True, markup=False)
            with TabPane("Context", id="context"):
                yield Static("", id="context-view")
        with Horizontal(id="composer"):
            yield Input(placeholder="Ask AlmaCode to work in this workspace...", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()
        self.query_one("#chat-log", RichLog).write(
            "TUI mode. Commands: /exit, /image <path...>, /clear-images, F1-F4 to switch tabs."
        )
        if self._opening_task.strip():
            self._write_chat(f"[you] {self._opening_task.strip()}")
            self.run_prompt(self._opening_task.strip())

    def action_show_chat(self) -> None:
        self.query_one(TabbedContent).active = "chat"

    def action_show_status(self) -> None:
        self.query_one(TabbedContent).active = "status"

    def action_show_logs(self) -> None:
        self.query_one(TabbedContent).active = "logs"

    def action_show_context(self) -> None:
        self.query_one(TabbedContent).active = "context"

    def action_copy_current_tab(self) -> None:
        active = self.query_one(TabbedContent).active
        if active == "chat":
            content = "\n".join(self._chat_plain_lines)
        elif active == "logs":
            content = "\n".join(self._tool_plain_lines)
        elif active == "status":
            content = self.query_one("#status-view", Static).renderable
        else:
            content = self.query_one("#context-view", Static).renderable
        text = str(content)
        self.copy_to_clipboard(text)
        self.notify("Copied current tab to clipboard")

    @on(Input.Submitted, "#prompt")
    def handle_submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        if prompt in {"/exit", "/quit"}:
            self.exit()
            return
        if prompt.startswith("/image "):
            self._active_images = prompt.split()[1:]
            self._write_chat(f"[images] {', '.join(self._active_images)}")
            self._refresh_status()
            return
        if prompt == "/clear-images":
            self._active_images = []
            self._write_chat("[images cleared]")
            self._refresh_status()
            return
        self._write_chat(f"[you] {prompt}")
        self.run_prompt(prompt)

    @work(thread=True, exclusive=True)
    def run_prompt(self, prompt: str) -> None:
        try:
            self._agent.run(prompt, session=self._session, image_refs=self._active_images)
        except AgentRuntimeError as exc:
            self.call_from_thread(self._handle_agent_event_main, "error", str(exc))
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._handle_agent_event_main, "error", f"Unexpected failure: {exc}")
        finally:
            self.call_from_thread(self._refresh_status)

    def handle_agent_event(self, kind: str, message: str) -> None:
        self.call_from_thread(self._handle_agent_event_main, kind, message)

    def _handle_agent_event_main(self, kind: str, message: str) -> None:
        if kind in {"thought", "status", "final", "error"}:
            self._write_chat(f"[{kind}] {message}", group=kind)
        if kind == "tool_start":
            self._write_chat(self._tool_summary_for_chat(message, started=True), group="tool")
        if kind == "tool":
            self.query_one("#tool-log", RichLog).write(message)
            self._tool_plain_lines.extend(["", message])
            self._write_chat(self._tool_summary_for_chat(message, started=False), group="tool")
        if kind == "context":
            self.query_one("#context-view", Static).update(message)
        if kind == "status":
            self.query_one("#status-view", Static).update(message)
        if kind == "error":
            self.query_one("#status-view", Static).update(message)
        self._refresh_status()

    def _write_chat(self, message: str, group: str = "") -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        if group and group != self._last_chat_group:
            chat_log.write("")
            self._chat_plain_lines.append("")
        chat_log.write(message)
        self._chat_plain_lines.append(self._strip_markup(message))
        self._last_chat_group = group or self._last_chat_group

    def _refresh_status(self) -> None:
        images = ", ".join(self._active_images) if self._active_images else "none"
        bar = (
            f"workspace: {self._agent._config.workspace} | "
            f"server: {self._agent._config.server_url} | "
            f"compactions: {self._session.compactions} | "
            f"images: {images}"
        )
        self.query_one("#status-bar", Static).update(bar)
        self.query_one("#status-view", Static).update(
            "\n".join(
                [
                    f"Workspace: {self._agent._config.workspace}",
                    f"Server: {self._agent._config.server_url}",
                    f"Context window: {self._agent._config.context_window}",
                    f"Soft limit ratio: {self._agent._config.context_soft_limit_ratio}",
                    f"Compactions: {self._session.compactions}",
                    f"Recent history messages: {len(self._session.recent_history)}",
                    f"Active images: {images}",
                ]
            )
        )
        self.query_one("#context-view", Static).update(self._session.summary or "[no working-memory summary yet]")

    @staticmethod
    def _tool_summary_for_chat(message: str, *, started: bool) -> str:
        lines = message.splitlines()
        tool = next((line.split(": ", 1)[1] for line in lines if line.startswith("tool: ")), "tool")
        command = next((line.split(": ", 1)[1] for line in lines if line.startswith("command: ")), "")
        path = next((line.split(": ", 1)[1] for line in lines if line.startswith("path: ")), "")
        result_line = next((line for line in lines if line.startswith("{")), "")
        if tool == "run_command":
            if started:
                return f"[b]Command[/b] {command}"
            exit_code = "?"
            if '"exit_code":' in message:
                exit_code = message.split('"exit_code":', 1)[1].split(",", 1)[0].strip()
            return f"[b]Command done[/b] {command} (exit {exit_code})"
        if tool == "read_file":
            return f"[b]Read file[/b] {path}"
        if tool == "list_dir":
            return f"[b]List dir[/b] {path}"
        if tool == "write_file":
            return f"[b]Wrote file[/b] {path}"
        if tool == "replace_in_file":
            return f"[b]Changed file[/b] {path}"
        if tool == "make_dir":
            return f"[b]Created dir[/b] {path}"
        return f"[b]Tool[/b] {tool}"

    @staticmethod
    def _strip_markup(message: str) -> str:
        return (
            message.replace("[b]", "")
            .replace("[/b]", "")
            .replace("[you]", "you:")
            .replace("[tool]", "tool:")
        )
