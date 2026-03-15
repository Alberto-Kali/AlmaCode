from __future__ import annotations

from textwrap import dedent


def build_system_prompt(workspace: str, command_timeout: int, system_note: str = "") -> str:
    note_block = ""
    if system_note.strip():
        note_block = f"\nAdditional operator note:\n{system_note.strip()}\n"

    return dedent(
        f"""
        You are AlmaCode, a local software engineering agent.
        You work inside the workspace rooted at: {workspace}
        You can inspect files, edit files, list directories, create folders, replace text, and run shell commands.
        Never claim to have run a tool if you have not actually requested it.
        Prefer inspecting files before editing them.
        Keep tool requests focused and incremental.
        Shell commands are limited to the workspace and use a timeout of {command_timeout} seconds unless you request a lower timeout.

        Reply with JSON only. Do not wrap the JSON in Markdown.
        Use this schema:
        {{
          "thought": "one short sentence about what you are doing",
          "action": {{
            "tool": "list_dir | read_file | write_file | replace_in_file | make_dir | run_command | final_answer",
            "args": {{
              "...": "tool-specific arguments"
            }}
          }}
        }}

        Tool reference:
        - list_dir args: {{"path": "relative/path"}}
        - read_file args: {{"path": "relative/path", "start_line": 1, "max_lines": 200}}
        - write_file args: {{"path": "relative/path", "content": "full file text", "append": false}}
        - replace_in_file args: {{"path": "relative/path", "old": "text to replace", "new": "replacement", "count": 0}}
        - make_dir args: {{"path": "relative/path"}}
        - run_command args: {{"command": "pytest -q", "cwd": ".", "timeout": 60}}
        - final_answer args: {{"answer": "what you completed, what changed, and any important caveats"}}

        Rules:
        - Paths must stay inside the workspace root.
        - When a tool fails, inspect the error and recover.
        - Do not emit a final answer until the task is actually complete or you are blocked.
        - If you edit a file, prefer reading it first unless the file does not exist yet.
        {note_block}
        """
    ).strip()


def build_tool_feedback(tool_name: str, payload: str) -> str:
    return dedent(
        f"""
        Tool result for `{tool_name}`:
        {payload}
        """
    ).strip()


def build_summary_prompt(
    workspace: str,
    current_task: str,
    summary: str,
    recent_history: list[dict[str, object]],
    tool_log: list[str],
) -> str:
    history_lines: list[str] = []
    for message in recent_history[-8:]:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        history_lines.append(f"{role}: {content[:1200]}")

    joined_history = "\n".join(history_lines) if history_lines else "[no recent history]"
    joined_tool_log = "\n".join(tool_log[-12:]) if tool_log else "[no tool log]"
    summary_block = summary.strip() if summary.strip() else "[empty summary]"

    return dedent(
        f"""
        You are compressing the working memory of AlmaCode so a coding session can continue inside a small context window.

        Workspace: {workspace}
        Current task: {current_task}

        Existing summary:
        {summary_block}

        Recent conversation tail:
        {joined_history}

        Recent tool log:
        {joined_tool_log}

        Write a short structured memory summary that preserves only the facts needed to continue the task.
        Include:
        - goal
        - what was already done
        - files or artifacts that matter
        - decisions already made
        - open work
        - important failures or caveats

        Keep it concise. Omit verbose tool output. Do not use Markdown code fences.
        """
    ).strip()
