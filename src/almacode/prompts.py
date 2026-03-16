from __future__ import annotations

import json
from textwrap import dedent


def build_system_prompt(workspace: str, command_timeout: int, system_note: str = "") -> str:
    note_block = ""
    if system_note.strip():
        note_block = f"\nAdditional operator note:\n{system_note.strip()}\n"

    return dedent(
        f"""
        You are AlmaCode, a local coding agent working in this workspace: {workspace}
        You may use tools to inspect files, edit files, run shell commands, search the web, and open web pages.
        Reply with JSON only. No Markdown. No prose outside JSON.

        Core behavior:
        - Be direct and incremental.
        - Prefer one useful tool call at a time.
        - Read tool results carefully before the next step.
        - Do not repeat the same failed or unhelpful step.
        - Finish with final_answer as soon as the task is done or clearly blocked.

        Shell rules:
        - Commands run inside the workspace and time out after {command_timeout} seconds unless you request less.
        - Each run_command call starts a fresh shell. Shell state does not persist.
        - If you need a venv, do activation and install in one command or call that venv's python/pip directly.
        - For run_command, exit_code is the source of truth.
        - stderr with exit_code 0 usually means warnings or notices, not failure.

        Reply with JSON only. Do not wrap the JSON in Markdown.
        Use this schema:
        {{
          "thought": "one short sentence about what you are doing",
          "action": {{
            "tool": "list_dir | read_file | write_file | replace_in_file | make_dir | run_command | web_search | open_url | complete_step | final_answer",
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
        - web_search args: {{"query": "latest python packaging guide", "limit": 5}}
        - open_url args: {{"url": "https://example.com", "max_chars": 12000}}
        - complete_step args: {{"summary": "what was finished for the current plan step"}}
        - final_answer args: {{"answer": "what you completed, what changed, and any important caveats"}}

        Rules:
        - Paths must stay inside the workspace root.
        - Never claim a tool ran unless you actually requested it.
        - When a tool fails, inspect the result and recover.
        - If you edit a file, prefer reading it first unless it does not exist yet.
        {note_block}
        """
    ).strip()


def build_plan_prompt(task: str, workspace: str, research_summary: str = "") -> str:
    research_block = research_summary.strip() or "[none]"
    return dedent(
        f"""
        Break the user task into a short execution plan for a local coding agent.

        Workspace: {workspace}
        Task: {task}
        Research summary:
        {research_block}

        Return JSON only with this schema:
        {{
          "steps": [
            {{"title": "short step title", "details": "one sentence"}},
            {{"title": "short step title", "details": "one sentence"}}
          ]
        }}

        Rules:
        - First classify the user request:
          - task_request: the user wants work to be done
          - conversational: the user is greeting, thanking, or making light conversation
        - For task_request, return 1 to 10 concrete ordered steps.
        - For conversational, return 1 or 2 light steps such as "greet user" or "check environment, then greet user" only if useful.
        - Steps must be concrete and executable.
        - Steps should be ordered.
        - Avoid redundant inspection steps.
        - If setup is needed, include it as one step.
        """
    ).strip()


def build_step_prompt(
    *,
    task: str,
    step_index: int,
    total_steps: int,
    step_title: str,
    step_details: str,
    completed_steps: list[str],
    research_summary: str,
) -> str:
    completed_block = "\n".join(completed_steps) if completed_steps else "[none]"
    research_block = research_summary.strip() or "[none]"
    return dedent(
        f"""
        Overall task: {task}
        Current plan step: {step_index}/{total_steps}
        Step title: {step_title}
        Step details: {step_details or '[none]'}

        Completed plan steps:
        {completed_block}

        Research summary:
        {research_block}

        Work only on the current plan step.
        Use tools if needed.
        When this step is complete, respond with tool=complete_step and a short summary.
        Do not return final_answer until all plan steps are complete or the task is blocked.
        """
    ).strip()


def build_research_summary_prompt(task: str, notes: str) -> str:
    return dedent(
        f"""
        Summarize the external research notes for a coding agent.

        Task:
        {task}

        Notes:
        {notes}

        Return a concise plain-text summary with:
        - useful facts
        - likely solution direction
        - warnings or caveats
        """
    ).strip()


def build_tool_feedback(tool_name: str, payload: str) -> str:
    if tool_name == "run_command":
        command_feedback = _format_run_command_feedback(payload)
        return dedent(
            f"""
            Tool result for `{tool_name}`:
            {command_feedback}
            """
        ).strip()

    return dedent(
        f"""
        Tool result for `{tool_name}`:
        {payload}
        """
    ).strip()


def _format_run_command_feedback(payload: str) -> str:
    try:
        command_result = json.loads(payload)
    except json.JSONDecodeError:
        return payload

    exit_code = command_result.get("exit_code", "?")
    success = exit_code == 0
    command = str(command_result.get("command", ""))
    cwd = str(command_result.get("cwd", "."))
    virtual_env = command_result.get("virtual_env")
    stdout = str(command_result.get("stdout", "")).strip()
    stderr = str(command_result.get("stderr", "")).strip()

    lines = [
        f"status: {'success' if success else 'failure'}",
        f"exit_code: {exit_code}",
        f"command: {command}",
        f"cwd: {cwd}",
    ]
    if virtual_env:
        lines.append(f"virtual_env: {virtual_env}")
    if success:
        lines.append("interpretation: the command completed successfully")
        if stderr:
            lines.append("stderr_note: stderr contains warnings/notices but the command still succeeded")
    else:
        lines.append("interpretation: the command failed and needs follow-up")

    lines.append("stdout:")
    lines.append(stdout or "[empty]")
    lines.append("stderr:")
    lines.append(stderr or "[empty]")
    lines.append("raw_json:")
    lines.append(payload)
    return "\n".join(lines)


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
