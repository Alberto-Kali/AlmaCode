from __future__ import annotations

import argparse
import json
import subprocess


def gh_api(method: str, path: str, payload: dict | None = None) -> None:
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command.extend(["--input", "-"])
        body = json.dumps(payload).encode("utf-8")
    else:
        body = None

    completed = subprocess.run(command, input=body, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def set_default_branch(repo: str) -> None:
    gh_api("PATCH", f"repos/{repo}", {"default_branch": "dev"})


def protect_branch(repo: str, branch: str, contexts: list[str]) -> None:
    payload = {
        "required_status_checks": {
            "strict": True,
            "contexts": contexts,
        },
        "enforce_admins": False,
        "required_pull_request_reviews": None,
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": False,
        "lock_branch": False,
        "allow_fork_syncing": True,
    }
    gh_api("PUT", f"repos/{repo}/branches/{branch}/protection", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure GitHub branch defaults and protection")
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    args = parser.parse_args()

    set_default_branch(args.repo)
    protect_branch(args.repo, "dev", ["dev-build"])
    protect_branch(args.repo, "release", ["release-linux", "release-macos", "release-windows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

