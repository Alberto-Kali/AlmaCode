# AlmaCode

AlmaCode is a local coding agent for GGUF models. It runs entirely on your machine with `llama.cpp` via `llama-cpp-python`, can inspect and edit files in a workspace, and can execute terminal commands in a controlled loop similar to Claude Code style agents.

## Features

- Runs local GGUF models from disk through `llama-cpp-python`
- Supports single-shot tasks and an interactive chat mode
- Gives the model access to:
  - directory listing
  - file reads
  - file writes and appends
  - targeted string replacements
  - shell command execution with timeouts
- Restricts file operations to the chosen workspace root
- Packages into a standalone binary with PyInstaller
- Includes CI for `dev` and release builds for `release`

## Quick start

1. Create a virtual environment with Python 3.10-3.12.
2. Install the project:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

3. Run the agent with a local GGUF model:

```bash
almacode run \
  --model /models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  "Open the project, inspect the failing tests, and fix them."
```

4. Start an interactive session:

```bash
almacode chat \
  --model /models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --workspace .
```

## Recommended models

Instruction-tuned coding models with a GGUF chat template work best. Good starting points:

- Qwen2.5-Coder Instruct GGUF
- DeepSeek Coder Instruct GGUF
- Codestral GGUF variants with a matching prompt template

## Installation notes

`llama-cpp-python` can run CPU-only out of the box, but it also supports backend-specific acceleration. The upstream README documents current install flags such as:

- OpenBLAS on CPU via `CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"`
- CUDA via `CMAKE_ARGS="-DGGML_CUDA=on"`
- Metal via `CMAKE_ARGS="-DGGML_METAL=on"`

Examples from the official docs:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

The project CI uses Python 3.12 because the upstream project documents prebuilt acceleration wheels for Python 3.10-3.12.

CI uses the official extra index URLs from the upstream project where that helps:

- CPU wheels: `https://abetlen.github.io/llama-cpp-python/whl/cpu`
- Metal wheels: `https://abetlen.github.io/llama-cpp-python/whl/metal`

For Linux release artifacts, the project intentionally builds `llama-cpp-python` from source in CI instead of reusing the generic CPU wheel. That keeps the bundled `libllama.so` linked against the runner's normal glibc toolchain and avoids runtime failures caused by musl-linked binaries on standard desktop distributions.

## CLI

### `almacode run`

Runs one autonomous task and exits after the model returns a final answer or the step budget is exhausted.

```bash
almacode run --model /models/model.gguf --workspace . "Refactor src/ and explain what changed."
```

### `almacode chat`

Starts an interactive loop. Each new user prompt includes the previous high-level conversation but does not replay the full low-level tool trace.

```bash
almacode chat --model /models/model.gguf --chat-format chatml
```

### Useful flags

- `--workspace`: restrict file access and shell working directories to this root
- `--chat-format`: force a prompt format if the GGUF metadata is missing or wrong
- `--n-ctx`: context window
- `--n-gpu-layers`: number of layers offloaded to GPU, `-1` for all supported layers
- `--max-steps`: upper bound on tool-use turns
- `--command-timeout`: timeout in seconds for shell commands
- `--verbose`: show the raw model JSON for debugging

## Branching model

- `dev`: default working branch, protected, runs build verification and tests
- `release`: protected stabilization branch, runs cross-platform binary builds

Suggested flow:

1. Work on feature branches from `dev`
2. Merge into `dev` after the `dev-build` check passes
3. Merge `dev` into `release` when you want a releasable state
4. Tag a release commit on `release` with `vX.Y.Z` to publish binaries to GitHub Releases

## GitHub automation

### `dev-build`

- Triggers on pushes and pull requests targeting `dev`
- Installs the package
- Runs unit tests
- Builds the one-file binary as a smoke test

### `release-build`

- Triggers on pushes to `release`, release tags `v*`, and manual dispatch
- Builds binaries for Linux, macOS, and Windows
- Uploads workflow artifacts on branch builds
- Publishes binaries to GitHub Releases on version tags

## Branch protection

This repo is configured for solo development with guardrails but without locking the owner out:

- `dev` requires the `dev-build` status check
- `release` requires all three release build checks
- force pushes and branch deletion are disabled
- linear history is required
- admin enforcement is intentionally disabled so the repo owner can bypass in emergencies

You can re-apply the GitHub settings with:

```bash
python scripts/configure_github.py --repo Alberto-Kali/AlmaCode
```

## Development

Run tests:

```bash
pytest
```

Build a local binary:

```bash
python scripts/build_binary.py
```

## Sources

- `llama-cpp-python` official README: chat completions, JSON mode, hardware backends
- `llama.cpp` official README: GGUF requirement and local model usage
- GitHub Docs: protected branches and required status checks
