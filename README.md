# AlmaCode

AlmaCode is a local coding agent split into two parts:

- `almacode`: a single CLI binary that behaves like a coding agent and talks to a running `llama-server`
- `almacode-server`: a Linux-first wrapper/installer package that builds and manages `llama-server` for local GGUF inference

The client no longer loads GGUF models directly and has no runtime dependency on `llama-cpp-python`.

## Architecture

### Client

- one-file CLI binary via PyInstaller
- handles tool use, shell access, workspace file operations, and chat UX
- sends OpenAI-compatible requests to `llama-server`
- supports text and multimodal prompts through `messages`

### Server

- Linux x86_64 package distributed as `almacode-server-linux-x86_64.zip`
- installs a local wrapper and configuration under `~/.local/share/almacode-server` by default
- builds `llama.cpp/llama-server` from source for CUDA 12.8
- owns model selection, `mmproj`, GPU layout, and runtime flags

## Client quick start

1. Install the project for development:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

2. Configure the client with one of:

- `--server-url http://127.0.0.1:8080`
- `ALMACODE_SERVER_URL=http://127.0.0.1:8080`
- `~/.config/almacode/client.json`

Example config:

```json
{
  "server_url": "http://127.0.0.1:8080",
  "api_key": null,
  "request_timeout": 120
}
```

3. Run the client:

```bash
almacode run \
  --server-host 127.0.0.1 \
  --server-port 8080 \
  --workspace . \
  "Inspect the repo and write a tiny hello world example."
```

4. Start interactive chat:

```bash
almacode chat --server-host 127.0.0.1 --server-port 8080 --workspace .
```

`almacode chat` now starts a Textual TUI by default. Use `almacode chat --plain` for the legacy line-by-line mode.

## Context compaction

The client now auto-compacts long sessions before they hit the server context limit:

- it keeps a short working-memory summary
- it preserves only a small recent tail of messages
- if the context is still too large, it falls back to a minimal dev-log summary and continues

This prevents long coding sessions from crashing on `exceed_context_size_error`.

## Multimodal client usage

Attach images with `--image`:

```bash
almacode run \
  --server-url http://127.0.0.1:8080 \
  --image ./screenshot.png \
  "Read the screenshot and explain the error."
```

In `chat` mode:

- `/image path1 path2` sets active images
- `/clear-images` clears them

## Deprecated local model flags

The client still parses old local-runtime flags for migration, but using them now fails fast with a migration message:

- `--model`
- `--mmproj`
- `--mm-handler`
- `--backend`
- `--llama-server-binary`
- `--chat-format`
- `--n-gpu-layers`

Those values now belong in the server config, not in the client process.

## Server package

Build the server installer zip:

```bash
python scripts/build_server_package.py
```

Install it:

```bash
cd server_package
./install.sh
```

Default install location:

```text
~/.local/share/almacode-server
```

Main commands:

```bash
~/.local/share/almacode-server/bin/almacode-server doctor
~/.local/share/almacode-server/bin/almacode-server configure --write-default
~/.local/share/almacode-server/bin/almacode-server configure --set model.model_path=/models/model.gguf
~/.local/share/almacode-server/bin/almacode-server install
~/.local/share/almacode-server/bin/almacode-server start
~/.local/share/almacode-server/bin/almacode-server status
~/.local/share/almacode-server/bin/almacode-server logs
~/.local/share/almacode-server/bin/almacode-server stop
```

## Server config

The server config lives at:

```text
<server-home>/config/server.json
```

It contains:

- `server.host`, `server.port`
- `model.model_path`, `model.mmproj_path`, `model.chat_template`
- `runtime.ctx_size`, `runtime.gpu_layers`, `runtime.threads`, `runtime.threads_batch`, `runtime.batch_size`, `runtime.ubatch_size`, `runtime.flash_attn`
- `multi_gpu.tensor_split`, `multi_gpu.row_split`
- `advanced.extra_flags`, `advanced.cache_type_k`, `advanced.cache_type_v`, `advanced.numa`, `advanced.no_mmap`, `advanced.mlock`

## CUDA 12.8 and multi-GPU

The server package is Linux-first and builds `llama-server` from `llama.cpp` source instead of relying on Python wheels.

Required for `almacode-server doctor` / `install`:

- Linux x86_64
- `python3`
- `git`
- `cmake`
- `ninja` or `make`
- `nvcc`
- CUDA toolkit `12.8`
- NVIDIA driver / `nvidia-smi`

Multi-GPU configuration follows `llama-server` style flags inspired by `text-generation-webui`:

- `tensor_split`: comma-separated proportions like `60,40`
- `row_split`: when enabled, renders `--split-mode row`
- `gpu_layers=-1`: recommended default for full offload when supported

## Development

Run tests:

```bash
pytest
```

Build the client binary:

```bash
python scripts/build_binary.py
```

Build the server installer zip:

```bash
python scripts/build_server_package.py
```

## CI/CD

### `dev-build`

- runs tests
- builds the client binary
- assembles the Linux server installer zip

### `release-build`

- builds client binaries for Linux, macOS, and Windows
- builds `almacode-server-linux-x86_64.zip` on Linux
- publishes all artifacts to GitHub Releases on version tags

## Sources

- `text-generation-webui` `llama_cpp_server.py` for `llama-server` launch/config ideas
- `llama.cpp` official repository for `llama-server`
- GitHub Docs for protected branches and required status checks

## License

- This product created and published by Alberto Genuardy (MIT)
