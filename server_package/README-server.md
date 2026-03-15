# AlmaCode Server

This package installs the AlmaCode server wrapper for Linux x86_64.

## What it does

- creates an isolated runtime directory
- writes a default `config/server.json`
- exposes `almacode-server` commands through a local wrapper script
- builds `llama-server` from `llama.cpp` source during `almacode-server install`

## Quick start

```bash
./install.sh
~/.local/share/almacode-server/bin/almacode-server doctor
~/.local/share/almacode-server/bin/almacode-server configure --write-default
~/.local/share/almacode-server/bin/almacode-server install
~/.local/share/almacode-server/bin/almacode-server start
```

## CUDA

The wrapper expects a local CUDA 12.8 toolkit and builds `llama-server` from source.
