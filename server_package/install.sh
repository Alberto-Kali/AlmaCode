#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/share/almacode-server}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$INSTALL_DIR"
VENV_DIR="$RUNTIME_DIR/.venv"
BIN_DIR="$RUNTIME_DIR/bin"

mkdir -p "$RUNTIME_DIR"
cp -R "$PACKAGE_DIR/src" "$RUNTIME_DIR/"
cp "$PACKAGE_DIR/README-server.md" "$RUNTIME_DIR/"
mkdir -p "$RUNTIME_DIR/config"
cp "$PACKAGE_DIR/config/server.json" "$RUNTIME_DIR/config/server.json"
mkdir -p "$BIN_DIR"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

cat > "$BIN_DIR/almacode-server" <<EOF
#!/usr/bin/env bash
export ALMACODE_SERVER_HOME="$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_DIR/src"
exec "$VENV_DIR/bin/python" -m almacode_server.cli "\$@"
EOF

chmod +x "$BIN_DIR/almacode-server"

echo "Installed AlmaCode server wrapper to $RUNTIME_DIR"
echo "Use: $BIN_DIR/almacode-server doctor"
