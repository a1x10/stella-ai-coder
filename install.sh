#!/usr/bin/env sh
set -eu

REPO_RAW="${STELLA_REPO_RAW:-https://raw.githubusercontent.com/a1x10/stella-ai-coder/main}"
INSTALL_DIR="${HOME}/.stella-ai-coder"
VENV_DIR="${INSTALL_DIR}/.venv"
MODEL="${STELLA_MODEL:-qwen2.5-coder:1.5b}"
BIN_DIR="${HOME}/.local/bin"

printf "\n=== Stella AI Coder installer ===\n"
printf "Install dir: %s\n" "$INSTALL_DIR"
printf "Model: %s\n\n" "$MODEL"

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  name="$1"
  printf "Downloading %s\n" "$name"
  if need_cmd curl; then
    curl -fsSL "${REPO_RAW}/${name}" -o "${INSTALL_DIR}/${name}"
  elif need_cmd wget; then
    wget -q "${REPO_RAW}/${name}" -O "${INSTALL_DIR}/${name}"
  else
    echo "curl or wget is required."
    exit 1
  fi
}

if ! need_cmd python3; then
  echo "Python 3.10+ is required. Install Python and run this command again."
  exit 1
fi

if ! need_cmd ollama; then
  echo "Ollama was not found."
  printf "Install Ollama now using the official installer? Type y to continue: "
  read ans
  case "$ans" in
    y|Y) curl -fsSL https://ollama.com/install.sh | sh ;;
    *) echo "Install Ollama from https://ollama.com/download and run again."; exit 1 ;;
  esac
fi

download_file "stella_ai_coder.py"
download_file "requirements.txt"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

"${VENV_DIR}/bin/python" -m pip install -U pip
"${VENV_DIR}/bin/python" -m pip install -r "${INSTALL_DIR}/requirements.txt"

if ! curl -fsSL http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama in background"
  nohup ollama serve >/tmp/stella-ollama.log 2>&1 &
  sleep 5
fi

ollama pull "$MODEL"

cat > "${BIN_DIR}/stella" <<EOF
#!/usr/bin/env sh
export STELLA_MODEL="${MODEL}"
exec "${VENV_DIR}/bin/python" "${INSTALL_DIR}/stella_ai_coder.py" "\$@"
EOF
chmod +x "${BIN_DIR}/stella"

printf "\nStella is installed.\n"
printf "Run anytime: stella\n"
printf "If your shell cannot find it, add this to PATH: %s\n\n" "$BIN_DIR"
exec "${BIN_DIR}/stella"
