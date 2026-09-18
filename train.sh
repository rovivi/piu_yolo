#!/bin/bash
# Lanzador portable: usa venv si existe, si no conda (env piu_yolo), si no python3.
# Uso:  ./train.sh              (entrenar)
#       ./train.sh --resume     (continuar el último run interrumpido)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--resume" ]; then
  shift; SCRIPT=continue.py
else
  SCRIPT=training.py
fi

if [ -d venv ]; then
  source venv/bin/activate
  exec python "$SCRIPT" "$@"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^piu_yolo '; then
  exec conda run --no-capture-output -n piu_yolo python "$SCRIPT" "$@"
else
  exec python3 "$SCRIPT" "$@"
fi
