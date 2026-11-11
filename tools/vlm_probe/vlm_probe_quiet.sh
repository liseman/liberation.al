#!/bin/zsh
# macOS only (MLX): stops the vlm service so the probe has the GPU to itself, runs the probe, then starts it again.
# The application runs that service (pack-app, or pack-supervisor run without a window). With the application
# closed and no per-service unit, nothing here is holding the GPU, so the probe just runs.
cd "$(dirname "$0")/../.."
held=0
# status prints one line per service the application, or an older unit, is running. No vlm line means nothing to stop.
if uv run pack-services status | grep -q '^vlm '; then
  uv run pack-services stop vlm
  held=1
  sleep 3
fi
restart() {
  if (( held )); then
    uv run pack-services start vlm
  fi
}
trap restart EXIT
uv run tools/vlm_probe/vlm_probe.py mlx-community/Qwen3-VL-4B-Instruct-4bit 448,336 2>&1 | grep -v -i 'warn\|Fetching'
uv run tools/vlm_probe/vlm_probe.py mlx-community/Qwen3-VL-2B-Instruct-4bit 448 2>&1 | grep -v -i 'warn\|Fetching'
