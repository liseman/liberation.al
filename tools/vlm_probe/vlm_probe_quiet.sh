#!/bin/zsh
# macOS only (MLX): stops the vision service so the probe has the GPU to itself, runs the probe, then restarts it.
cd "$(dirname "$0")/../.."
U=$(id -u)
restart() { launchctl bootstrap gui/$U ~/Library/LaunchAgents/pack.vision.plist 2>&1; echo "restarted: $(launchctl print gui/$U/pack.vision | grep -E 'pid|job state')"; }
trap restart EXIT
launchctl bootout gui/$U/pack.vision; sleep 3
uv run tools/vlm_probe/vlm_probe.py mlx-community/Qwen3-VL-4B-Instruct-4bit 448,336 2>&1 | grep -v -i 'warn\|Fetching'
uv run tools/vlm_probe/vlm_probe.py mlx-community/Qwen3-VL-2B-Instruct-4bit 448 2>&1 | grep -v -i 'warn\|Fetching'
