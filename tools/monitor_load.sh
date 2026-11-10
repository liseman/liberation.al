#!/bin/zsh
# macOS only (reads Apple GPU utilisation from ioreg).
# Sample CPU% / RSS of a pid plus GPU utilisation (ioreg AGX PerformanceStatistics) every N s. Usage: monitor_load.sh PID [interval] >> log
pid=$1; iv=${2:-5}
while kill -0 $pid 2>/dev/null; do
  cpu=$(ps -o %cpu= -p $pid | tr -d ' '); rss=$(ps -o rss= -p $pid | tr -d ' ')
  gpu=$(ioreg -r -d 1 -c AGXAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$')
  echo "$(date +%H:%M:%S) cpu=${cpu}% rss=$((rss/1024))MB gpu=${gpu:-?}%"
  sleep $iv
done
