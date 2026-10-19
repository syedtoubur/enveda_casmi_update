S=$1
rm -f /tmp/drive_go ~/drive_url.txt
( for i in $(seq 1 1200); do [ -f /tmp/drive_go ] && break; sleep 1; done; echo ) | ~/.local/bin/colab drivemount -s $S 2>&1 | grep -v "│" | tail -3
tail -n 1 ~/.config/colab-cli/history/$S.jsonl | cut -c1-200
