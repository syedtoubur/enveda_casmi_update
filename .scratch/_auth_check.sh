f=$(ls -t ~/.config/colab-cli/history/*.jsonl | head -1); echo $f
tail -n 6 $f | cut -c1-220
~/.local/bin/colab sessions; ~/.local/bin/colab usage
