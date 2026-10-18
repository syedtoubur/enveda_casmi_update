import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr to prevent charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    prompt_file = Path(r"e:\ENVEDA_casmi\research\c3\opus_cross_val_prompt.txt")
    if not prompt_file.exists():
        print(f"Error: {prompt_file} does not exist.")
        sys.exit(1)

    prompt = prompt_file.read_text(encoding="utf-8")
    log_file = Path(r"e:\ENVEDA_casmi\research\c3\opus_audit_result.md")

    print(f"Starting headless Claude Opus audit...")
    print(f"Prompt length: {len(prompt)} characters.")

    cmd = [
        "claude",
        "-p", prompt,
        "--model", "opus",
        "--permission-mode", "dontAsk"
    ]

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    with open(log_file, "w", encoding="utf-8") as out:
        p = subprocess.Popen(
            cmd,
            cwd=r"e:\ENVEDA_casmi",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        for line in p.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            out.write(line)
            out.flush()
        rc = p.wait()

    print(f"\nClaude Opus audit finished with return code {rc}.")
    sys.exit(rc)

if __name__ == "__main__":
    main()
